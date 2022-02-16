import math
import shutil
from pathlib import Path

import torch
import hydra
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from torch_ema import ExponentialMovingAverage
from torchvision.utils import save_image
from cleanfid import fid

from dataset.mesh_real_features import FaceGraphMeshDataset
from dataset import to_vertex_colors_scatter, GraphDataLoader, to_device
from model.augment import AugmentPipe
from model.differentiable_renderer import DifferentiableRenderer
from model.graph import TwinGraphEncoder
from model.graph_generator_u_deep import Generator
from model.discriminator import Discriminator
from model.loss import PathLengthPenalty, compute_gradient_penalty
from trainer import create_trainer
from util.timer import Timer

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False


class StyleGAN2Trainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.train_set = FaceGraphMeshDataset(config)
        self.val_set = FaceGraphMeshDataset(config, config.num_eval_images)
        self.G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, channel_base=config.g_channel_base, channel_max=config.g_channel_max)
        self.D = Discriminator(config.image_size, 3, w_num_layers=config.num_mapping_layers, mbstd_on=config.mbstd_on, channel_base=config.d_channel_base)
        self.E = TwinGraphEncoder(self.train_set.num_feats, 1)
        self.R = None
        self.augment_pipe = AugmentPipe(config.ada_start_p, config.ada_target, config.ada_interval, config.ada_fixed, config.batch_size, config.views_per_sample, config.colorspace)
        # print_module_summary(self.G, (torch.zeros(self.config.batch_size, self.config.latent_dim), ))
        # print_module_summary(self.D, (torch.zeros(self.config.batch_size, 3, config.image_size, config.image_size), ))
        self.grid_z = torch.randn(config.num_eval_images, self.config.latent_dim)

        self.automatic_optimization = False
        self.path_length_penalty = PathLengthPenalty(0.01, 2)
        self.ema = None

    def configure_optimizers(self):
        g_opt = torch.optim.Adam([
            {'params': list(self.G.parameters()), 'lr': self.config.lr_g, 'betas': (0.0, 0.99), 'eps': 1e-8},
            {'params': list(self.E.parameters()), 'lr': self.config.lr_e, 'eps': 1e-8, 'weight_decay': 1e-4}
        ])
        d_opt = torch.optim.Adam(self.D.parameters(), lr=self.config.lr_d, betas=(0.0, 0.99), eps=1e-8)
        return g_opt, d_opt

    def forward(self, batch, limit_batch_size=False):
        z = self.latent(limit_batch_size)
        w = self.get_mapped_latent(z, 0.9)
        fake = self.G.synthesis(batch['graph_data'], w, batch['shape'])
        return fake, w

    def g_step(self, batch):
        g_opt = self.optimizers()[0]
        g_opt.zero_grad(set_to_none=True)
        fake, w = self.forward(batch)
        p_fake = self.D(self.augment_pipe(self.render(fake, batch)))
        gen_loss = torch.nn.functional.softplus(-p_fake).mean()
        self.manual_backward(gen_loss)
        log_gen_loss = gen_loss.item()
        step(g_opt, self.G)
        self.log("G", log_gen_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

    def g_regularizer(self, batch):
        g_opt = self.optimizers()[0]
        for idx in range(len(batch['shape'])):
            batch['shape'][idx] = batch['shape'][idx].detach()
        g_opt.zero_grad(set_to_none=True)
        fake, w = self.forward(batch)
        plp = self.path_length_penalty(self.render(fake, batch), w)
        if not torch.isnan(plp):
            gen_loss = self.config.lambda_plp * plp * self.config.lazy_path_penalty_interval
            self.log("rPLP", plp, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.manual_backward(gen_loss)
            step(g_opt, self.G)

    def d_step(self, batch):
        d_opt = self.optimizers()[1]
        d_opt.zero_grad(set_to_none=True)

        fake, _ = self.forward(batch)
        p_fake = self.D(self.augment_pipe(self.render(fake.detach(), batch)))
        fake_loss = torch.nn.functional.softplus(p_fake).mean()
        self.manual_backward(fake_loss)

        p_real = self.D(self.augment_pipe(self.train_set.get_color_bg_real(batch)))
        self.augment_pipe.accumulate_real_sign(p_real.sign().detach())

        # Get discriminator loss
        real_loss = torch.nn.functional.softplus(-p_real).mean()
        self.manual_backward(real_loss)

        step(d_opt, self.D)

        self.log("D_real", real_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log("D_fake", fake_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        disc_loss = real_loss + fake_loss
        self.log("D", disc_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

    def d_regularizer(self, batch):
        d_opt = self.optimizers()[1]
        d_opt.zero_grad(set_to_none=True)
        image = self.train_set.get_color_bg_real(batch)
        image.requires_grad_()
        p_real = self.D(self.augment_pipe(image, True))
        gp = compute_gradient_penalty(image, p_real)
        disc_loss = self.config.lambda_gp * gp * self.config.lazy_gradient_penalty_interval
        self.manual_backward(disc_loss)
        step(d_opt, self.D)
        self.log("rGP", gp, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

    def render(self, face_colors, batch, use_bg_color=True):
        rendered_color = self.R.render(batch['vertices'], batch['indices'], to_vertex_colors_scatter(face_colors, batch), batch["ranges"].cpu(), batch['bg'] if use_bg_color else None, resolution=self.config.render_size)
        ret_val = rendered_color.permute((0, 3, 1, 2))
        if self.config.render_size != self.config.image_size:
            ret_val = torch.nn.functional.interpolate(ret_val, (self.config.image_size, self.config.image_size), mode='bilinear', align_corners=True)
        return ret_val

    def training_step(self, batch, batch_idx):
        self.set_shape_codes(batch)
        # optimize generator
        self.g_step(batch)

        if self.global_step > self.config.lazy_path_penalty_after and (self.global_step + 1) % self.config.lazy_path_penalty_interval == 0:
            self.g_regularizer(batch)

        # torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)

        self.ema.update(self.G.parameters())

        # optimize discriminator

        self.d_step(batch)

        if (self.global_step + 1) % self.config.lazy_gradient_penalty_interval == 0:
            self.d_regularizer(batch)

        self.execute_ada_heuristics()

    def execute_ada_heuristics(self):
        if (self.global_step + 1) % self.config.ada_interval == 0:
            self.augment_pipe.heuristic_update()
        self.log("aug_p", self.augment_pipe.p.item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        pass

    @rank_zero_only
    def validation_epoch_end(self, _val_step_outputs):
        (Path("runs") / self.config.experiment / "checkpoints").mkdir(exist_ok=True)
        torch.save(self.ema, Path("runs") / self.config.experiment / "checkpoints" / f"ema_{self.global_step:09d}.pth")
        with Timer("export_grid"):
            odir_real, odir_fake, odir_samples, odir_grid, odir_meshes = self.create_directories()
            self.export_grid("", odir_grid, None)
            self.ema.store(self.G.parameters())
            self.ema.copy_to([p for p in self.G.parameters() if p.requires_grad])
            self.export_grid("ema_", odir_grid, odir_fake)
            self.export_mesh(odir_meshes)
        with Timer("export_samples"):
            latents = self.grid_z.split(self.config.batch_size)
            for iter_idx, batch in enumerate(self.val_dataloader()):
                batch = to_device(batch, self.device)
                self.set_shape_codes(batch)
                shape = batch['shape']
                real_render = batch['real'].cpu()
                fake_render = self.render(self.G(batch['graph_data'], latents[iter_idx % len(latents)].to(self.device), shape, noise_mode='const'), batch, use_bg_color=False).cpu()
                real_render = self.train_set.cspace_convert_back(real_render)
                fake_render = self.train_set.cspace_convert_back(fake_render)
                save_image(real_render, odir_samples / f"real_{iter_idx}.jpg", value_range=(-1, 1), normalize=True)
                save_image(fake_render, odir_samples / f"fake_{iter_idx}.jpg", value_range=(-1, 1), normalize=True)
                for batch_idx in range(real_render.shape[0]):
                    save_image(real_render[batch_idx], odir_real / f"{iter_idx}_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)
        self.ema.restore([p for p in self.G.parameters() if p.requires_grad])
        fid_score = fid.compute_fid(str(odir_real), str(odir_fake), device=self.device, num_workers=0)
        print(f'FID: {fid_score:.3f}')
        kid_score = fid.compute_kid(str(odir_real), str(odir_fake), device=self.device, num_workers=0)
        print(f'KID: {kid_score:.3f}')
        self.log(f"fid", fid_score, on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True, sync_dist=True)
        self.log(f"kid", kid_score, on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True, sync_dist=True)
        shutil.rmtree(odir_real.parent)

    def get_mapped_latent(self, z, style_mixing_prob):
        if torch.rand(()).item() < style_mixing_prob:
            cross_over_point = int(torch.rand(()).item() * self.G.mapping.num_ws)
            w1 = self.G.mapping(z[0])[:, :cross_over_point, :]
            w2 = self.G.mapping(z[1], skip_w_avg_update=True)[:, cross_over_point:, :]
            return torch.cat((w1, w2), dim=1)
        else:
            w = self.G.mapping(z[0])
            return w

    def latent(self, limit_batch_size=False):
        batch_size = self.config.batch_size if not limit_batch_size else self.config.batch_size // self.path_length_penalty.pl_batch_shrink
        z1 = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        z2 = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        return z1, z2

    def set_shape_codes(self, batch):
        code = self.E(batch['x'], batch['graph_data']['ff2_maps'][0], batch['graph_data'])
        batch['shape'] = code

    def train_dataloader(self):
        return GraphDataLoader(self.train_set, self.config.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return GraphDataLoader(self.val_set, self.config.batch_size, shuffle=True, drop_last=True, num_workers=self.config.num_workers)

    def export_grid(self, prefix, output_dir_vis, output_dir_fid):
        (Path("runs") / self.config.experiment / "checkpoints").mkdir(exist_ok=True)
        torch.save(self.ema, Path("runs") / self.config.experiment / "checkpoints" / f"ema_{self.global_step:09d}.pth")
        vis_generated_images = []
        grid_loader = iter(GraphDataLoader(self.train_set, batch_size=self.config.batch_size))
        for iter_idx, z in enumerate(self.grid_z.split(self.config.batch_size)):
            z = z.to(self.device)
            eval_batch = to_device(next(grid_loader), self.device)
            self.set_shape_codes(eval_batch)
            fake = self.render(self.G(eval_batch['graph_data'], z, eval_batch['shape'], noise_mode='const'), eval_batch, use_bg_color=False).cpu()
            fake = self.train_set.cspace_convert_back(fake)
            if output_dir_fid is not None:
                for batch_idx in range(fake.shape[0]):
                    save_image(fake[batch_idx], output_dir_fid / f"{iter_idx}_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)
            if iter_idx < self.config.num_vis_images // self.config.batch_size:
                vis_generated_images.append(fake)
        torch.cuda.empty_cache()
        vis_generated_images = torch.cat(vis_generated_images, dim=0)
        save_image(vis_generated_images, output_dir_vis / f"{prefix}{self.global_step:06d}.png", nrow=int(math.sqrt(vis_generated_images.shape[0])), value_range=(-1, 1), normalize=True)

    def export_mesh(self, outdir):
        grid_loader = iter(GraphDataLoader(self.train_set, batch_size=self.config.batch_size, shuffle=True))
        for iter_idx, z in enumerate(self.grid_z.split(self.config.batch_size)):
            if iter_idx < self.config.num_vis_meshes // self.config.batch_size:
                z = z.to(self.device)
                eval_batch = to_device(next(grid_loader), self.device)
                self.set_shape_codes(eval_batch)
                generated_colors = torch.clamp(self.G(eval_batch['graph_data'], z, eval_batch['shape'], noise_mode='const'), -1, 1)
                generated_colors = self.train_set.cspace_convert_back(generated_colors) * 0.5 + 0.5
                for bidx in range(generated_colors.shape[0] // self.config.num_faces[0]):
                    self.train_set.export_mesh(eval_batch['name'][bidx],
                                               generated_colors[self.config.num_faces[0] * bidx: self.config.num_faces[0] * (bidx + 1)], outdir / f"{eval_batch['name'][bidx]}.obj")

    def create_directories(self):
        output_dir_fid_real = Path(f'runs/{self.config.experiment}/fid/real')
        output_dir_fid_fake = Path(f'runs/{self.config.experiment}/fid/fake')
        output_dir_samples = Path(f'runs/{self.config.experiment}/images/{self.global_step:06d}')
        output_dir_textures = Path(f'runs/{self.config.experiment}/textures/')
        output_dir_meshes = Path(f'runs/{self.config.experiment}/meshes//{self.global_step:06d}')
        for odir in [output_dir_fid_real, output_dir_fid_fake, output_dir_samples, output_dir_textures, output_dir_meshes]:
            odir.mkdir(exist_ok=True, parents=True)
        return output_dir_fid_real, output_dir_fid_fake, output_dir_samples, output_dir_textures, output_dir_meshes

    def on_train_start(self):
        if self.ema is None:
            self.ema = ExponentialMovingAverage(self.G.parameters(), 0.995)
        if self.R is None:
            self.R = DifferentiableRenderer(self.config.image_size, "bounds", self.config.colorspace)

    def on_validation_start(self):
        if self.ema is None:
            self.ema = ExponentialMovingAverage(self.G.parameters(), 0.995)
        if self.R is None:
            self.R = DifferentiableRenderer(self.config.image_size, "bounds", self.config.colorspace)


def step(opt, module):
    for param in module.parameters():
        if param.grad is not None:
            torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
    torch.nn.utils.clip_grad_norm_(module.parameters(), 1)
    opt.step()


@hydra.main(config_path='../config', config_name='stylegan2')
def main(config):
    trainer = create_trainer("StyleGAN23D", config)
    model = StyleGAN2Trainer(config)
    trainer.fit(model)


if __name__ == '__main__':
    main()
