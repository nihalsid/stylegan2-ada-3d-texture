"""Train pi-GAN. Supports distributed training."""

import argparse
import itertools
import os
from random import random

import numpy as np
import math

from collections import deque

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from dataset import GraphDataLoader, to_vertex_colors_scatter, to_device
from dataset.mesh_real_pigan import SDFGridDataset
from model.differentiable_renderer import DifferentiableRenderer
from model.pigan import fid_evaluation, siren

from tqdm import tqdm
from datetime import datetime
import copy

from torch_ema import ExponentialMovingAverage

from model.pigan import curriculums
from model.pigan import discriminators
from model.styleganvox import SDFEncoder
from util.misc import EasyDict

render_helper = DifferentiableRenderer(128, "bounds")


def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
    return z


def train(rank, opt):
    torch.manual_seed(0)

    device = torch.device(rank)

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)

    fixed_z = z_sampler((16, 256), device='cpu', dist=metadata['z_dist'])
    eval_loader = itertools.cycle(GraphDataLoader(SDFGridDataset(EasyDict(metadata)), fixed_z.shape[0], shuffle=True, pin_memory=False, drop_last=True, num_workers=0))

    scaler = torch.cuda.amp.GradScaler()

    if opt.load_dir != '':
        encoder = torch.load(os.path.join(opt.load_dir, 'encoder.pth'), map_location=device)
        generator = torch.load(os.path.join(opt.load_dir, 'generator.pth'), map_location=device)
        discriminator = torch.load(os.path.join(opt.load_dir, 'discriminator.pth'), map_location=device)
        ema = torch.load(os.path.join(opt.load_dir, 'ema.pth'), map_location=device)
        ema2 = torch.load(os.path.join(opt.load_dir, 'ema2.pth'), map_location=device)
    else:
        encoder = SDFEncoder(1).to(device)
        generator = getattr(siren, metadata['model'])(3, z_dim=metadata['latent_dim'], hidden_dim=512, shape_dim=256).to(device)
        discriminator = getattr(discriminators, metadata['discriminator'])().to(device)
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)

    if metadata.get('unique_lr', False):
        mapping_network_param_names = [name for name, _ in generator.mapping_network.named_parameters()]
        mapping_network_parameters = [p for n, p in generator.named_parameters() if n in mapping_network_param_names]
        generator_parameters = [p for n, p in generator.named_parameters() if n not in mapping_network_param_names]
        optimizer_G = torch.optim.Adam([{'params': generator_parameters, 'name': 'generator'},
                                        {'params': mapping_network_parameters, 'name': 'mapping_network', 'lr': metadata['gen_lr'] * 5e-2},
                                        {'params': encoder.parameters(), 'name': 'encoder', 'lr': metadata['enc_lr']}],
                                       lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])
    else:
        optimizer_G = torch.optim.Adam([{'params': generator.parameters(), 'name': 'generator'},
                                        {'params': encoder.parameters(), 'name': 'encoder', 'lr': metadata['enc_lr']}],
                                       lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=metadata['disc_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    if opt.load_dir != '':
        optimizer_G.load_state_dict(torch.load(os.path.join(opt.load_dir, 'optimizer_G.pth')))
        optimizer_D.load_state_dict(torch.load(os.path.join(opt.load_dir, 'optimizer_D.pth')))
        if not metadata.get('disable_scaler', False):
            scaler.load_state_dict(torch.load(os.path.join(opt.load_dir, 'scaler.pth')))

    generator_losses = []
    discriminator_losses = []

    if opt.set_step != None:
        discriminator.step = opt.set_step

    if metadata.get('disable_scaler', False):
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    # ----------
    #  Training
    # ----------

    with open(os.path.join(opt.output_dir, 'options.txt'), 'w') as f:
        f.write(str(opt))
        f.write('\n\n')
        f.write(str(generator))
        f.write('\n\n')
        f.write(str(encoder))
        f.write('\n\n')
        f.write(str(discriminator))
        f.write('\n\n')
        f.write(str(curriculum))

    torch.manual_seed(rank)
    dataloader = None
    total_progress_bar = tqdm(total=opt.n_epochs, desc="Total progress", dynamic_ncols=True)
    total_progress_bar.update(discriminator.epoch)
    interior_step_bar = tqdm(dynamic_ncols=True)

    for _ in range(opt.n_epochs):
        total_progress_bar.update(1)

        metadata = curriculums.extract_metadata(curriculum, discriminator.step)

        # Set learning rates
        for param_group in optimizer_G.param_groups:
            if param_group.get('name', None) == 'mapping_network':
                param_group['lr'] = metadata['gen_lr'] * 5e-2
            elif param_group.get('name', None) == 'generator':
                param_group['lr'] = metadata['gen_lr']
            elif param_group.get('name', None) == 'encoder':
                param_group['lr'] = metadata['enc_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = metadata['disc_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']

        if not dataloader or dataloader.batch_size != metadata['batch_size']:
            dataloader = GraphDataLoader(SDFGridDataset(EasyDict(metadata)), metadata['batch_size'], shuffle=True, pin_memory=True, drop_last=True, num_workers=metadata['num_workers'])

            step_next_upsample = curriculums.next_upsample_step(curriculum, discriminator.step)
            step_last_upsample = curriculums.last_upsample_step(curriculum, discriminator.step)

            interior_step_bar.reset(total=(step_next_upsample - step_last_upsample))
            interior_step_bar.set_description(f"Progress to next stage")
            interior_step_bar.update((discriminator.step - step_last_upsample))

        for i, batch in enumerate(dataloader):
            batch = to_device(batch, device)
            if discriminator.step % opt.model_save_interval == 0 and rank == 0:
                now = datetime.now()
                now = now.strftime("%d%m%H%M_")
                torch.save(ema, os.path.join(opt.output_dir, now + 'ema.pth'))
                torch.save(ema2, os.path.join(opt.output_dir, now + 'ema2.pth'))
                torch.save(generator, os.path.join(opt.output_dir, now + 'generator.pth'))
                torch.save(encoder, os.path.join(opt.output_dir, now + 'encoder.pth'))
                torch.save(discriminator, os.path.join(opt.output_dir, now + 'discriminator.pth'))
                torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, now + 'optimizer_G.pth'))
                torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, now + 'optimizer_D.pth'))
                torch.save(scaler.state_dict(), os.path.join(opt.output_dir, now + 'scaler.pth'))
            metadata = curriculums.extract_metadata(curriculum, discriminator.step)

            if dataloader.batch_size != metadata['batch_size']:
                break

            if scaler.get_scale() < 1:
                scaler.update(1.)

            generator.train()
            encoder.train()
            discriminator.train()

            alpha = min(1, (discriminator.step - step_last_upsample) / (metadata['fade_steps']))

            uniform_colors = torch.rand(batch['real'].shape[0], batch['real'].shape[1], 1, 1).repeat(1, 1, batch['real'].shape[2], batch['real'].shape[3]).to(batch['real'].device) * 2 - 1
            real_imgs = batch['real'] * (1 - batch['mask']) + uniform_colors * batch['mask']

            metadata['nerf_noise'] = max(0, 1. - discriminator.step / 5000.)

            # TRAIN DISCRIMINATOR
            # Generate images for discriminator training
            with torch.no_grad():
                z = z_sampler((real_imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])
                split_batch_size = z.shape[0] // metadata['batch_split']
                gen_imgs = []
                for split in range(metadata['batch_split']):
                    subset_z = z[split * split_batch_size:(split + 1) * split_batch_size]
                    faces = batch['faces'][split * split_batch_size:(split + 1) * split_batch_size, :]
                    shape = encoder(batch['sdf_x'][split * split_batch_size:(split + 1) * split_batch_size])[4].mean((2, 3, 4))
                    g_imgs = render(generator(noise(faces), subset_z, shape), batch, metadata['img_size'])
                    gen_imgs.append(g_imgs)
                gen_imgs = torch.cat(gen_imgs, axis=0)

            real_imgs.requires_grad = True
            r_preds, _, _ = discriminator(real_imgs, alpha)

            if metadata['r1_lambda'] > 0:
                # Gradient penalty
                grad_real = torch.autograd.grad(outputs=scaler.scale(r_preds.sum()), inputs=real_imgs, create_graph=True)
                inv_scale = 1. / scaler.get_scale()
                grad_real = [p * inv_scale for p in grad_real][0]
                grad_penalty = (grad_real.reshape(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = 0.5 * metadata['r1_lambda'] * grad_penalty
            else:
                grad_penalty = 0
            g_preds, g_pred_latent, _ = discriminator(gen_imgs, alpha)
            if metadata['z_lambda'] > 0 or metadata['pos_lambda'] > 0:
                latent_penalty = torch.nn.MSELoss()(g_pred_latent, z) * metadata['z_lambda']
                identity_penalty = latent_penalty
            else:
                identity_penalty = 0

            d_loss = torch.nn.functional.softplus(g_preds).mean() + torch.nn.functional.softplus(-r_preds).mean() + grad_penalty + identity_penalty
            discriminator_losses.append(d_loss.item())

            optimizer_D.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.unscale_(optimizer_D)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), metadata['grad_clip'])
            scaler.step(optimizer_D)

            # TRAIN GENERATOR
            z = z_sampler((batch['real'].shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])

            split_batch_size = z.shape[0] // metadata['batch_split']

            for split in range(metadata['batch_split']):
                subset_z = z[split * split_batch_size:(split + 1) * split_batch_size]
                faces = batch['faces'][split * split_batch_size:(split + 1) * split_batch_size, :]
                shape = encoder(batch['sdf_x'][split * split_batch_size:(split + 1) * split_batch_size])[4].mean((2, 3, 4))
                gen_imgs = render(generator(noise(faces), subset_z, shape), batch, metadata['img_size'])
                g_preds, g_pred_latent, _ = discriminator(gen_imgs, alpha)

                topk_percentage = max(0.99 ** (discriminator.step / metadata['topk_interval']), metadata['topk_v']) if 'topk_interval' in metadata and 'topk_v' in metadata else 1
                topk_num = math.ceil(topk_percentage * g_preds.shape[0])

                g_preds = torch.topk(g_preds, topk_num, dim=0).values

                if metadata['z_lambda'] > 0 or metadata['pos_lambda'] > 0:
                    latent_penalty = torch.nn.MSELoss()(g_pred_latent, subset_z) * metadata['z_lambda']
                    identity_penalty = latent_penalty
                else:
                    identity_penalty = 0

                g_loss = torch.nn.functional.softplus(-g_preds).mean() + identity_penalty
                generator_losses.append(g_loss.item())

                scaler.scale(g_loss).backward()

            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), metadata.get('grad_clip', 0.3))
            scaler.step(optimizer_G)
            scaler.update()
            optimizer_G.zero_grad()
            ema.update(generator.parameters())
            ema2.update(generator.parameters())

            if rank == 0:
                interior_step_bar.update(1)
                if i % 10 == 0:
                    tqdm.write(
                        f"[Experiment: {opt.output_dir}] [Epoch: {discriminator.epoch}/{opt.n_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [Step: {discriminator.step}] [Alpha: {alpha:.2f}] [Img Size: {metadata['img_size']}] [Batch Size: {metadata['batch_size']}] [TopK: {topk_num}] [Scale: {scaler.get_scale()}]")

                if discriminator.step % opt.sample_interval == 0:
                    eval_sample = next(eval_loader)
                    eval_sample = to_device(eval_sample, device)
                    generator.eval()
                    encoder.eval()
                    with torch.no_grad():
                        eval_faces = eval_sample['faces']
                        eval_shape = encoder(eval_sample['sdf_x'])[4].mean((2, 3, 4))

                    with torch.no_grad():
                        copied_metadata = copy.deepcopy(metadata)
                        copied_metadata['img_size'] = 128
                        gen_imgs = render(generator(eval_faces, fixed_z.to(device), eval_shape), eval_sample, copied_metadata['img_size'])
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_fixed.png"), nrow=4, normalize=True, range=(-1, 1))

                    ema.store(generator.parameters())
                    ema.copy_to(generator.parameters())
                    generator.eval()
                    encoder.eval()
                    with torch.no_grad():
                        copied_metadata = copy.deepcopy(metadata)
                        copied_metadata['img_size'] = 128
                        gen_imgs = render(generator(eval_faces, fixed_z.to(device), eval_shape), eval_sample, copied_metadata['img_size'])
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_fixed_ema.png"), nrow=4, normalize=True, range=(-1, 1))

                    save_image(real_imgs.detach(), os.path.join(opt.output_dir, f"real_{discriminator.step}.png"), nrow=4, normalize=True, range=(-1, 1))

                    ema.restore(generator.parameters())

                if discriminator.step % opt.sample_interval == 0:
                    torch.save(ema, os.path.join(opt.output_dir, 'ema.pth'))
                    torch.save(ema2, os.path.join(opt.output_dir, 'ema2.pth'))
                    torch.save(generator, os.path.join(opt.output_dir, 'generator.pth'))
                    torch.save(encoder, os.path.join(opt.output_dir, 'encoder.pth'))
                    torch.save(discriminator, os.path.join(opt.output_dir, 'discriminator.pth'))
                    torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, 'optimizer_G.pth'))
                    torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, 'optimizer_D.pth'))
                    torch.save(scaler.state_dict(), os.path.join(opt.output_dir, 'scaler.pth'))
                    torch.save(generator_losses, os.path.join(opt.output_dir, 'generator.losses'))
                    torch.save(discriminator_losses, os.path.join(opt.output_dir, 'discriminator.losses'))

            if opt.eval_freq > 0 and (discriminator.step + 1) % opt.eval_freq == 0:
                generated_dir = os.path.join(opt.output_dir, 'evaluation/generated')
                copied_metadata = copy.deepcopy(metadata)
                copied_metadata['img_size'] = 128
                fid_evaluation.setup_evaluation(copied_metadata, generated_dir, target_size=128)
                ema.store(generator.parameters())
                ema.copy_to(generator.parameters())
                generator.eval()
                fid_num_imgs = 2048
                img_counter = 0
                pbar = tqdm("generating images", total=fid_num_imgs)

                with torch.no_grad():
                    copied_metadata = copy.deepcopy(metadata)
                    copied_metadata['img_size'] = 128
                    fid_dataloader = iter(GraphDataLoader(SDFGridDataset(EasyDict(copied_metadata)), fixed_z.shape[0], shuffle=True, pin_memory=False, drop_last=True, num_workers=0))
                    while img_counter < fid_num_imgs:
                        batch = next(fid_dataloader)
                        batch = to_device(batch, device)
                        z = torch.randn((batch['real'].shape[0], generator.z_dim), device=device)
                        faces = batch['faces']
                        shape = encoder(batch['sdf_x'])[4].mean((2, 3, 4))
                        generated_imgs = render(generator(faces, z, shape), batch, 128).cpu()

                        for img in generated_imgs:
                            save_image(img, os.path.join(generated_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))
                            img_counter += 1
                            pbar.update(1)
                pbar.close()

                ema.restore(generator.parameters())
                if rank == 0:
                    fid = fid_evaluation.calculate_fid(metadata['dataset'], generated_dir, target_size=128)
                    with open(os.path.join(opt.output_dir, f'fid.txt'), 'a') as f:
                        f.write(f'\n{discriminator.step}:{fid}')

                torch.cuda.empty_cache()

            discriminator.step += 1
        discriminator.epoch += 1


def render(face_colors, batch, img_resolution):
    rendered_color = render_helper.render(batch['vertices'], batch['indices'], to_vertex_colors_scatter(face_colors.reshape(-1, 3), batch), batch["ranges"].cpu(), resolution=img_resolution)
    return rendered_color.permute((0, 3, 1, 2))


def noise(face_positions):
    return face_positions + (torch.randn_like(face_positions) - 0.5) * 0.01


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
    parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='runs/debug')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--port', type=str, default='12355')
    parser.add_argument('--curriculum', type=str, default='CARLA')
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=5000)

    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    num_gpus = 1
    train(0, opt)
