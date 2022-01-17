from pathlib import Path

import hydra
from torchvision.utils import save_image
from tqdm import tqdm
import torch

from dataset.mesh_real_features_patch import FaceGraphMeshDataset
from dataset import GraphDataLoader, to_device, to_vertex_colors_scatter
from model.differentiable_renderer import DifferentiableRenderer
from trainer.train_stylegan_real_feature_patch import StyleGAN2Trainer


@hydra.main(config_path='../config', config_name='stylegan2')
def test_patches(config):
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=config.batch_size, num_workers=0)
    render_helper = DifferentiableRenderer(config.image_size, 'bounds', config.colorspace, num_channels=4).cuda()
    Path("runs/patches").mkdir(exist_ok=True)
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, torch.device("cuda:0"))
        batch['real_hres'] = dataset.get_color_bg_real_hres(batch)
        first_views = list(range(0, batch['real_hres'].shape[0], config.views_per_sample))
        patches, patches_mask = StyleGAN2Trainer.extract_patches_from_tensor(batch['real_hres'][first_views], batch['mask_hres'][first_views, 0, :, :], config.num_patch_per_view * config.views_per_sample, config.patch_size)
        patches = patches.reshape((config.batch_size * config.views_per_sample * config.num_patch_per_view, 3, config.patch_size, config.patch_size))
        patches_mask = patches_mask.reshape((config.batch_size * config.views_per_sample * config.num_patch_per_view, 1, config.patch_size, config.patch_size))
        rendered_color_gt = render_helper.render(batch['vertices'], batch['indices'], to_vertex_colors_scatter(batch["y"], batch), batch["ranges"].cpu(), batch['bg'], resolution=config.image_size_hres).permute((0, 3, 1, 2))
        rend_patches, rend_patches_mask = StyleGAN2Trainer.extract_patches_from_tensor(rendered_color_gt[:, :3, :, :], 1 - rendered_color_gt[:, 3, :, :], config.num_patch_per_view, config.patch_size)
        rend_patches = rend_patches.reshape((config.batch_size * config.views_per_sample * config.num_patch_per_view, 3, config.patch_size, config.patch_size))
        rend_patches_mask = rend_patches_mask.reshape((config.batch_size * config.views_per_sample * config.num_patch_per_view, 1, config.patch_size, config.patch_size))
        patches = dataset.cspace_convert_back(patches)
        batch['real_hres'] = dataset.cspace_convert_back(batch['real_hres'])
        rendered_color_gt = dataset.cspace_convert_back(rendered_color_gt)
        rend_patches = dataset.cspace_convert_back(rend_patches)
        save_image(torch.cat([batch['real_hres'], rendered_color_gt[:, :3, :, :]], dim=0), f"runs/patches/{batch_idx:04d}_view.png", nrow=config.batch_size, value_range=(-1, 1), normalize=True)
        save_image(rendered_color_gt[:, 3, :, :].unsqueeze(1), f"runs/patches/{batch_idx:04d}_mask.png", nrow=config.batch_size, value_range=(0, 1), normalize=True)
        save_image(torch.cat([patches, patches_mask.expand(-1, 3, -1, -1), rend_patches, rend_patches_mask.expand(-1, 3, -1, -1)], dim=0), f"runs/patches/{batch_idx:04d}_patch.png", nrow=config.batch_size, value_range=(-1, 1), normalize=True)


if __name__ == '__main__':
    test_patches()
