from pathlib import Path

import hydra
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import torch

from dataset.mesh_real_features import FaceGraphMeshDataset
from dataset import GraphDataLoader, to_device, to_vertex_colors_scatter
from dataset.mesh_real_sdfgrid_sparse import Collater
from dataset.mesh_real_sdfgrid import SDFGridDataset
from dataset.mesh_real_sdfgrid_sparse import SparseSDFGridDataset
from model.augment import AugmentPipe
from model.differentiable_renderer import DifferentiableRenderer
from model.graph import pool, unpool


@hydra.main(config_path='../config', config_name='stylegan2')
def test_dataloader(config):
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=1, num_workers=0)
    render_helper = DifferentiableRenderer(config.image_size).cuda()
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # sanity test render + target colors
        batch = to_device(batch, torch.device("cuda:0"))
        print(batch['name'])
        rendered_color_gt = render_helper.render(batch['vertices'], batch['indices'], to_vertex_colors_scatter(batch["y"], batch), batch["ranges"].cpu())
        save_image(rendered_color_gt.permute((0, 3, 1, 2)), "test_dataloader_fake.png", nrow=4, value_range=(-1, 1), normalize=True)
        # sanity test graph counts and pool maps
        x_0 = pool(batch['y'], batch['graph_data']['node_counts'][0], batch['graph_data']['pool_maps'][0])
        x_1 = pool(x_0, batch['graph_data']['node_counts'][1], batch['graph_data']['pool_maps'][1])
        x_1 = unpool(x_1, batch['graph_data']['pool_maps'][1])
        x_0 = unpool(x_1, batch['graph_data']['pool_maps'][0])
        # works only if uv's are present
        # save_image(dataset.to_image(batch["y"], batch["graph_data"]["level_masks"][0]), "test_target.png", nrow=4, value_range=(-1, 1), normalize=True)
        # save_image(dataset.to_image(x_0, batch["graph_data"]["level_masks"][0]), "test_pooled.png", nrow=4, value_range=(-1, 1), normalize=True)
        # break


@hydra.main(config_path='../config', config_name='stylegan2')
def test_view_angles_together(config):
    from dataset.mesh_real_features import FaceGraphMeshDataset
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=8, num_workers=0)
    render_helper = DifferentiableRenderer(config.image_size, 'bounds', config.colorspace).cuda()
    augment_pipe = AugmentPipe(config.ada_start_p, config.ada_target, config.ada_interval, config.ada_fixed, config.batch_size, config.views_per_sample, config.colorspace).cuda()
    Path("runs/images_compare").mkdir(exist_ok=True)
    for epoch in range(2):
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            batch = to_device(batch, torch.device("cuda:0"))
            batch['real'] = augment_pipe(dataset.get_color_bg_real(batch))
            rendered_color_gt = render_helper.render(batch['vertices'], batch['indices'], to_vertex_colors_scatter(batch["y"][:, :3], batch), batch["ranges"].cpu(), batch['bg']).permute((0, 3, 1, 2))
            batch['real'] = dataset.cspace_convert_back(batch['real'])
            rendered_color_gt = dataset.cspace_convert_back(rendered_color_gt)
            save_image(torch.cat([batch['real'], batch['mask'].expand(-1, 3, -1, -1), rendered_color_gt]), f"runs/images_compare/test_view_{batch_idx:04d}.png", nrow=4, value_range=(-1, 1), normalize=True)


@hydra.main(config_path='../config', config_name='stylegan2')
def test_view_angles_fake(config):
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=8, num_workers=0)
    render_helper = DifferentiableRenderer(config.image_size, 'bounds').cuda()
    Path("runs/images_fake").mkdir(exist_ok=True)
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, torch.device("cuda:0"))
        rendered_color_gt = render_helper.render(batch['vertices'], batch['indices'], to_vertex_colors_scatter(batch["y"], batch), batch["ranges"].cpu())
        save_image(rendered_color_gt.permute((0, 3, 1, 2)), f"runs/images_fake/test_view_{batch_idx:04d}.png", nrow=4, value_range=(-1, 1), normalize=True)


@hydra.main(config_path='../config', config_name='stylegan2')
def test_view_angles_real(config):
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=32, num_workers=0)
    Path("runs/images_real").mkdir(exist_ok=True)
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        save_image(batch['real'], f"runs/images_real/test_view_{batch_idx:04d}.png", nrow=8, value_range=(-1, 1), normalize=True)


@hydra.main(config_path='../config', config_name='stylegan2')
def test_masks(config):
    import numpy as np
    from PIL import Image
    for mask in tqdm(list(Path(config.mask_path).iterdir())):
        if np.array(Image.open(mask)).sum() == 0:
            print(mask)


@hydra.main(config_path='../config', config_name='stylegan2')
def test_grid_dataloader(config):
    from model.styleganvox.generator import Generator
    from model.styleganvox import SDFEncoder
    from model.raycast_rgbd.raycast_rgbd import Raycast2DHandler
    from PIL import Image
    import numpy as np
    batch_size, render_shape = 4, (config.image_size, config.image_size)
    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, 64, 3).cuda()
    E = SDFEncoder(1).cuda()
    dataset = SDFGridDataset(config)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    Path("runs/images_sdf").mkdir(exist_ok=True)
    dims = (64, 64, 64)
    voxel_size = 0.03125
    trunc = 5 * voxel_size
    raycast_handler = Raycast2DHandler(torch.device("cuda"), batch_size, dims, render_shape, voxel_size, trunc)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            batch = to_device(batch, torch.device("cuda"))
            shape = E(batch["x"])
            fake_grid = G(torch.randn(batch_size, 512).cuda(), shape, noise_mode='const')
            r_color, r_depth, r_normals = raycast_handler.raycast_sdf(batch['x'], batch['y'],
                                                                      batch['view'], batch['intrinsic'])
            r_color = r_color.permute((0, 3, 1, 2)).cpu()
            r_color = r_color.permute((0, 2, 3, 1))
            r_color = (r_color + 1) / 2
            for i in range(r_color.shape[0]):
                color_i = r_color[i].numpy()
                color_i[color_i == -float('inf')] = 1
                color_i = color_i * 255
                Image.fromarray(color_i.astype(np.uint8)).save(f"runs/images_sdf/render_{batch_idx * 8 + i}.jpg")


@hydra.main(config_path='../config', config_name='stylegan2')
def test_sparsegrid_dataloader(config):
    from model.raycast_rgbd.raycast_rgbd import Raycast2DSparseHandler
    from PIL import Image
    import numpy as np
    batch_size, render_shape = 4, (config.image_size, config.image_size)
    dataset = SparseSDFGridDataset(config)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=Collater([], []))
    dims = (64, 64, 64)
    voxel_size = 0.03125
    trunc = 5 * voxel_size
    Path("runs/images_sparsesdf").mkdir(exist_ok=True)
    raycast_handler = Raycast2DSparseHandler(torch.device("cuda"), batch_size, dims, render_shape, voxel_size, trunc)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            batch = to_device(batch, torch.device("cuda"))
            r_color, r_depth, r_normals = raycast_handler.raycast_sdf(batch["x_dense"], batch['sparse_data'][0], batch['sparse_data'][1],
                                                                      batch['sparse_data'][2], batch['view'], batch['intrinsic'])
            r_color = r_color.permute((0, 3, 1, 2)).cpu()
            r_color = r_color.permute((0, 2, 3, 1))
            r_color = (r_color + 1) / 2
            for i in range(r_color.shape[0]):
                color_i = r_color[i].numpy()
                color_i[color_i == -float('inf')] = 0
                color_i = color_i * 255
                Image.fromarray(color_i.astype(np.uint8)).save(f"runs/images_sparsesdf/render_{batch_idx * 8 + i}.jpg")


@hydra.main(config_path='../config', config_name='stylegan2')
def test_pigan_dataloader(config):
    from dataset.mesh_real_pigan import SDFGridDataset
    dataset = SDFGridDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=4, num_workers=0)
    render_helper = DifferentiableRenderer(config.image_size, 'bounds').cuda()
    Path("runs/images_fake").mkdir(exist_ok=True)
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, torch.device("cuda:0"))
        colors = batch["faces"]
        rendered_color_gt = render_helper.render(batch['vertices'], batch['indices'], to_vertex_colors_scatter(colors.reshape(-1, 3), batch), batch["ranges"].cpu())
        save_image(torch.cat([rendered_color_gt.permute((0, 3, 1, 2)), batch['real']], 0), f"runs/images_fake/test_view_{batch_idx:04d}.png", nrow=4, value_range=(-1, 1), normalize=True)
        break


@hydra.main(config_path='../config', config_name='stylegan2')
def test_uv_dataloader(config):
    from dataset.mesh_real_features_uv import FaceGraphMeshDataset
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=config.batch_size, num_workers=0)
    render_helper = DifferentiableRenderer(config.image_size, 'bounds').cuda()
    Path("runs/images_fake").mkdir(exist_ok=True)
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, torch.device("cuda:0"))
        texture_atlas = torch.zeros((config.batch_size, 6, 3, config.image_size, config.image_size)).to(batch['uv'].device)
        for c in range(6):
            texture_atlas[:, c, :, :, :] = -1 + 2 * c / 6
        texture_atlas = texture_atlas.reshape((-1, texture_atlas.shape[2], texture_atlas.shape[3], texture_atlas.shape[4]))
        vertices_mapped = texture_atlas[batch["uv"][:, 0].long(), :, (batch["uv"][:, 1] * config.image_size).long(), (batch["uv"][:, 2] * config.image_size).long()]
        vertices_mapped = torch.cat([vertices_mapped, torch.ones((vertices_mapped.shape[0], 1), device=vertices_mapped.device)], dim=-1)
        rendered_color = render_helper.render(batch['vertices'], batch['indices'], vertices_mapped, batch["ranges"].cpu(), None)
        save_image(torch.cat([rendered_color.permute((0, 3, 1, 2)), batch['real']], 0), f"runs/images_fake/test_view_{batch_idx:04d}.png", nrow=4, value_range=(-1, 1), normalize=True)
        break


@hydra.main(config_path='../config', config_name='stylegan2_car')
def test_compcars_together(config):
    from dataset.meshcar_real_features_ff2 import FaceGraphMeshDataset
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=1, num_workers=0)
    render_helper = DifferentiableRenderer(config.image_size, 'bounds', config.colorspace).cuda()
    augment_pipe = AugmentPipe(config.ada_start_p, config.ada_target, config.ada_interval, config.ada_fixed, config.batch_size, config.views_per_sample, config.colorspace).cuda()
    Path("runs/images_compare").mkdir(exist_ok=True)
    for epoch in range(2):
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            batch = to_device(batch, torch.device("cuda:0"))
            batch['real'] = augment_pipe(dataset.get_color_bg_real(batch))
            rendered_color_gt = render_helper.render(batch['vertices'], batch['indices'], to_vertex_colors_scatter(batch["y"][:, :3], batch), batch["ranges"].cpu(), batch['bg']).permute((0, 3, 1, 2))
            batch['real'] = dataset.cspace_convert_back(batch['real'])
            rendered_color_gt = dataset.cspace_convert_back(rendered_color_gt)
            save_image(torch.cat([batch['real'], batch['mask'].expand(-1, 3, -1, -1), rendered_color_gt]), f"runs/images_compare/test_view_{batch_idx:04d}.png", nrow=4, value_range=(-1, 1), normalize=True)


if __name__ == '__main__':
    # test_view_angles_together()
    # test_uv_dataloader()
    test_compcars_together()
