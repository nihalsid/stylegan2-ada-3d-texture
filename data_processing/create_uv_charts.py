import hydra
from pathlib import Path
import scipy
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
import torch
import trimesh
from dataset import GraphDataLoader, to_device, to_vertex_colors_scatter
from dataset.mesh_real_features_uv import split_into_six
from model.differentiable_renderer import DifferentiableRenderer
from PIL import Image


@hydra.main(config_path='../config', config_name='stylegan2')
def create_silhouttes(config):
    from dataset.mesh_real_features_atlas import FaceGraphMeshDataset
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=1, num_workers=0)
    render_helper = DifferentiableRenderer(config.image_size, 'bounds', config.colorspace, num_channels=4).cuda()
    Path("runs/uv_mask").mkdir(exist_ok=True)
    Path("runs/uv_positions").mkdir(exist_ok=True)
    Path("runs/uv_texture").mkdir(exist_ok=True)
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, torch.device("cuda:0"))
        rendered_color_gt = render_helper.render(batch['vertices'], batch['indices'],
                                                 to_vertex_colors_scatter(batch["x"][:, :3], batch),
                                                 batch["ranges"].cpu(), batch['bg']).permute((0, 3, 1, 2))
        rendered_color_gt_texture = render_helper.render(batch['vertices'], batch['indices'],
                                                 to_vertex_colors_scatter(batch["y"][:, :3], batch),
                                                 batch["ranges"].cpu(), batch['bg']).permute((0, 3, 1, 2))[:, :3, :, :]
        rendered_color_gt_texture = ((rendered_color_gt_texture * 0.5 + 0.5) * 255).int()
        rendered_color_gt_pos = rendered_color_gt[:, :3, :, :]
        rendered_color_gt_mask = ((1 - rendered_color_gt[:, 3, :, :]) * 255).int()

        row_0 = torch.cat([rendered_color_gt_mask[0, :, :], rendered_color_gt_mask[1, :, :], rendered_color_gt_mask[2, :, :]], dim=-1)
        row_1 = torch.cat([rendered_color_gt_mask[3, :, :], rendered_color_gt_mask[4, :, :], rendered_color_gt_mask[5, :, :]], dim=-1)
        mask = Image.fromarray(torch.cat([row_0, row_1], dim=-2).cpu().numpy().astype(np.uint8))

        row_0 = torch.cat([rendered_color_gt_pos[0, :, :, :], rendered_color_gt_pos[1, :, :, :], rendered_color_gt_pos[2, :, :, :]], dim=-1)
        row_1 = torch.cat([rendered_color_gt_pos[3, :, :, :], rendered_color_gt_pos[4, :, :, :], rendered_color_gt_pos[5, :, :, :]], dim=-1)
        positions = torch.cat([row_0, row_1], dim=-2).permute((1, 2, 0)).cpu().numpy()

        row_0 = torch.cat([rendered_color_gt_texture[0, :, :, :], rendered_color_gt_texture[1, :, :, :], rendered_color_gt_texture[2, :, :, :]], dim=-1)
        row_1 = torch.cat([rendered_color_gt_texture[3, :, :, :], rendered_color_gt_texture[4, :, :, :], rendered_color_gt_texture[5, :, :, :]], dim=-1)
        colors = Image.fromarray(torch.cat([row_0, row_1], dim=-2).permute((1, 2, 0)).cpu().numpy().astype(np.uint8))

        mask.save(str(Path("runs/uv_mask") / f"{batch['name'][0]}.jpg"))
        colors.save(str(Path("runs/uv_texture") / f"{batch['name'][0]}.jpg"))
        np.savez_compressed(str(Path("runs/uv_positions") / f"{batch['name'][0]}.npz"), positions)


def create_uv_mapping(proc, num_proc):
    mesh_path = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape/shapenet-chairs-manifold-highres")
    map_path = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape/uv_positions")
    mask_path = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape/uv_mask")
    output_path = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape/uv_map")
    output_path.mkdir(exist_ok=True)
    meshes = list(mesh_path.iterdir())
    meshes = [x for i, x in enumerate(meshes) if i % num_proc == proc]
    for mesh in tqdm(meshes):
        if not (map_path / f'{mesh.stem}.npz').exists():
            continue
        tmesh = trimesh.load(mesh / "model_normalized.obj", process=False)
        uv_map = split_into_six(np.load(str(map_path / f'{mesh.stem}.npz'))['arr'])
        uv_mask = split_into_six(np.array(Image.open(mask_path / f'{mesh.stem}.jpg'))[:, :, np.newaxis])

        selected_mapping = np.zeros([tmesh.vertices.shape[0], 4])
        selected_mapping[:, 3] = float('inf')

        for mp_idx in range(uv_map.shape[0]):
            all_pos = uv_map[mp_idx].reshape((-1, 3))
            max_i = uv_map[mp_idx].shape[0]
            max_j = uv_map[mp_idx].shape[1]
            pixel_coordinates_i, pixel_coordinates_j = np.meshgrid(list(range(max_i)), list(range(max_j)), indexing='ij')
            pixel_coordinates_i = pixel_coordinates_i / max_i
            pixel_coordinates_j = pixel_coordinates_j / max_j
            pixel_coordinates = np.stack([pixel_coordinates_i, pixel_coordinates_j], axis=-1).reshape((-1, 2))
            valid_pos = uv_mask[mp_idx].flatten() == 255
            pixel_coordinates = pixel_coordinates[valid_pos]
            kdtree = scipy.spatial.cKDTree(all_pos[valid_pos])
            dist, indices = kdtree.query(np.array(tmesh.vertices), k=1)
            dmask = selected_mapping[:, 3] - dist > 0.02
            selected_mapping[dmask, 1:3] = pixel_coordinates[indices, :][dmask, :]
            selected_mapping[dmask, 0] = mp_idx
            selected_mapping[dmask, 3] = dist[dmask]

        np.save(str(output_path / f'{mesh.stem}.npy'), selected_mapping[:, 0:3])
        print(mesh)


@hydra.main(config_path='../config', config_name='stylegan2')
def render_with_uv(config):
    from dataset.mesh_real_features_uv import FaceGraphMeshDataset
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=config.batch_size, num_workers=0)
    render_helper = DifferentiableRenderer(config.image_size, 'bounds', config.colorspace).cuda()
    Path("runs/uv_mask").mkdir(exist_ok=True)
    Path("runs/uv_positions").mkdir(exist_ok=True)
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, torch.device("cuda:0"))
        texture_map = []
        for name in batch['name']:
            texture_map.append(torch.from_numpy(split_into_six(np.array(Image.open(Path("runs/uv_texture") / f'{name}.jpg')))).permute(0, 3, 1, 2))
        texture_map = torch.cat(texture_map, dim=0).to(batch['vertices'].device).float() / 255 * 2 - 1
        vertices_mapped = texture_map[batch["uv"][:, 0].long(), :, (batch["uv"][:, 1] * config.image_size).long(), (batch["uv"][:, 2] * config.image_size).long()]
        rendered_texture = render_helper.render(batch['vertices'], batch['indices'],
                                                torch.cat([vertices_mapped, torch.ones((vertices_mapped.shape[0], 1), device=vertices_mapped.device)], dim=-1),
                                                batch["ranges"].cpu(),
                                                batch['bg']).permute((0, 3, 1, 2))
        save_image(rendered_texture, f"runs/images/{batch_idx:04d}.png", nrow=6, value_range=(-1, 1), normalize=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)
    args = parser.parse_args()

    create_uv_mapping(args.proc, args.num_proc)
    # create_silhouttes()
    # render_with_uv()
