from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from cleanfid import fid

from dataset import GraphDataLoader, to_device, to_vertex_colors_scatter
from model.differentiable_renderer import DifferentiableRenderer
import hydra


OUTPUT_DIR = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/photoshape")
REAL_DIR = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/photoshape/real")


# TODO: Run with EMA

def render_faces(R, face_colors, batch, render_size, image_size):
    rendered_color = R.render(batch['vertices'], batch['indices'], to_vertex_colors_scatter(face_colors, batch), batch["ranges"].cpu(), resolution=render_size)
    ret_val = rendered_color.permute((0, 3, 1, 2))
    if render_size != image_size:
        ret_val = torch.nn.functional.interpolate(ret_val, (image_size, image_size), mode='bilinear', align_corners=True)
    return ret_val


def get_parameters_from_state_dict(state_dict, filter_key):
    new_state_dict = OrderedDict()
    for k in state_dict:
        if k.startswith(filter_key):
            new_state_dict[".".join(k.split(".")[1:])] = state_dict[k]
    return new_state_dict


def compute_fid(output_dir_fake, output_dir_real, filepath, device):
    fid_score = fid.compute_fid(str(output_dir_real), str(output_dir_fake), device=device, dataset_res=256, num_workers=0)
    print(f'FID: {fid_score:.3f}')
    kid_score = fid.compute_kid(str(output_dir_real), str(output_dir_fake), device=device, dataset_res=256, num_workers=0)
    print(f'KID: {kid_score:.3f}')
    Path(filepath).write_text(f"fid = {fid_score}\nkid = {kid_score}")


@hydra.main(config_path='../config', config_name='stylegan2')
def create_real(config):
    from PIL import Image
    image_size = 256
    REAL_DIR.mkdir(exist_ok=True, parents=True)
    for idx, p in enumerate(tqdm(list(Path(config.image_path).iterdir()))):
        img = Image.open(p)
        width, height = img.size
        result = Image.new(img.mode, (int(width * 1.2), int(height * 1.2)), (255, 255, 255))
        result.paste(img, (int(width * 0.1), int(height * 0.1)))
        result = result.resize((image_size, image_size), resample=Image.LANCZOS)
        result.save(REAL_DIR / f"{idx:06d}.jpg")


# UV Parameterization
@hydra.main(config_path='../config', config_name='stylegan2')
def evaluate_uv_parameterized_gan(config):
    config.views_per_sample = 4
    config.image_size = 128
    config.render_size = 256
    num_latent = 4

    from model.stylegan2.generator import Generator
    from dataset.mesh_real_features_uv import FaceGraphMeshDataset
    OUTPUT_DIR_UV = OUTPUT_DIR / f"uv_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_UV.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/06022206_StyleGAN23D_uv_silhoutte_256/checkpoints/_epoch=59.ckpt"

    def render(r_texture_atlas, r_batch):
        r_texture_atlas = r_texture_atlas.reshape((-1, r_texture_atlas.shape[2], r_texture_atlas.shape[3], r_texture_atlas.shape[4]))
        vertices_mapped = r_texture_atlas[r_batch["uv"][:, 0].long(), :, (r_batch["uv"][:, 1] * config.image_size).long(), (r_batch["uv"][:, 2] * config.image_size).long()]
        vertices_mapped = torch.cat([vertices_mapped, torch.ones((vertices_mapped.shape[0], 1), device=vertices_mapped.device)], dim=-1)
        rendered_color = R.render(r_batch['vertices'], r_batch['indices'], vertices_mapped, r_batch["ranges"].cpu(), resolution=config.render_size)
        return rendered_color.permute((0, 3, 1, 2))

    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, shuffle=False, drop_last=True, num_workers=0)
    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.image_size, 3).to(device)
    R = DifferentiableRenderer(config.image_size, "bounds", config.colorspace)
    G.load_state_dict(get_parameters_from_state_dict(torch.load(CHECKPOINT, map_location=device)["state_dict"], "G"))
    G.eval()
    for iter_idx, batch in enumerate(tqdm(eval_loader)):
        eval_batch = to_device(batch, device)
        silhoutte = eval_batch['silhoutte'].reshape([config.batch_size * 6, 2, config.image_size, config.image_size])
        for z_idx in range(num_latent):
            z = torch.randn(config.batch_size, config.latent_dim).to(device)
            atlas = G(z, silhoutte, noise_mode='const')
            fake_render = render(atlas, eval_batch).cpu()
            for batch_idx in range(fake_render.shape[0]):
                save_image(fake_render[batch_idx], OUTPUT_DIR_UV / f"{iter_idx}_{batch_idx}_{z_idx}.jpg", value_range=(-1, 1), normalize=True)
    compute_fid(OUTPUT_DIR_UV, REAL_DIR, OUTPUT_DIR_UV.parent / f"score_uv_{config.views_per_sample}_{num_latent}.txt", device)


# 3D Grid Parameterization
@hydra.main(config_path='../config', config_name='stylegan2')
def evaluate_3d_parameterized_gan(config):
    config.batch_size = 1
    config.views_per_sample = 4
    config.image_size = 256
    config.render_size = 256
    config.dataset_path = "/cluster/gimli/ysiddiqui/CADTextures/Photoshape/shapenet-chairs-sdfgrid/"
    num_latent = 4

    from model.styleganvox_sparse import SDFEncoder
    from model.styleganvox_sparse.generator import Generator
    from dataset.mesh_real_sdfgrid_sparse import SparseSDFGridDataset
    from dataset.mesh_real_sdfgrid_sparse import Collater
    from MinkowskiEngine.MinkowskiOps import MinkowskiToDenseTensor
    from MinkowskiEngine.MinkowskiSparseTensor import SparseTensor
    import marching_cubes as mc
    import trimesh
    from model.raycast_rgbd.raycast_rgbd import Raycast2DSparseHandler

    OUTPUT_DIR_GRID = OUTPUT_DIR / f"grid_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_GRID.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/05020007_StyleGAN23D_sparsereal-bgg-lrd1g14-m5-256/checkpoints/_epoch=24.ckpt"
    use_tsdf_renderer = True

    device = torch.device("cuda:0")
    eval_dataset = SparseSDFGridDataset(config)
    eval_loader = DataLoader(eval_dataset, config.batch_size, shuffle=False, drop_last=True, num_workers=0, collate_fn=Collater([], []))

    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, 64, 3).to(device)
    E = SDFEncoder(1).to(device)
    if use_tsdf_renderer:
        R = Raycast2DSparseHandler(device, config.batch_size, (128, 128, 128), (config.render_size, config.render_size), 0.015625, 0.015625 * 5)
    else:
        R = DifferentiableRenderer(config.image_size, "bounds", config.colorspace)
    state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
    G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
    E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
    G.eval()
    E.eval()

    to_dense = MinkowskiToDenseTensor(torch.Size([config.batch_size, 3, 128, 128, 128]))

    for iter_idx, batch in enumerate(tqdm(eval_loader)):
        eval_batch = to_device(batch, device)
        shape = E(eval_batch['x_dense_064'])
        if not use_tsdf_renderer:
            vertices, triangles = mc.marching_cubes(eval_batch['x_dense'][0][0].cpu().numpy(), 0)
            vertices = vertices / 128
            mc.export_obj(vertices, triangles, f"/tmp/{batch['name'][0]}.obj")
            mesh = trimesh.load(f"/tmp/{batch['name'][0]}.obj", process=False)
            scale = (mesh.bounds[1] - mesh.bounds[0]).max()
        for z_idx in range(num_latent):
            z = torch.randn(config.batch_size, config.latent_dim).to(device)
            csdf = G(z, eval_batch['sparse_data_064'][0].long(), eval_batch['sparse_data'][0].long(), shape, noise_mode='const')
            if not use_tsdf_renderer:
                query_positions = torch.clamp(eval_batch['faces'] * 2 * scale, -1, 1)
                query_positions = torch.cat([query_positions[:, :, 2:3], query_positions[:, :, 1:2], query_positions[:, :, 0:1]], dim=-1)
                query_positions = query_positions.reshape(config.batch_size, -1, 1, 1, 3)
                csdf_dense = to_dense(SparseTensor(csdf, eval_batch['sparse_data'][0].int()))
                # Path(f"{iter_idx}_query.obj").write_text("\n".join([f"v {eval_batch['faces'][0][i, 0] * scale} {eval_batch['faces'][0][i, 1] * scale} {eval_batch['faces'][0][i, 2] * scale}" for i in range(eval_batch['faces'][0].shape[0])]))
                # Path(f"{iter_idx}_grid.obj").write_text("\n".join([f"v {eval_batch['sparse_data'][0][i, 1] / 128 - 0.5} {eval_batch['sparse_data'][0][i, 2] / 128 - 0.5} {eval_batch['sparse_data'][0][i, 3] / 128 - 0.5}" for i in range(eval_batch['sparse_data'][0].shape[0])]))
                face_colors = torch.nn.functional.grid_sample(csdf_dense, query_positions, mode='bilinear', padding_mode='border', align_corners=False)
                face_colors = face_colors.squeeze(-1).squeeze(-1).permute((0, 2, 1)).reshape(-1, 3)
                fake_render = R.render(eval_batch['vertices'], eval_batch['indices'], to_vertex_colors_scatter(face_colors, eval_batch), eval_batch["ranges"].cpu(), resolution=config.render_size).permute((0, 3, 1, 2)).cpu()
            else:
                fake_render, _, _ = R.raycast_sdf(eval_batch['x_dense'], eval_batch['sparse_data'][0], eval_batch['sparse_data'][1], csdf.contiguous(), eval_batch['view'], eval_batch['intrinsic'])
                fake_render[fake_render == -float('inf')] = 1
                fake_render = fake_render.permute((0, 3, 1, 2)).cpu()
            for batch_idx in range(fake_render.shape[0]):
                save_image(fake_render[batch_idx], OUTPUT_DIR_GRID / f"{iter_idx}_{batch_idx}_{z_idx}.jpg", value_range=(-1, 1), normalize=True)

    compute_fid(OUTPUT_DIR_GRID, REAL_DIR, OUTPUT_DIR_GRID.parent / f"score_grid_{config.views_per_sample}_{num_latent}.txt", device)


# Implicit Parameterization


# Triplane Implicit Parameterization


# Ours
@hydra.main(config_path='../config', config_name='stylegan2')
def evaluate_our_gan(config):
    from model.graph_generator_u import Generator
    from model.graph import GraphEncoder
    from dataset.mesh_real_features import FaceGraphMeshDataset
    config.batch_size = 1
    config.views_per_sample = 4
    config.image_size = 256
    config.render_size = 1024
    config.num_mapping_layers = 5
    config.g_channel_base = 32768
    num_latent = 4

    OUTPUT_DIR_OURS = OUTPUT_DIR / f"ours_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21011752_StyleGAN23D_fg3bgg-lrd1g14-v2m5-1024/checkpoints/_epoch=129.ckpt"

    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
    E = GraphEncoder(eval_dataset.num_feats).to(device)
    R = DifferentiableRenderer(config.render_size, "bounds", config.colorspace)
    state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
    G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
    E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
    G.eval()
    E.eval()

    for iter_idx, batch in enumerate(tqdm(eval_loader)):
        eval_batch = to_device(batch, device)
        shape = E(eval_batch['x'], eval_batch['graph_data'])
        for z_idx in range(num_latent):
            z = torch.randn(config.batch_size, config.latent_dim).to(device)
            fake = G(eval_batch['graph_data'], z, shape, noise_mode='const')
            fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
            for batch_idx in range(fake_render.shape[0]):
                save_image(fake_render[batch_idx], OUTPUT_DIR_OURS / f"{iter_idx}_{batch_idx}_{z_idx}.jpg", value_range=(-1, 1), normalize=True)

    compute_fid(OUTPUT_DIR_OURS, REAL_DIR, OUTPUT_DIR_OURS.parent / f"score_ours_{config.views_per_sample}_{num_latent}.txt", device)


if __name__ == "__main__":
    # create_real()
    # evaluate_uv_parameterized_gan()
    # evaluate_3d_parameterized_gan()
    evaluate_our_gan()
