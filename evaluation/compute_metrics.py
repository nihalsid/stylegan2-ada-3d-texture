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


@hydra.main(config_path='../config', config_name='stylegan2')
def render_latent(config):

    from model.graph_generator_u_deep import Generator
    from model.graph import TwinGraphEncoder
    from dataset.mesh_real_features import FaceGraphMeshDataset
    import math

    config.batch_size = 1
    config.views_per_sample = 9
    config.image_size = 512
    config.render_size = 512
    config.num_mapping_layers = 5
    config.g_channel_base = 32768
    config.g_channel_max = 768

    num_random_latent = 100
    OUTPUT_DIR_OURS_IMAGES = OUTPUT_DIR / "ours_vis" / "images"
    OUTPUT_DIR_OURS_CODES = OUTPUT_DIR / "ours_vis" / "codes"
    OUTPUT_DIR_OURS_IMAGES.mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR_OURS_CODES.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/_epoch=119.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/ema_000076439.pth"

    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

    z_all = torch.randn(num_random_latent, config.batch_size, config.latent_dim).to(device)

    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
    E = TwinGraphEncoder(eval_dataset.num_feats, 1).to(device)
    R = DifferentiableRenderer(config.render_size, "bounds", config.colorspace)
    state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
    G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
    ema = torch.load(CHECKPOINT_EMA, map_location=device)
    ema.copy_to([p for p in G.parameters() if p.requires_grad])
    E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
    G.eval()
    E.eval()

    interesting_indices = [
        0, 1, 2, 5, 7, 8, 9, 11, 13, 15, 24, 41, 48, 55, 61, 97, 99, 107, 119, 123, 159, 195, 211, 218, 279, 287,
        314, 322, 356, 379, 405, 474, 628, 631, 696, 708, 724, 726, 762, 776, 781, 792, 794, 814, 842, 861, 868, 873,
        887, 888, 889, 891, 914, 930, 975, 988, 1064, 1075, 1097, 1151, 1155, 1168, 1170, 1176, 1188, 1222, 1226, 1230,
        1231, 1233, 1237, 1240, 1328, 1337, 1371, 1373, 1445, 1463, 1466, 1469, 1480, 1482, 1502, 1503, 1505, 1510, 1513,
        1517, 1603, 1604, 1620, 1625, 1626, 1642, 1690, 1693, 1702, 1707, 1711, 1740, 1790, 1799, 1808, 1821, 1824, 1829,
        1840, 1852, 1853, 1871, 1875, 1925, 1927, 1952, 1955, 1959, 1962, 1963, 1972, 2018, 2024, 2067, 2095, 2158, 2198,
        2245, 2244, 2250, 2266, 2291, 2311, 2481, 2482, 2620, 2683, 2883, 2891, 2914, 2929, 3018, 3040, 3065, 3146, 3150,
        3154, 3322, 3323, 3482, 3581, 3665, 3669, 3762, 3787, 3817, 3874, 3885, 3932, 3950, 3954, 4848, 4883
    ]
    print("#II:", len(interesting_indices))
    for iter_idx, batch in enumerate(tqdm(eval_loader)):
        if iter_idx not in interesting_indices:
            continue
        eval_batch = to_device(batch, device)
        shape = E(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
        (OUTPUT_DIR_OURS_IMAGES / f"{iter_idx:04d}").mkdir(exist_ok=True)
        (OUTPUT_DIR_OURS_CODES / f"{iter_idx:04d}").mkdir(exist_ok=True)
        for z_idx in range(num_random_latent):
            z = z_all[z_idx]
            fake = G(eval_batch['graph_data'], z, shape, noise_mode='const', )
            fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
            torch.save(z.cpu(), OUTPUT_DIR_OURS_CODES / f"{iter_idx:04d}" / f"{z_idx:04d}.pt")
            save_image(fake_render, OUTPUT_DIR_OURS_IMAGES / f"{iter_idx:04d}" / f"{z_idx:04d}.jpg", nrow=int(math.sqrt(config.views_per_sample)), value_range=(-1, 1), normalize=True)


@hydra.main(config_path='../config', config_name='stylegan2')
def render_mesh(config):
    from model.graph_generator_u import Generator
    from model.graph import GraphEncoder
    from dataset.mesh_real_features import FaceGraphMeshDataset
    import trimesh

    config.batch_size = 1
    config.views_per_sample = 1
    config.image_size = 512
    config.render_size = 512
    config.num_mapping_layers = 5
    config.g_channel_base = 32768
    config.mbstd_on = True

    OUTPUT_DIR_OURS_MESHES = OUTPUT_DIR / "ours_vis" / "meshes"
    OUTPUT_DIR_OURS_CODES = OUTPUT_DIR / "ours_vis" / "codes"
    OUTPUT_DIR_OURS_MESHES.mkdir(exist_ok=True, parents=True)

    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21011752_StyleGAN23D_fg3bgg-lrd1g14-v2m5-1024/checkpoints/_epoch=129.ckpt"

    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
    E = GraphEncoder(eval_dataset.num_feats).to(device)
    state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
    G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
    E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
    G.eval()
    E.eval()

    eval_mesh_ids = [9, 48, 2482, 3065]
    eval_mesh_codes = [[9, 10, 52], [3, 68, 11], [0, 91, 27], [34, 61, 55]]

    with torch.no_grad():
        for iter_idx, batch in enumerate(tqdm(eval_loader)):
            if iter_idx not in eval_mesh_ids:
                continue
            eval_batch = to_device(batch, device)
            shape = E(eval_batch['x'], eval_batch['graph_data'])
            for z_idx in eval_mesh_codes[eval_mesh_ids.index(iter_idx)]:
                z = torch.load(OUTPUT_DIR_OURS_CODES / f'{iter_idx:04d}' / f'{z_idx:04d}.pt').to(device)
                fake = G(eval_batch['graph_data'], z, shape, noise_mode='const', )
                vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                out_mesh.export(OUTPUT_DIR_OURS_MESHES / f"{iter_idx:04d}_{z_idx:04d}.obj")


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
@hydra.main(config_path='../config', config_name='stylegan2')
def evaluate_texturefields_gan(config):
    from dataset.mesh_real_eg3d import FaceGraphMeshDataset
    import trimesh
    import numpy as np

    OUTPUT_DIR_TEXFIELD = OUTPUT_DIR / f"texfield"
    OUTPUT_DIR_TEXFIELD.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda:0")
    config.batch_size = 1
    config.views_per_sample = 1
    config.image_size = 256
    config.render_size = 256
    R = DifferentiableRenderer(config.render_size, "bounds", config.colorspace)

    chair_data_meshes = Path("/cluster_HDD/gondor/ysiddiqui/CADTextures/runs/texture_fields/GAN/chair/eval_fix/fake")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=0)
    for iter_idx, batch in enumerate(tqdm(eval_loader)):
        eval_batch = to_device(batch, device)
        fake_mesh = trimesh.load(chair_data_meshes / f"{batch['name'][0]}.obj", process=False)
        fake_colors = fake_mesh.visual.vertex_colors[fake_mesh.faces[:, 0], :].astype(np.float32) + fake_mesh.visual.vertex_colors[fake_mesh.faces[:, 1], :].astype(np.float32) \
                      + fake_mesh.visual.vertex_colors[fake_mesh.faces[:, 2], :].astype(np.float32) + fake_mesh.visual.vertex_colors[fake_mesh.faces[:, 3], :].astype(np.float32)
        fake_colors = fake_colors[:, :3] / 4
        fake = torch.from_numpy(fake_colors).to(device) / 127.5 - 1
        fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
        for batch_idx in range(fake_render.shape[0]):
            save_image(fake_render[batch_idx], OUTPUT_DIR_TEXFIELD / f"{iter_idx}_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)

    compute_fid(OUTPUT_DIR_TEXFIELD, REAL_DIR, OUTPUT_DIR_TEXFIELD.parent / f"score_texfield.txt", device)


# Triplane Implicit Parameterization
@hydra.main(config_path='../config', config_name='stylegan2')
def evaluate_triplane_gan(config):
    from model.eg3d.generator import Generator
    from model.styleganvox import SDFEncoder
    from dataset.mesh_real_eg3d import FaceGraphMeshDataset
    config.batch_size = 1
    config.views_per_sample = 4
    config.image_size = 256
    config.render_size = 256
    config.num_mapping_layers = 8
    num_latent = 4

    OUTPUT_DIR_EG3D = OUTPUT_DIR / f"eg3d_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_EG3D.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/16022208_StyleGAN23D_eg3d_geo_condition/checkpoints/_epoch=59.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/16022208_StyleGAN23D_eg3d_geo_condition/checkpoints/ema_000076439.pth"

    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.image_size, 96, c_dim=256).to(device)
    E = SDFEncoder(1).to(device)
    R = DifferentiableRenderer(config.render_size, "bounds", config.colorspace)
    state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
    G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
    ema = torch.load(CHECKPOINT_EMA, map_location=device)
    ema.copy_to([p for p in G.parameters() if p.requires_grad])
    E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
    G.eval()
    E.eval()

    for iter_idx, batch in enumerate(tqdm(eval_loader)):
        eval_batch = to_device(batch, device)
        code = E(eval_batch['sdf_x'])
        shape = code[4].mean((2, 3, 4))
        shape_grid = code[3]
        for z_idx in range(num_latent):
            z = torch.randn(config.batch_size, config.latent_dim).to(device)
            fake = G(eval_batch['faces'], z, shape, shape_grid, noise_mode='const')
            fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
            for batch_idx in range(fake_render.shape[0]):
                save_image(fake_render[batch_idx], OUTPUT_DIR_EG3D / f"{iter_idx}_{batch_idx}_{z_idx}.jpg", value_range=(-1, 1), normalize=True)

    compute_fid(OUTPUT_DIR_EG3D, REAL_DIR, OUTPUT_DIR_EG3D.parent / f"score_eg3d_{config.views_per_sample}_{num_latent}.txt", device)


# Ours old
@hydra.main(config_path='../config', config_name='stylegan2')
def evaluate_our_gan_old(config):
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


# Ours
@hydra.main(config_path='../config', config_name='stylegan2')
def evaluate_our_gan(config):
    from model.graph_generator_u_deep import Generator
    from model.graph import TwinGraphEncoder
    from dataset.mesh_real_features import FaceGraphMeshDataset
    config.batch_size = 1
    config.views_per_sample = 4
    config.image_size = 256
    config.render_size = 1024
    config.num_mapping_layers = 5
    config.g_channel_base = 32768
    config.g_channel_max = 768

    num_latent = 1

    OUTPUT_DIR_OURS = OUTPUT_DIR / f"ours_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/_epoch=119.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/ema_000076439.pth"

    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
    E = TwinGraphEncoder(eval_dataset.num_feats, 1).to(device)
    R = DifferentiableRenderer(config.render_size, "bounds", config.colorspace)
    state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
    G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
    ema = torch.load(CHECKPOINT_EMA, map_location=device)
    ema.copy_to([p for p in G.parameters() if p.requires_grad])
    E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
    G.eval()
    E.eval()

    for iter_idx, batch in enumerate(tqdm(eval_loader)):
        eval_batch = to_device(batch, device)
        shape = E(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
        for z_idx in range(num_latent):
            z = torch.randn(config.batch_size, config.latent_dim).to(device)
            fake = G(eval_batch['graph_data'], z, shape, noise_mode='const')
            fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
            for batch_idx in range(fake_render.shape[0]):
                save_image(fake_render[batch_idx], OUTPUT_DIR_OURS / f"{iter_idx}_{batch_idx}_{z_idx}.jpg", value_range=(-1, 1), normalize=True)

    compute_fid(OUTPUT_DIR_OURS, REAL_DIR, OUTPUT_DIR_OURS.parent / f"score_ours_{config.views_per_sample}_{num_latent}.txt", device)


@hydra.main(config_path='../config', config_name='stylegan2')
def ablate_our_gan_no_feat(config):
    from model.graph_generator_u_deep import Generator
    from model.graph import GraphEncoder
    from dataset.mesh_real_features import FaceGraphMeshDataset
    config.batch_size = 1
    config.views_per_sample = 4
    config.image_size = 256
    config.render_size = 256
    config.num_mapping_layers = 5

    num_latent = 1

    OUTPUT_DIR_OURS = OUTPUT_DIR / f"ablate_no_feat_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/18020231_StyleGAN23D_bgg-nofeat-lrd1g14-v2m5-256/checkpoints/_epoch=99.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/18020231_StyleGAN23D_bgg-nofeat-lrd1g14-v2m5-256/checkpoints/ema_000127399.pth"

    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
    E = GraphEncoder(eval_dataset.num_feats).to(device)
    R = DifferentiableRenderer(config.render_size, "bounds", config.colorspace)
    state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
    G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
    ema = torch.load(CHECKPOINT_EMA, map_location=device)
    ema.copy_to([p for p in G.parameters() if p.requires_grad])
    E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
    G.eval()
    E.eval()

    for iter_idx, batch in enumerate(tqdm(eval_loader)):
        eval_batch = to_device(batch, device)
        shape = [torch.zeros_like(x) for x in E(eval_batch['x'], eval_batch['graph_data'])]
        for z_idx in range(num_latent):
            z = torch.randn(config.batch_size, config.latent_dim).to(device)
            fake = G(eval_batch['graph_data'], z, shape, noise_mode='const')
            fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
            for batch_idx in range(fake_render.shape[0]):
                save_image(fake_render[batch_idx], OUTPUT_DIR_OURS / f"{iter_idx}_{batch_idx}_{z_idx}.jpg", value_range=(-1, 1), normalize=True)

    compute_fid(OUTPUT_DIR_OURS, REAL_DIR, OUTPUT_DIR_OURS.parent / f"score_ablate_no_feat_{config.views_per_sample}_{num_latent}.txt", device)


@hydra.main(config_path='../config', config_name='stylegan2')
def ablate_our_gan_1_view(config):
    from model.graph_generator_u_deep import Generator
    from model.graph import GraphEncoder
    from dataset.mesh_real_features import FaceGraphMeshDataset
    config.batch_size = 1
    config.views_per_sample = 4
    config.image_size = 256
    config.render_size = 512
    config.num_mapping_layers = 8
    config.g_channel_base = 32768

    num_latent = 1

    OUTPUT_DIR_OURS = OUTPUT_DIR / f"ablate_1view_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/18020916_StyleGAN23D_fg3bgg-notwin-big-lrd1g14-v1m5-512/checkpoints/_epoch=119.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/18020916_StyleGAN23D_fg3bgg-notwin-big-lrd1g14-v1m5-512/checkpoints/ema_000076439.pth"

    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
    E = GraphEncoder(eval_dataset.num_feats).to(device)
    R = DifferentiableRenderer(config.render_size, "bounds", config.colorspace)
    state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
    G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
    ema = torch.load(CHECKPOINT_EMA, map_location=device)
    ema.copy_to([p for p in G.parameters() if p.requires_grad])
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

    compute_fid(OUTPUT_DIR_OURS, REAL_DIR, OUTPUT_DIR_OURS.parent / f"score_ablate_1view_{config.views_per_sample}_{num_latent}.txt", device)


@hydra.main(config_path='../config', config_name='stylegan2')
def ablate_our_gan_64(config):
    from model.graph_generator_u_deep import Generator
    from model.graph import GraphEncoder
    from dataset.mesh_real_features import FaceGraphMeshDataset
    config.batch_size = 1
    config.views_per_sample = 4
    config.image_size = 256
    config.render_size = 1024
    config.num_mapping_layers = 5
    config.g_channel_base = 32768

    num_latent = 1

    OUTPUT_DIR_OURS = OUTPUT_DIR / f"ablate_64_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/19020403_StyleGAN23D_fg3bgg-notwin-lrd1g14-v2m5-64/checkpoints/_epoch=119.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/19020403_StyleGAN23D_fg3bgg-notwin-lrd1g14-v2m5-64/checkpoints/ema_000076439.pth"

    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
    E = GraphEncoder(eval_dataset.num_feats).to(device)
    R = DifferentiableRenderer(config.render_size, "bounds", config.colorspace)
    state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
    G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
    ema = torch.load(CHECKPOINT_EMA, map_location=device)
    ema.copy_to([p for p in G.parameters() if p.requires_grad])
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

    compute_fid(OUTPUT_DIR_OURS, REAL_DIR, OUTPUT_DIR_OURS.parent / f"score_ablate_64_{config.views_per_sample}_{num_latent}.txt", device)


@hydra.main(config_path='../config', config_name='stylegan2')
def ablate_our_gan_128(config):
    from model.graph_generator_u_deep import Generator
    from model.graph import GraphEncoder
    from dataset.mesh_real_features import FaceGraphMeshDataset
    config.batch_size = 1
    config.views_per_sample = 4
    config.image_size = 256
    config.render_size = 1024
    config.num_mapping_layers = 5
    config.g_channel_base = 32768

    num_latent = 1

    OUTPUT_DIR_OURS = OUTPUT_DIR / f"ablate_128_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/18020722_StyleGAN23D_fg3bgg-notwin-lrd1g14-v2m5-128/checkpoints/_epoch=119.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/18020722_StyleGAN23D_fg3bgg-notwin-lrd1g14-v2m5-128/checkpoints/ema_000076439.pth"

    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
    E = GraphEncoder(eval_dataset.num_feats).to(device)
    R = DifferentiableRenderer(config.render_size, "bounds", config.colorspace)
    state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
    G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
    ema = torch.load(CHECKPOINT_EMA, map_location=device)
    ema.copy_to([p for p in G.parameters() if p.requires_grad])
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

    compute_fid(OUTPUT_DIR_OURS, REAL_DIR, OUTPUT_DIR_OURS.parent / f"score_ablate_128_{config.views_per_sample}_{num_latent}.txt", device)


@hydra.main(config_path='../config', config_name='stylegan2')
def ablate_our_gan_256(config):
    from model.graph_generator_u_deep import Generator
    from model.graph import GraphEncoder
    from dataset.mesh_real_features import FaceGraphMeshDataset
    config.batch_size = 1
    config.views_per_sample = 4
    config.image_size = 256
    config.render_size = 512
    config.num_mapping_layers = 5
    config.g_channel_base = 32768

    num_latent = 1

    OUTPUT_DIR_OURS = OUTPUT_DIR / f"ablate_256_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/18020840_StyleGAN23D_fg3bgg-notwin-lrd1g14-v2m5-256/checkpoints/_epoch=119.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/18020840_StyleGAN23D_fg3bgg-notwin-lrd1g14-v2m5-256/checkpoints/ema_000076439.pth"

    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
    E = GraphEncoder(eval_dataset.num_feats).to(device)
    R = DifferentiableRenderer(config.render_size, "bounds", config.colorspace)
    state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
    G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
    ema = torch.load(CHECKPOINT_EMA, map_location=device)
    ema.copy_to([p for p in G.parameters() if p.requires_grad])
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

    compute_fid(OUTPUT_DIR_OURS, REAL_DIR, OUTPUT_DIR_OURS.parent / f"score_ablate_256_{config.views_per_sample}_{num_latent}.txt", device)


@hydra.main(config_path='../config', config_name='stylegan2')
def ablate_our_gan_position(config):
    from model.graph_generator_u import Generator
    from model.graph import GraphEncoder
    from dataset.mesh_real_features import FaceGraphMeshDataset
    config.batch_size = 1
    config.views_per_sample = 4
    config.image_size = 256
    config.render_size = 512
    config.num_mapping_layers = 5
    config.g_channel_base = 32768
    config.g_channel_max = 512
    config.features = "position"
    num_latent = 1

    OUTPUT_DIR_OURS = OUTPUT_DIR / f"ablate_position_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22011039_StyleGAN23D_pos_fg3bgg-lrd1g14-v2m5-256/checkpoints/_epoch=89.ckpt"

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

    compute_fid(OUTPUT_DIR_OURS, REAL_DIR, OUTPUT_DIR_OURS.parent / f"score_ablate_position_{config.views_per_sample}_{num_latent}.txt", device)


@hydra.main(config_path='../config', config_name='stylegan2')
def ablate_our_gan_laplacian(config):
    from model.graph_generator_u import Generator
    from model.graph import GraphEncoder
    from dataset.mesh_real_features import FaceGraphMeshDataset
    config.batch_size = 1
    config.views_per_sample = 4
    config.image_size = 256
    config.render_size = 512
    config.num_mapping_layers = 5
    config.g_channel_base = 32768
    config.g_channel_max = 512
    config.features = "laplacian"
    num_latent = 1

    OUTPUT_DIR_OURS = OUTPUT_DIR / f"ablate_laplacian_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22011317_StyleGAN23D_lap_fg3bgg-lrd1g14-v2m5-256/checkpoints/_epoch=79.ckpt"

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

    compute_fid(OUTPUT_DIR_OURS, REAL_DIR, OUTPUT_DIR_OURS.parent / f"score_ablate_laplacian_{config.views_per_sample}_{num_latent}.txt", device)


@hydra.main(config_path='../config', config_name='stylegan2')
def ablate_our_gan_ff2(config):
    from model.graph_generator_u import Generator
    from model.graph import GraphEncoder
    from dataset.mesh_real_features import FaceGraphMeshDataset
    config.batch_size = 1
    config.views_per_sample = 4
    config.image_size = 256
    config.render_size = 512
    config.num_mapping_layers = 5
    config.g_channel_base = 32768
    config.g_channel_max = 512
    config.features = "ff2"
    num_latent = 1

    OUTPUT_DIR_OURS = OUTPUT_DIR / f"ablate_ff2_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22011107_StyleGAN23D_ff2_fg3bgg-lrd1g14-v2m5-256/checkpoints/_epoch=139.ckpt"

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

    compute_fid(OUTPUT_DIR_OURS, REAL_DIR, OUTPUT_DIR_OURS.parent / f"score_ablate_ff2_{config.views_per_sample}_{num_latent}.txt", device)


@hydra.main(config_path='../config', config_name='stylegan2')
def ablate_our_gan_curvature(config):
    from model.graph_generator_u import Generator
    from model.graph import GraphEncoder
    from dataset.mesh_real_features import FaceGraphMeshDataset
    config.batch_size = 1
    config.views_per_sample = 4
    config.image_size = 256
    config.render_size = 512
    config.num_mapping_layers = 5
    config.g_channel_base = 32768
    config.g_channel_max = 512
    config.features = "curvature"
    num_latent = 1

    OUTPUT_DIR_OURS = OUTPUT_DIR / f"ablate_curvature_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22011243_StyleGAN23D_curv_fg3bgg-lrd1g14-v2m5-256/checkpoints/_epoch=129.ckpt"

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

    compute_fid(OUTPUT_DIR_OURS, REAL_DIR, OUTPUT_DIR_OURS.parent / f"score_ablate_curvature_{config.views_per_sample}_{num_latent}.txt", device)


@hydra.main(config_path='../config', config_name='stylegan2')
def ablate_our_gan_normals(config):
    from model.graph_generator_u_deep import Generator
    from model.graph import GraphEncoder
    from dataset.mesh_real_features import FaceGraphMeshDataset
    config.batch_size = 1
    config.views_per_sample = 4
    config.image_size = 256
    config.render_size = 1024
    config.num_mapping_layers = 5
    config.g_channel_base = 32768
    config.g_channel_max = 768

    num_latent = 1

    OUTPUT_DIR_OURS = OUTPUT_DIR / f"ablate_normals_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/16021631_StyleGAN23D_fg3bgg-notwin-big-lrd1g14-v2m5-1K512/checkpoints/_epoch=139.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/16021631_StyleGAN23D_fg3bgg-notwin-big-lrd1g14-v2m5-1K512/checkpoints/ema_000089179.pth"

    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
    E = GraphEncoder(eval_dataset.num_feats).to(device)
    R = DifferentiableRenderer(config.render_size, "bounds", config.colorspace)
    state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
    G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
    ema = torch.load(CHECKPOINT_EMA, map_location=device)
    ema.copy_to([p for p in G.parameters() if p.requires_grad])
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

    compute_fid(OUTPUT_DIR_OURS, REAL_DIR, OUTPUT_DIR_OURS.parent / f"score_ablate_normals_{config.views_per_sample}_{num_latent}.txt", device)


if __name__ == "__main__":
    # create_real()
    # evaluate_uv_parameterized_gan()
    # evaluate_3d_parameterized_gan()
    # evaluate_our_gan()
    # evaluate_triplane_gan()
    # evaluate_texturefields_gan()
    render_latent()
    # render_mesh()
    # ablate_our_gan_1_view()
    # ablate_our_gan_no_feat()
    # ablate_our_gan_128()
    # ablate_our_gan_256()
    # ablate_our_gan_64()
    # ablate_our_gan_position()
    # ablate_our_gan_laplacian()
    # ablate_our_gan_ff2()
    # ablate_our_gan_normals()
    # ablate_our_gan_curvature()
