from pathlib import Path

import torch
from util.misc import compute_fid
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import GraphDataLoader, to_device, to_vertex_colors_scatter
from model.differentiable_renderer import DifferentiableRenderer
import hydra

from util.misc import get_parameters_from_state_dict

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
    from model.graph_generator_u_deep import Generator
    from model.graph import TwinGraphEncoder
    from dataset.mesh_real_features import FaceGraphMeshDataset
    import trimesh

    config.batch_size = 1
    config.views_per_sample = 1
    config.image_size = 512
    config.render_size = 512
    config.num_mapping_layers = 5
    config.g_channel_base = 32768
    config.g_channel_max = 768

    OUTPUT_DIR_OURS_MESHES = OUTPUT_DIR / "ours_vis" / "meshes"
    OUTPUT_DIR_OURS_CODES = OUTPUT_DIR / "ours_vis" / "codes"
    OUTPUT_DIR_OURS_MESHES.mkdir(exist_ok=True, parents=True)

    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/_epoch=119.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/ema_000076439.pth"

    # eval_mesh_ids = [24, 889, 1328]
    # eval_mesh_codes = [[1, 34, 61, 92, 52], [3, 24, 29, 38], [1, 26, 34, 41, 79, 90]]

    eval_mesh_ids = [889]#[1176, 1517]#[405, 776]
    eval_mesh_codes = [[100]]#[[83, 70, 58, 44, 18, 0]]#[[27, 42, 60], [27, 42, 60]]

    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_dataset.items = [eval_dataset.items[i] for i in eval_mesh_ids]
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

    E = TwinGraphEncoder(eval_dataset.num_feats, 1).to(device)
    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=E.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
    state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
    G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
    ema = torch.load(CHECKPOINT_EMA, map_location=device)
    ema.copy_to([p for p in G.parameters() if p.requires_grad])
    E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))

    G.eval()
    E.eval()

    # eval_mesh_ids = [9, 48, 2482, 3065]
    # eval_mesh_codes = [[9, 10, 52], [3, 68, 11], [0, 91, 27], [34, 61, 55]]

    with torch.no_grad():
        for iter_idx, batch in enumerate(tqdm(eval_loader)):
            eval_batch = to_device(batch, device)
            shape = E(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
            for z_idx in eval_mesh_codes[iter_idx]:
                z = torch.load(OUTPUT_DIR_OURS_CODES / f'{eval_mesh_ids[iter_idx]:04d}' / f'{z_idx:04d}.pt').to(device)
                fake = G(eval_batch['graph_data'], z, shape, noise_mode='const', )
                vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                out_mesh.export(OUTPUT_DIR_OURS_MESHES / f"{eval_mesh_ids[iter_idx]:04d}_{z_idx:04d}.obj")


def render_faces(R, face_colors, batch, render_size, image_size):
    rendered_color = R.render(batch['vertices'], batch['indices'], to_vertex_colors_scatter(face_colors, batch), batch["ranges"].cpu(), resolution=render_size)
    ret_val = rendered_color.permute((0, 3, 1, 2))
    if render_size != image_size:
        ret_val = torch.nn.functional.interpolate(ret_val, (image_size, image_size), mode='bilinear', align_corners=True)
    return ret_val


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


# UV Parameterization - Norm Aligned
@hydra.main(config_path='../config', config_name='stylegan2')
def evaluate_uv_normalign_parameterized_gan(config):
    config.views_per_sample = 4
    config.image_size = 128
    config.render_size = 512
    num_latent = 1

    from model.stylegan2.generator import Generator
    from dataset.mesh_real_features_uv import FaceGraphMeshDataset
    OUTPUT_DIR_UV = OUTPUT_DIR / f"uv_na_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_UV.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22021749_StyleGAN23D_uv_normaligned_silhoutte_256/checkpoints/_epoch=79.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22021749_StyleGAN23D_uv_normaligned_silhoutte_256/checkpoints/ema_000203839.pth"

    def render(r_texture_atlas, r_batch):
        r_texture_atlas = r_texture_atlas.reshape((-1, r_texture_atlas.shape[2], r_texture_atlas.shape[3], r_texture_atlas.shape[4]))
        vertices_mapped = r_texture_atlas[r_batch["uv"][:, 0].long(), :, (r_batch["uv"][:, 1] * config.image_size).long(), (r_batch["uv"][:, 2] * config.image_size).long()]
        vertices_mapped = torch.cat([vertices_mapped, torch.ones((vertices_mapped.shape[0], 1), device=vertices_mapped.device)], dim=-1)
        rendered_color = R.render(r_batch['vertices'], r_batch['indices'], vertices_mapped, r_batch["ranges"].cpu(), resolution=config.render_size).permute((0, 3, 1, 2))
        rendered_color = torch.nn.functional.interpolate(rendered_color, (256, 256), mode='bilinear', align_corners=True)
        return rendered_color

    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, shuffle=False, drop_last=True, num_workers=0)
    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.image_size, 3).to(device)
    R = DifferentiableRenderer(config.render_size, "bounds", config.colorspace)
    G.load_state_dict(get_parameters_from_state_dict(torch.load(CHECKPOINT, map_location=device)["state_dict"], "G"))
    G.eval()
    ema = torch.load(CHECKPOINT_EMA, map_location=device)
    ema.copy_to([p for p in G.parameters() if p.requires_grad])
    for iter_idx, batch in enumerate(tqdm(eval_loader)):
        eval_batch = to_device(batch, device)
        silhoutte = eval_batch['silhoutte'].reshape([config.batch_size * 6, 2, config.image_size, config.image_size])
        for z_idx in range(num_latent):
            z = torch.randn(config.batch_size, config.latent_dim).to(device)
            atlas = G(z, silhoutte, noise_mode='const')
            fake_render = render(atlas, eval_batch).cpu()
            for batch_idx in range(fake_render.shape[0]):
                save_image(fake_render[batch_idx], OUTPUT_DIR_UV / f"{iter_idx}_{batch_idx}_{z_idx}.jpg", value_range=(-1, 1), normalize=True)
    compute_fid(OUTPUT_DIR_UV, REAL_DIR, OUTPUT_DIR_UV.parent / f"score_uv_na_{config.views_per_sample}_{num_latent}.txt", device)


# UV Parameterization - Ours
@hydra.main(config_path='../config', config_name='stylegan2')
def evaluate_uv_ours_parameterized_gan(config):
    config.views_per_sample = 4
    config.image_size = 256
    config.render_size = 512
    num_latent = 1

    from model.uv.generator import Generator, NormalEncoder
    from dataset.mesh_real_features_uv_ours import FaceGraphMeshDataset, split_tensor_into_six
    OUTPUT_DIR_UV = OUTPUT_DIR / f"uv_ours_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_UV.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22021749_StyleGAN23D_uv_ours_silhoutte_256/checkpoints/_epoch=59.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22021749_StyleGAN23D_uv_ours_silhoutte_256/checkpoints/ema_000152879.pth"

    def render(r_texture_atlas, r_batch):
        r_texture_atlas = split_tensor_into_six(r_texture_atlas)
        vertices_mapped = r_texture_atlas[r_batch["uv"][:, 0].long(), :, (r_batch["uv"][:, 1] * (config.image_size // 2)).long(), (r_batch["uv"][:, 2] * (config.image_size // 3)).long()]
        vertices_mapped = torch.cat([vertices_mapped, torch.ones((vertices_mapped.shape[0], 1), device=vertices_mapped.device)], dim=-1)
        rendered_color = R.render(r_batch['vertices'], r_batch['indices'], vertices_mapped, r_batch["ranges"].cpu())
        rendered_color = torch.nn.functional.interpolate(rendered_color.permute((0, 3, 1, 2)), (256, 256), mode='bilinear', align_corners=True)
        return rendered_color

    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, shuffle=False, drop_last=True, num_workers=0)
    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.image_size, 3).to(device)
    R = DifferentiableRenderer(config.render_size, "bounds", config.colorspace)
    E = NormalEncoder(3).to(device)
    state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
    G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
    E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
    G.eval()
    E.eval()
    ema = torch.load(CHECKPOINT_EMA, map_location=device)
    ema.copy_to([p for p in G.parameters() if p.requires_grad])
    for iter_idx, batch in enumerate(tqdm(eval_loader)):
        eval_batch = to_device(batch, device)
        code = E(eval_batch['uv_normals'])
        for z_idx in range(num_latent):
            z = torch.randn(config.batch_size, config.latent_dim).to(device)
            atlas = G(z, code, noise_mode='const')
            fake_render = render(atlas, eval_batch).cpu()
            for batch_idx in range(fake_render.shape[0]):
                save_image(fake_render[batch_idx], OUTPUT_DIR_UV / f"{iter_idx}_{batch_idx}_{z_idx}.jpg", value_range=(-1, 1), normalize=True)
    compute_fid(OUTPUT_DIR_UV, REAL_DIR, OUTPUT_DIR_UV.parent / f"score_uv_ours_{config.views_per_sample}_{num_latent}.txt", device)


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
    config.render_size = 512
    config.num_mapping_layers = 8
    num_latent = 1

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
    config.render_size = 512
    config.num_mapping_layers = 5
    config.g_channel_base = 32768
    config.g_channel_max = 768
    config.dataset_path = "/cluster/gimli/ysiddiqui/CADTextures/Photoshape/shapenet-chairs-manifold-highres-part_processed_color_left"
    num_latent = 1

    # OUTPUT_DIR_OURS = OUTPUT_DIR / f"ours_256-2_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_OURS = OUTPUT_DIR / f"ours_subset_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    # CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/_epoch=119.ckpt"
    # CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/ema_000076439.pth"
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/28021405_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512-subset/checkpoints/_epoch=149.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/28021405_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512-subset/checkpoints/ema_000065549.pth"
    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

    E = TwinGraphEncoder(eval_dataset.num_feats, 1).to(device)
    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=E.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
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

    compute_fid(OUTPUT_DIR_OURS, REAL_DIR, OUTPUT_DIR_OURS.parent / f"score_ours-subset_{config.views_per_sample}_{num_latent}.txt", device)


@hydra.main(config_path='../config', config_name='stylegan2')
def evaluate_our_gan_texconv(config):
    from dataset.mesh_real_features import FaceGraphMeshDataset
    from model.graph import GraphEncoder, FaceConv, TextureConv
    from model.graph_generator_u_texconv import Generator
    config.batch_size = 1
    config.views_per_sample = 4
    config.image_size = 256
    config.render_size = 512
    config.num_mapping_layers = 5
    config.g_channel_base = 65536
    config.g_channel_max = 768
    config.conv_aggregation = "mean"
    config.enc_conv = "texture"
    num_latent = 1

    OUTPUT_DIR_OURS = OUTPUT_DIR / f"ours_texconv_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22011007_StyleGAN23D_tcmean-etc_fg6bgg-lrd1g14-v2m5-256/checkpoints/_epoch=109.ckpt"

    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

    E = GraphEncoder(eval_dataset.num_feats, conv_layer=FaceConv if config.enc_conv == 'face' else TextureConv).to(device)
    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, aggregation_function=config.conv_aggregation, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
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

    compute_fid(OUTPUT_DIR_OURS, REAL_DIR, OUTPUT_DIR_OURS.parent / f"score_texconv_{config.views_per_sample}_{num_latent}.txt", device)


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
def ablate_our_gan_6k(config):
    from model.graph_generator_u_6k import Generator
    from model.graph import GraphEncoder
    from dataset.mesh_real_features import FaceGraphMeshDataset
    config.batch_size = 1
    config.views_per_sample = 4
    config.image_size = 256
    config.render_size = 512
    config.num_mapping_layers = 5
    config.g_channel_base = 32768
    num_latent = 1

    OUTPUT_DIR_OURS = OUTPUT_DIR / f"ablate_6k_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22021833_StyleGAN23D_lowres6K-notwin-lrd1g14-v2m5-256/checkpoints/_epoch=44.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22021833_StyleGAN23D_lowres6K-notwin-lrd1g14-v2m5-256/checkpoints/ema_000057329.pth"

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
        shape = E(batch['x'], batch['graph_data'])
        for z_idx in range(num_latent):
            z = torch.randn(config.batch_size, config.latent_dim).to(device)
            fake = G(eval_batch['graph_data'], z, shape, noise_mode='const')
            fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
            for batch_idx in range(fake_render.shape[0]):
                save_image(fake_render[batch_idx], OUTPUT_DIR_OURS / f"{iter_idx}_{batch_idx}_{z_idx}.jpg", value_range=(-1, 1), normalize=True)

    compute_fid(OUTPUT_DIR_OURS, REAL_DIR, OUTPUT_DIR_OURS.parent / f"score_ablate_6k_{config.views_per_sample}_{num_latent}.txt", device)


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


def create_meshes_for_baselines():
    import shutil
    import trimesh

    selected_shapes = [15, 97, 2482, 3150, 3665]
    comparisons_dir = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/photoshape/compare_sota")
    device = torch.device("cuda:0")

    for idx in selected_shapes:
        (comparisons_dir / f"{idx:05d}").mkdir(exist_ok=True, parents=True)

    # Texture Field
    @hydra.main(config_path='../config', config_name='stylegan2')
    def texture_fields(config):
        from dataset.mesh_real_eg3d import FaceGraphMeshDataset
        eval_dataset = FaceGraphMeshDataset(config)
        chair_data_meshes = Path("/cluster_HDD/gondor/ysiddiqui/CADTextures/runs/texture_fields/GAN/chair/eval_fix/fake")
        for idx in selected_shapes:
            shutil.copyfile(chair_data_meshes / f"{eval_dataset[idx]['name']}.obj", comparisons_dir / f"{idx:05d}" / "texture_fields.obj")

    # LTG
    @hydra.main(config_path='../config', config_name='stylegan2')
    def LTG(config):
        from model.stylegan2.generator import Generator
        from dataset.mesh_real_features_uv import FaceGraphMeshDataset
        config.batch_size = 1
        config.views_per_sample = 1
        config.image_size = 128
        config.render_size = 512
        num_latent = 10
        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22021749_StyleGAN23D_uv_normaligned_silhoutte_256/checkpoints/_epoch=79.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22021749_StyleGAN23D_uv_normaligned_silhoutte_256/checkpoints/ema_000203839.pth"

        def map_to_vertex(r_texture_atlas, r_batch):
            r_texture_atlas = r_texture_atlas.reshape((-1, r_texture_atlas.shape[2], r_texture_atlas.shape[3], r_texture_atlas.shape[4]))
            vertices_mapped = r_texture_atlas[r_batch["uv"][:, 0].long(), :, (r_batch["uv"][:, 1] * config.image_size).long(), (r_batch["uv"][:, 2] * config.image_size).long()]
            vertices_mapped = torch.cat([vertices_mapped, torch.ones((vertices_mapped.shape[0], 1), device=vertices_mapped.device)], dim=-1)
            return vertices_mapped

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [x for iidx, x in enumerate(eval_dataset.items) if iidx in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, shuffle=False, drop_last=True, num_workers=0)
        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.image_size, 3).to(device)
        G.load_state_dict(get_parameters_from_state_dict(torch.load(CHECKPOINT, map_location=device)["state_dict"], "G"))
        G.eval()
        ema = torch.load(CHECKPOINT_EMA, map_location=device)
        ema.copy_to([p for p in G.parameters() if p.requires_grad])

        for iter_idx, batch in enumerate(tqdm(eval_loader)):
            eval_batch = to_device(batch, device)
            silhoutte = eval_batch['silhoutte'].reshape([config.batch_size * 6, 2, config.image_size, config.image_size])
            for z_idx in range(num_latent):
                z = torch.randn(config.batch_size, config.latent_dim).to(device)
                atlas = G(z, silhoutte, noise_mode='const')
                fake_render = map_to_vertex(atlas, eval_batch).cpu()
                fake_render = ((torch.clamp(fake_render, -1, 1) * 0.5 + 0.5) * 255).int()
                mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=fake_render.numpy(), process=False)
                out_mesh.export(comparisons_dir / f"{selected_shapes[iter_idx]:05d}" / f"ltg_{z_idx:02d}.obj")

    # SPSG
    @hydra.main(config_path='../config', config_name='stylegan2')
    def SPSG(config):
        config.batch_size = 1
        config.views_per_sample = 1
        config.image_size = 256
        config.render_size = 256

        num_latent = 10

        from MinkowskiEngine.MinkowskiOps import MinkowskiToDenseTensor
        from MinkowskiEngine.MinkowskiSparseTensor import SparseTensor
        from model.styleganvox_sparse import SDFEncoder
        from model.styleganvox_sparse.generator import Generator
        from dataset.mesh_real_sdfgrid_sparse import SparseSDFGridDataset
        from dataset.mesh_real_sdfgrid_sparse import Collater
        from dataset.mesh_real_features_uv import FaceGraphMeshDataset
        import marching_cubes as mc

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/05020007_StyleGAN23D_sparsereal-bgg-lrd1g14-m5-256/checkpoints/_epoch=24.ckpt"

        eval_dataset_ltg = FaceGraphMeshDataset(config)
        config.dataset_path = "/cluster/gimli/ysiddiqui/CADTextures/Photoshape/shapenet-chairs-sdfgrid/"
        eval_dataset = SparseSDFGridDataset(config)
        eval_dataset.items = eval_dataset_ltg.items
        eval_dataset.items = [x for iidx, x in enumerate(eval_dataset.items) if iidx in selected_shapes]
        eval_loader = DataLoader(eval_dataset, config.batch_size, shuffle=False, drop_last=True, num_workers=0, collate_fn=Collater([], []))

        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, 64, 3).to(device)
        E = SDFEncoder(1).to(device)
        state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
        G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
        G.eval()
        E.eval()

        to_dense = MinkowskiToDenseTensor(torch.Size([config.batch_size, 3, 128, 128, 128]))

        for iter_idx, batch in enumerate(tqdm(eval_loader)):
            eval_batch = to_device(batch, device)
            shape = E(eval_batch['x_dense_064'])
            for z_idx in range(num_latent):
                z = torch.randn(config.batch_size, config.latent_dim).to(device)
                csdf = G(z, eval_batch['sparse_data_064'][0].long(), eval_batch['sparse_data'][0].long(), shape, noise_mode='const')
                csdf_dense = torch.clamp(to_dense(SparseTensor(csdf, eval_batch['sparse_data'][0].int())).squeeze(0).permute((1, 2, 3, 0)), -1.0, 1.0).cpu().numpy() * 0.5 + 0.5
                tsdf_dense = batch['x_dense'].squeeze(0).squeeze(0).permute((2, 1, 0)).cpu().numpy()
                vertices, triangles = mc.marching_cubes_color(tsdf_dense, csdf_dense, 0)
                vertices[:, :3] = vertices[:, :3] / 128 - 0.5
                mc.export_obj(vertices, triangles, comparisons_dir / f"{selected_shapes[iter_idx]:05d}" / f"grid_{z_idx:02d}.obj")

    # EG3D
    @hydra.main(config_path='../config', config_name='stylegan2')
    def EG3D(config):
        from model.eg3d.generator import Generator
        from model.styleganvox import SDFEncoder
        from dataset.mesh_real_eg3d import FaceGraphMeshDataset
        config.batch_size = 1
        config.views_per_sample = 1
        config.image_size = 256
        config.render_size = 512
        config.num_mapping_layers = 8
        num_latent = 10

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/16022208_StyleGAN23D_eg3d_geo_condition/checkpoints/_epoch=59.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/16022208_StyleGAN23D_eg3d_geo_condition/checkpoints/ema_000076439.pth"

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [x for iidx, x in enumerate(eval_dataset.items) if iidx in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.image_size, 96, c_dim=256).to(device)
        E = SDFEncoder(1).to(device)
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
                vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                out_mesh.export(comparisons_dir / f"{selected_shapes[iter_idx]:05d}" / f"eg3d_{z_idx:02d}.obj")

    # Ours UV
    @hydra.main(config_path='../config', config_name='stylegan2')
    def OursUV(config):
        config.batch_size = 1
        config.views_per_sample = 1
        config.image_size = 256
        config.render_size = 512
        num_latent = 10

        from model.uv.generator import Generator, NormalEncoder
        from dataset.mesh_real_features_uv_ours import FaceGraphMeshDataset, split_tensor_into_six

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22021749_StyleGAN23D_uv_ours_silhoutte_256/checkpoints/_epoch=59.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22021749_StyleGAN23D_uv_ours_silhoutte_256/checkpoints/ema_000152879.pth"

        def map_to_vertex(r_texture_atlas, r_batch):
            r_texture_atlas = split_tensor_into_six(r_texture_atlas)
            vertices_mapped = r_texture_atlas[r_batch["uv"][:, 0].long(), :, (r_batch["uv"][:, 1] * (config.image_size // 2)).long(), (r_batch["uv"][:, 2] * (config.image_size // 3)).long()]
            vertices_mapped = torch.cat([vertices_mapped, torch.ones((vertices_mapped.shape[0], 1), device=vertices_mapped.device)], dim=-1)
            return vertices_mapped

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [x for iidx, x in enumerate(eval_dataset.items) if iidx in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, shuffle=False, drop_last=True, num_workers=0)
        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.image_size, 3).to(device)
        R = DifferentiableRenderer(config.render_size, "bounds", config.colorspace)
        E = NormalEncoder(3).to(device)
        state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
        G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
        G.eval()
        E.eval()
        ema = torch.load(CHECKPOINT_EMA, map_location=device)
        ema.copy_to([p for p in G.parameters() if p.requires_grad])

        for iter_idx, batch in enumerate(tqdm(eval_loader)):
            eval_batch = to_device(batch, device)
            code = E(eval_batch['uv_normals'])

            for z_idx in range(num_latent):
                z = torch.randn(config.batch_size, config.latent_dim).to(device)
                atlas = G(z, code, noise_mode='const')
                fake_render = map_to_vertex(atlas, eval_batch).cpu()
                fake_render = ((torch.clamp(fake_render, -1, 1) * 0.5 + 0.5) * 255).int()
                mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=fake_render.numpy(), process=False)
                out_mesh.export(comparisons_dir / f"{selected_shapes[iter_idx]:05d}" / f"uv_{z_idx:02d}.obj")

    # Ours
    @hydra.main(config_path='../config', config_name='stylegan2')
    def Ours(config):
        from model.graph_generator_u_deep import Generator
        from model.graph import TwinGraphEncoder
        from dataset.mesh_real_features import FaceGraphMeshDataset

        config.batch_size = 1
        config.views_per_sample = 1
        config.image_size = 512
        config.render_size = 512
        config.num_mapping_layers = 5
        config.g_channel_base = 32768
        config.g_channel_max = 768

        OUTPUT_DIR_OURS_CODES = OUTPUT_DIR / "ours_vis" / "codes"

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/_epoch=119.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/ema_000076439.pth"

        device = torch.device("cuda:0")
        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [x for iidx, x in enumerate(eval_dataset.items) if iidx in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

        E = TwinGraphEncoder(eval_dataset.num_feats, 1).to(device)
        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=E.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
        state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
        G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        ema = torch.load(CHECKPOINT_EMA, map_location=device)
        ema.copy_to([p for p in G.parameters() if p.requires_grad])
        E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))

        G.eval()
        E.eval()

        eval_mesh_codes = [[73, 1, 34], [12, 24, 42, 52, 73], [42, 41, 55, 60, 92, 8], [42, 23], [61]]

        with torch.no_grad():
            for iter_idx, batch in enumerate(tqdm(eval_loader)):
                eval_batch = to_device(batch, device)
                shape = E(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
                for z_idx in eval_mesh_codes[iter_idx]:
                    z = torch.load(OUTPUT_DIR_OURS_CODES / f'{selected_shapes[iter_idx]:04d}' / f'{z_idx:04d}.pt').to(device)
                    fake = G(eval_batch['graph_data'], z, shape, noise_mode='const', )
                    vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                    mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                    out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                    out_mesh.export(comparisons_dir / f"{selected_shapes[iter_idx]:05d}" / f"ours_{z_idx:02d}.obj")

    # Ours TexConv
    @hydra.main(config_path='../config', config_name='stylegan2')
    def OursTConv(config):
        from dataset.mesh_real_features import FaceGraphMeshDataset
        from model.graph import GraphEncoder, FaceConv, TextureConv
        from model.graph_generator_u_texconv import Generator
        config.batch_size = 1
        config.views_per_sample = 1
        config.image_size = 256
        config.render_size = 512
        config.num_mapping_layers = 5
        config.g_channel_base = 65536
        config.g_channel_max = 768
        config.conv_aggregation = "mean"
        config.enc_conv = "texture"
        num_latent = 1

        OUTPUT_DIR_OURS_CODES = OUTPUT_DIR / "ours_vis" / "codes"
        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22011007_StyleGAN23D_tcmean-etc_fg6bgg-lrd1g14-v2m5-256/checkpoints/_epoch=109.ckpt"

        selected_shapes = [9, 24, 48, 889, 1829]
        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [x for idx, x in enumerate(eval_dataset.items) if idx in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

        E = GraphEncoder(eval_dataset.num_feats, conv_layer=FaceConv if config.enc_conv == 'face' else TextureConv).to(device)
        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, aggregation_function=config.conv_aggregation, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
        state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
        G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
        G.eval()
        E.eval()


        eval_mesh_codes = [[27, 36, 42, 82, 64], [73], [34], [24], [34]]

        with torch.no_grad():
            for iter_idx, batch in enumerate(tqdm(eval_loader)):
                eval_batch = to_device(batch, device)
                shape = E(eval_batch['x'], eval_batch['graph_data'])
                for z_idx in eval_mesh_codes[iter_idx]:
                    z = torch.load(OUTPUT_DIR_OURS_CODES / f'{selected_shapes[iter_idx]:04d}' / f'{z_idx:04d}.pt').to(device)
                    fake = G(eval_batch['graph_data'], z, shape, noise_mode='const', )
                    vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                    mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                    out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                    out_mesh.export(comparisons_dir / f"{selected_shapes[iter_idx]:05d}" / f"ours_tconv_{z_idx:02d}.obj")

    with torch.no_grad():
        texture_fields()
        LTG()
        SPSG()
        EG3D()
        OursUV()
        Ours()


def create_meshes_for_representation_ablation():
    import trimesh
    from PIL import Image
    import numpy as np

    device = torch.device("cuda:0")

    selected_shapes = [159] # [0, 1, 2, 5, 7, 8, 9, 11, 13, 15, 24, 41, 48, 55, 61, 97, 99, 107, 119, 123, 159, 195, 211, 218, 279, 287, 314, 322, 356, 379, 405, 474, 628, 631, 696, 708, 724, 726, 762, 776, 781, 792, 794, 814, 842, 861, 868, 873, 887, 888, 889, 891, 914, 930, 975, 988, 1064, 1075, 1097, 1151, 1155, 1168, 1170, 1176, 1188, 1222, 1226, 1230, 1231, 1233, 1237, 1240, 1328, 1337, 1371, 1373, 1445, 1463, 1466, 1469, 1480, 1482, 1502, 1503, 1505, 1510, 1513, 1517, 1603, 1604, 1620, 1625, 1626, 1642, 1690, 1693, 1702, 1707, 1711, 1740, 1790, 1799, 1808, 1821, 1824, 1829, 1840, 1852, 1853, 1871, 1875, 1925, 1927, 1952, 1955, 1959, 1962, 1963, 1972, 2018, 2024, 2067, 2095, 2158, 2198, 2244, 2245, 2250, 2266, 2291, 2311, 2481, 2482, 2620, 2683, 2883, 2891, 2914, 2929, 3018, 3040, 3065, 3146, 3150, 3154, 3322, 3323, 3482, 3581, 3665, 3669, 3762, 3787, 3817, 3874, 3885, 3932, 3950, 3954, 4848, 4883]
    selected_codes = [18, 35, 61, 70, 78]  # list(range(100))  # [1, 5, 10, 15, 20, 34, 50, 60]

    dump_meshes = True

    comparisons_dir = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/photoshape/compare_representation")
    combined_dir = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/photoshape/compare_representation/combined")
    comparisons_dir.mkdir(exist_ok=True)
    combined_dir.mkdir(exist_ok=True)
    device = torch.device("cuda:0")

    OUTPUT_DIR_OURS_CODES = OUTPUT_DIR / "ours_vis" / "codes"

    R = DifferentiableRenderer(512, "bounds")

    if dump_meshes:
        for idx in selected_shapes:
            (comparisons_dir / "mesh").mkdir(exist_ok=True, parents=True)

    @hydra.main(config_path='../config', config_name='stylegan2')
    def Ours(config):
        from model.graph_generator_u_deep import Generator
        from model.graph import TwinGraphEncoder
        from dataset.mesh_real_features import FaceGraphMeshDataset

        config.batch_size = 1
        config.views_per_sample = 1 if dump_meshes else 8
        config.image_size = 256
        config.render_size = 256
        config.num_mapping_layers = 5
        config.g_channel_base = 32768
        config.g_channel_max = 768

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/_epoch=119.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/ema_000076439.pth"

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [eval_dataset.items[i] for i in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)
        E = TwinGraphEncoder(eval_dataset.num_feats, 1).to(device)
        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=E.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
        state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
        G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        ema = torch.load(CHECKPOINT_EMA, map_location=device)
        ema.copy_to([p for p in G.parameters() if p.requires_grad])
        E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))

        G.eval()
        E.eval()

        with torch.no_grad():
            for iter_idx, batch in enumerate(tqdm(eval_loader)):
                eval_batch = to_device(batch, device)
                shape = E(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
                for z_idx in selected_codes:
                    z = torch.load(OUTPUT_DIR_OURS_CODES / f'{selected_shapes[iter_idx]:04d}' / f'{z_idx:04d}.pt').to(device)
                    fake = G(eval_batch['graph_data'], z, shape, noise_mode='const', )

                    if dump_meshes:
                        vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                        mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                        out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                        out_mesh.export(comparisons_dir / "mesh" / f"ours_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.obj")
                    else:
                        fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
                        save_image(fake_render, comparisons_dir / f"ours_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.jpg", nrow=config.views_per_sample, value_range=(-1, 1), normalize=True)

    @hydra.main(config_path='../config', config_name='stylegan2')
    def Ours_None(config):
        from model.graph_generator_u_deep import Generator
        from model.graph import GraphEncoder
        from dataset.mesh_real_features import FaceGraphMeshDataset
        config.batch_size = 1
        config.views_per_sample = 1 if dump_meshes else 8
        config.image_size = 256
        config.render_size = 256
        config.num_mapping_layers = 5

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/18020231_StyleGAN23D_bgg-nofeat-lrd1g14-v2m5-256/checkpoints/_epoch=99.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/18020231_StyleGAN23D_bgg-nofeat-lrd1g14-v2m5-256/checkpoints/ema_000127399.pth"

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [eval_dataset.items[i] for i in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

        E = GraphEncoder(eval_dataset.num_feats).to(device)
        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=E.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
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
            for z_idx in selected_codes:
                z = torch.load(OUTPUT_DIR_OURS_CODES / f'{selected_shapes[iter_idx]:04d}' / f'{z_idx:04d}.pt').to(device)
                fake = G(eval_batch['graph_data'], z, shape, noise_mode='const')
                if dump_meshes:
                    vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                    mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                    out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                    out_mesh.export(comparisons_dir / "mesh" / f"none_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.obj")
                else:
                    fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
                    save_image(fake_render, comparisons_dir / f"none_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.jpg", nrow=config.views_per_sample, value_range=(-1, 1), normalize=True)

    @hydra.main(config_path='../config', config_name='stylegan2')
    def Ours_Position(config):
        from model.graph_generator_u import Generator
        from model.graph import GraphEncoder
        from dataset.mesh_real_features import FaceGraphMeshDataset
        config.batch_size = 1
        config.views_per_sample = 1 if dump_meshes else 8
        config.image_size = 256
        config.render_size = 256
        config.num_mapping_layers = 5
        config.g_channel_base = 32768
        config.g_channel_max = 512
        config.features = "position"

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22011039_StyleGAN23D_pos_fg3bgg-lrd1g14-v2m5-256/checkpoints/_epoch=89.ckpt"

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [eval_dataset.items[i] for i in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
        E = GraphEncoder(eval_dataset.num_feats).to(device)
        state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
        G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
        G.eval()
        E.eval()

        for iter_idx, batch in enumerate(tqdm(eval_loader)):
            eval_batch = to_device(batch, device)
            shape = E(eval_batch['x'], eval_batch['graph_data'])
            for z_idx in selected_codes:
                z = torch.load(OUTPUT_DIR_OURS_CODES / f'{selected_shapes[iter_idx]:04d}' / f'{z_idx:04d}.pt').to(device)
                fake = G(eval_batch['graph_data'], z, shape, noise_mode='const')
                if dump_meshes:
                    vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                    mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                    out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                    out_mesh.export(comparisons_dir / "mesh" / f"pos_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.obj")
                else:
                    fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
                    save_image(fake_render, comparisons_dir / f"pos_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.jpg", nrow=config.views_per_sample, value_range=(-1, 1), normalize=True)

    @hydra.main(config_path='../config', config_name='stylegan2')
    def Ours_Laplacian(config):
        from model.graph_generator_u import Generator
        from model.graph import GraphEncoder
        from dataset.mesh_real_features import FaceGraphMeshDataset
        config.batch_size = 1
        config.views_per_sample = 1 if dump_meshes else 8
        config.image_size = 256
        config.render_size = 256
        config.num_mapping_layers = 5
        config.g_channel_base = 32768
        config.g_channel_max = 512
        config.features = "laplacian"

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22011317_StyleGAN23D_lap_fg3bgg-lrd1g14-v2m5-256/checkpoints/_epoch=79.ckpt"

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [eval_dataset.items[i] for i in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
        E = GraphEncoder(eval_dataset.num_feats).to(device)
        state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
        G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
        G.eval()
        E.eval()

        for iter_idx, batch in enumerate(tqdm(eval_loader)):
            eval_batch = to_device(batch, device)
            shape = E(eval_batch['x'], eval_batch['graph_data'])
            for z_idx in selected_codes:
                z = torch.load(OUTPUT_DIR_OURS_CODES / f'{iter_idx:04d}' / f'{z_idx:04d}.pt').to(device)
                fake = G(eval_batch['graph_data'], z, shape, noise_mode='const')
                if dump_meshes:
                    vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                    mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                    out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                    out_mesh.export(comparisons_dir / "mesh" / f"lap_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.obj")
                else:
                    fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
                    save_image(fake_render, comparisons_dir / f"lap_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.jpg", nrow=config.views_per_sample, value_range=(-1, 1), normalize=True)

    @hydra.main(config_path='../config', config_name='stylegan2')
    def Ours_Curvature(config):
        from model.graph_generator_u import Generator
        from model.graph import GraphEncoder
        from dataset.mesh_real_features import FaceGraphMeshDataset
        config.batch_size = 1
        config.views_per_sample = 1 if dump_meshes else 8
        config.image_size = 256
        config.render_size = 256
        config.num_mapping_layers = 5
        config.g_channel_base = 32768
        config.g_channel_max = 512
        config.features = "curvature"

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22011243_StyleGAN23D_curv_fg3bgg-lrd1g14-v2m5-256/checkpoints/_epoch=129.ckpt"

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [eval_dataset.items[i] for i in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
        E = GraphEncoder(eval_dataset.num_feats).to(device)
        state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
        G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
        G.eval()
        E.eval()

        for iter_idx, batch in enumerate(tqdm(eval_loader)):
            eval_batch = to_device(batch, device)
            shape = E(eval_batch['x'], eval_batch['graph_data'])
            for z_idx in selected_codes:
                z = torch.load(OUTPUT_DIR_OURS_CODES / f'{selected_shapes[iter_idx]:04d}' / f'{z_idx:04d}.pt').to(device)
                fake = G(eval_batch['graph_data'], z, shape, noise_mode='const')
                if dump_meshes:
                    vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                    mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                    out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                    out_mesh.export(comparisons_dir / "mesh" / f"curv_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.obj")
                else:
                    fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
                    save_image(fake_render, comparisons_dir / f"curv_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.jpg", nrow=config.views_per_sample, value_range=(-1, 1), normalize=True)

    @hydra.main(config_path='../config', config_name='stylegan2')
    def Ours_FF2(config):
        from model.graph_generator_u import Generator
        from model.graph import GraphEncoder
        from dataset.mesh_real_features import FaceGraphMeshDataset
        config.batch_size = 1
        config.views_per_sample = 1 if dump_meshes else 8
        config.image_size = 256
        config.render_size = 256
        config.num_mapping_layers = 5
        config.g_channel_base = 32768
        config.g_channel_max = 512
        config.features = "ff2"

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22011107_StyleGAN23D_ff2_fg3bgg-lrd1g14-v2m5-256/checkpoints/_epoch=139.ckpt"

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [eval_dataset.items[i] for i in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
        E = GraphEncoder(eval_dataset.num_feats).to(device)
        state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
        G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
        G.eval()
        E.eval()

        for iter_idx, batch in enumerate(tqdm(eval_loader)):
            eval_batch = to_device(batch, device)
            shape = E(eval_batch['x'], eval_batch['graph_data'])
            for z_idx in selected_codes:
                z = torch.load(OUTPUT_DIR_OURS_CODES / f'{selected_shapes[iter_idx]:04d}' / f'{z_idx:04d}.pt').to(device)
                fake = G(eval_batch['graph_data'], z, shape, noise_mode='const')
                if dump_meshes:
                    vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                    mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                    out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                    out_mesh.export(comparisons_dir / "mesh" / f"ff2_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.obj")
                else:
                    fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
                    save_image(fake_render, comparisons_dir / f"ff2_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.jpg", nrow=config.views_per_sample, value_range=(-1, 1), normalize=True)

    @hydra.main(config_path='../config', config_name='stylegan2')
    def Ours_Normal(config):
        from model.graph_generator_u_deep import Generator
        from model.graph import GraphEncoder
        from dataset.mesh_real_features import FaceGraphMeshDataset

        config.batch_size = 1
        config.views_per_sample = 1 if dump_meshes else 8
        config.image_size = 256
        config.render_size = 256
        config.num_mapping_layers = 5
        config.g_channel_base = 32768
        config.g_channel_max = 768

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/16021631_StyleGAN23D_fg3bgg-notwin-big-lrd1g14-v2m5-1K512/checkpoints/_epoch=139.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/16021631_StyleGAN23D_fg3bgg-notwin-big-lrd1g14-v2m5-1K512/checkpoints/ema_000089179.pth"

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [eval_dataset.items[i] for i in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

        E = GraphEncoder(eval_dataset.num_feats).to(device)
        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=E.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
        state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
        G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        ema = torch.load(CHECKPOINT_EMA, map_location=device)
        ema.copy_to([p for p in G.parameters() if p.requires_grad])
        E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))

        G.eval()
        E.eval()

        with torch.no_grad():
            for iter_idx, batch in enumerate(tqdm(eval_loader)):
                eval_batch = to_device(batch, device)
                shape = E(eval_batch['x'], eval_batch['graph_data'])
                for z_idx in selected_codes:
                    z = torch.load(OUTPUT_DIR_OURS_CODES / f'{selected_shapes[iter_idx]:04d}' / f'{z_idx:04d}.pt').to(device)
                    fake = G(eval_batch['graph_data'], z, shape, noise_mode='const', )

                    if dump_meshes:
                        vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                        mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                        out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                        out_mesh.export(comparisons_dir / "mesh" / f"norm_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.obj")
                    else:
                        fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
                        save_image(fake_render, comparisons_dir / f"norm_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.jpg", nrow=config.views_per_sample, value_range=(-1, 1), normalize=True)

    with torch.no_grad():
        Ours()
        Ours_None()
        Ours_Position()
        Ours_Laplacian()
        Ours_Curvature()
        Ours_FF2()
        Ours_Normal()
        if not dump_meshes:
            for i in selected_shapes:
                for j in selected_codes:
                    images = []
                    for m in ["none", "pos", "lap", "curv", "ff2", "norm", "ours"]:
                        images.append(np.array(Image.open(comparisons_dir / f"{m}_{i:04d}_{j:04d}.jpg")))
                    Image.fromarray(np.concatenate(images, axis=0)).save(combined_dir / f"{i:04d}_{j:04d}.jpg")


def create_meshes_for_rendres_ablation():
    import trimesh
    from PIL import Image
    import numpy as np

    device = torch.device("cuda:0")

    selected_shapes = [123]  # [0, 1, 2, 5, 7, 8, 9, 11, 13, 15, 24, 41, 48, 55, 61, 97, 99, 107, 119, 123, 159, 195, 211, 218, 279, 287, 314, 322, 356, 379, 405, 474, 628, 631, 696, 708, 724, 726, 762, 776, 781, 792, 794, 814, 842, 861, 868, 873, 887, 888, 889, 891, 914, 930, 975, 988, 1064, 1075, 1097, 1151, 1155, 1168, 1170, 1176, 1188, 1222, 1226, 1230, 1231, 1233, 1237, 1240, 1328, 1337, 1371, 1373, 1445, 1463, 1466, 1469, 1480, 1482, 1502, 1503, 1505, 1510, 1513, 1517, 1603, 1604, 1620, 1625, 1626, 1642, 1690, 1693, 1702, 1707, 1711, 1740, 1790, 1799, 1808, 1821, 1824, 1829, 1840, 1852, 1853, 1871, 1875, 1925, 1927, 1952, 1955, 1959, 1962, 1963, 1972, 2018, 2024, 2067, 2095, 2158, 2198, 2244, 2245, 2250, 2266, 2291, 2311, 2481, 2482, 2620, 2683, 2883, 2891, 2914, 2929, 3018, 3040, 3065, 3146, 3150, 3154, 3322, 3323, 3482, 3581, 3665, 3669, 3762, 3787, 3817, 3874, 3885, 3932, 3950, 3954, 4848, 4883]
    selected_codes = [1, 8, 10]  #list(range(40)) # list(range(100))  # [1, 5, 10, 15, 20, 34, 50, 60]

    dump_meshes = True

    comparisons_dir = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/photoshape/compare_resolution")
    combined_dir = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/photoshape/compare_resolution/combined")
    comparisons_dir.mkdir(exist_ok=True)
    combined_dir.mkdir(exist_ok=True)
    device = torch.device("cuda:0")

    OUTPUT_DIR_OURS_CODES = OUTPUT_DIR / "ours_vis" / "codes"

    R = DifferentiableRenderer(512, "bounds")

    if dump_meshes:
        for idx in selected_shapes:
            (comparisons_dir / "mesh").mkdir(exist_ok=True, parents=True)

    @hydra.main(config_path='../config', config_name='stylegan2')
    def Ours(config):
        from model.graph_generator_u_deep import Generator
        from model.graph import TwinGraphEncoder
        from dataset.mesh_real_features import FaceGraphMeshDataset

        config.batch_size = 1
        config.views_per_sample = 1 if dump_meshes else 8
        config.image_size = 256
        config.render_size = 256
        config.num_mapping_layers = 5
        config.g_channel_base = 32768
        config.g_channel_max = 768

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/_epoch=119.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/ema_000076439.pth"

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [eval_dataset.items[i] for i in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)
        E = TwinGraphEncoder(eval_dataset.num_feats, 1).to(device)
        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=E.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
        state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
        G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        ema = torch.load(CHECKPOINT_EMA, map_location=device)
        ema.copy_to([p for p in G.parameters() if p.requires_grad])
        E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))

        G.eval()
        E.eval()

        with torch.no_grad():
            for iter_idx, batch in enumerate(tqdm(eval_loader)):
                eval_batch = to_device(batch, device)
                shape = E(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
                for z_idx in selected_codes:
                    z = torch.load(OUTPUT_DIR_OURS_CODES / f'{selected_shapes[iter_idx]:04d}' / f'{z_idx:04d}.pt').to(device)
                    fake = G(eval_batch['graph_data'], z, shape, noise_mode='const', )

                    if dump_meshes:
                        vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                        mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                        out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                        out_mesh.export(comparisons_dir / "mesh" / f"ours_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.obj")
                    else:
                        fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
                        save_image(fake_render, comparisons_dir / f"ours_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.jpg", nrow=config.views_per_sample, value_range=(-1, 1), normalize=True)

    @hydra.main(config_path='../config', config_name='stylegan2')
    def Ours64(config):
        from model.graph_generator_u_deep import Generator
        from model.graph import GraphEncoder
        from dataset.mesh_real_features import FaceGraphMeshDataset
        config.batch_size = 1
        config.views_per_sample = 1 if dump_meshes else 8
        config.image_size = 256
        config.render_size = 256
        config.num_mapping_layers = 5
        config.g_channel_base = 32768

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/19020403_StyleGAN23D_fg3bgg-notwin-lrd1g14-v2m5-64/checkpoints/_epoch=119.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/19020403_StyleGAN23D_fg3bgg-notwin-lrd1g14-v2m5-64/checkpoints/ema_000076439.pth"

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [eval_dataset.items[i] for i in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

        E = GraphEncoder(eval_dataset.num_feats).to(device)
        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=E.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
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
            for z_idx in selected_codes:
                z = torch.load(OUTPUT_DIR_OURS_CODES / f'{selected_shapes[iter_idx]:04d}' / f'{z_idx:04d}.pt').to(device)
                fake = G(eval_batch['graph_data'], z, shape, noise_mode='const')
                if dump_meshes:
                    vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                    mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                    out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                    out_mesh.export(comparisons_dir / "mesh" / f"res64_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.obj")
                else:
                    fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
                    save_image(fake_render, comparisons_dir / f"res64_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.jpg", nrow=config.views_per_sample, value_range=(-1, 1), normalize=True)

    @hydra.main(config_path='../config', config_name='stylegan2')
    def Ours128(config):
        from model.graph_generator_u_deep import Generator
        from model.graph import GraphEncoder
        from dataset.mesh_real_features import FaceGraphMeshDataset
        config.batch_size = 1
        config.views_per_sample = 1 if dump_meshes else 8
        config.image_size = 256
        config.render_size = 256
        config.num_mapping_layers = 5
        config.g_channel_base = 32768

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/18020722_StyleGAN23D_fg3bgg-notwin-lrd1g14-v2m5-128/checkpoints/_epoch=119.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/18020722_StyleGAN23D_fg3bgg-notwin-lrd1g14-v2m5-128/checkpoints/ema_000076439.pth"

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [eval_dataset.items[i] for i in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

        E = GraphEncoder(eval_dataset.num_feats).to(device)
        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=E.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
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
            for z_idx in selected_codes:
                z = torch.load(OUTPUT_DIR_OURS_CODES / f'{selected_shapes[iter_idx]:04d}' / f'{z_idx:04d}.pt').to(device)
                fake = G(eval_batch['graph_data'], z, shape, noise_mode='const')
                if dump_meshes:
                    vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                    mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                    out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                    out_mesh.export(comparisons_dir / "mesh" / f"res128_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.obj")
                else:
                    fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
                    save_image(fake_render, comparisons_dir / f"res128_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.jpg", nrow=config.views_per_sample, value_range=(-1, 1), normalize=True)

    with torch.no_grad():
        Ours()
        Ours64()
        Ours128()
        if not dump_meshes:
            for i in selected_shapes:
                for j in selected_codes:
                    images = []
                    for m in ["res64", "res128", "ours"]:
                        images.append(np.array(Image.open(comparisons_dir / f"{m}_{i:04d}_{j:04d}.jpg")))
                    Image.fromarray(np.concatenate(images, axis=0)).save(combined_dir / f"{i:04d}_{j:04d}.jpg")


def create_meshes_for_faceres_ablation():
    import trimesh
    from PIL import Image
    import numpy as np

    device = torch.device("cuda:0")

    selected_shapes = [631, 724]  # [0, 1, 2, 5, 7, 8, 9, 11, 13, 15, 24, 41, 48, 55, 61, 97, 99, 107, 119, 123, 159, 195, 211, 218, 279, 287, 314, 322, 356, 379, 405, 474, 628, 631, 696, 708, 724, 726, 762, 776, 781, 792, 794, 814, 842, 861, 868, 873, 887, 888, 889, 891, 914, 930, 975, 988, 1064, 1075, 1097, 1151, 1155, 1168, 1170, 1176, 1188, 1222, 1226, 1230, 1231, 1233, 1237, 1240, 1328, 1337, 1371, 1373, 1445, 1463, 1466, 1469, 1480, 1482, 1502, 1503, 1505, 1510, 1513, 1517, 1603, 1604, 1620, 1625, 1626, 1642, 1690, 1693, 1702, 1707, 1711, 1740, 1790, 1799, 1808, 1821, 1824, 1829, 1840, 1852, 1853, 1871, 1875, 1925, 1927, 1952, 1955, 1959, 1962, 1963, 1972, 2018, 2024, 2067, 2095, 2158, 2198, 2244, 2245, 2250, 2266, 2291, 2311, 2481, 2482, 2620, 2683, 2883, 2891, 2914, 2929, 3018, 3040, 3065, 3146, 3150, 3154, 3322, 3323, 3482, 3581, 3665, 3669, 3762, 3787, 3817, 3874, 3885, 3932, 3950, 3954, 4848, 4883]
    selected_codes = [5, 24]  # list(range(40)) # list(range(100))  # [1, 5, 10, 15, 20, 34, 50, 60]

    dump_meshes = True

    comparisons_dir = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/photoshape/compare_facecount")
    combined_dir = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/photoshape/compare_facecount/combined")
    comparisons_dir.mkdir(exist_ok=True)
    combined_dir.mkdir(exist_ok=True)
    device = torch.device("cuda:0")

    OUTPUT_DIR_OURS_CODES = OUTPUT_DIR / "ours_vis" / "codes"

    R = DifferentiableRenderer(512, "bounds")

    if dump_meshes:
        for idx in selected_shapes:
            (comparisons_dir / "mesh").mkdir(exist_ok=True, parents=True)

    @hydra.main(config_path='../config', config_name='stylegan2')
    def Ours(config):
        from model.graph_generator_u_deep import Generator
        from model.graph import TwinGraphEncoder
        from dataset.mesh_real_features import FaceGraphMeshDataset

        config.batch_size = 1
        config.views_per_sample = 1 if dump_meshes else 8
        config.image_size = 256
        config.render_size = 256
        config.num_mapping_layers = 5
        config.g_channel_base = 32768
        config.g_channel_max = 768

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/_epoch=119.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/ema_000076439.pth"

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [eval_dataset.items[i] for i in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)
        E = TwinGraphEncoder(eval_dataset.num_feats, 1).to(device)
        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=E.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
        state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
        G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        ema = torch.load(CHECKPOINT_EMA, map_location=device)
        ema.copy_to([p for p in G.parameters() if p.requires_grad])
        E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))

        G.eval()
        E.eval()

        with torch.no_grad():
            for iter_idx, batch in enumerate(tqdm(eval_loader)):
                eval_batch = to_device(batch, device)
                shape = E(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
                for z_idx in selected_codes:
                    z = torch.load(OUTPUT_DIR_OURS_CODES / f'{selected_shapes[iter_idx]:04d}' / f'{z_idx:04d}.pt').to(device)
                    fake = G(eval_batch['graph_data'], z, shape, noise_mode='const', )

                    if dump_meshes:
                        vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                        mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                        out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                        out_mesh.export(comparisons_dir / "mesh" / f"ours_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.obj")
                    else:
                        fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
                        save_image(fake_render, comparisons_dir / f"ours_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.jpg", nrow=config.views_per_sample, value_range=(-1, 1), normalize=True)

    @hydra.main(config_path='../config', config_name='stylegan2')
    def Ours_face6k(config):
        from model.graph_generator_u_6k import Generator
        from model.graph import GraphEncoder
        from dataset.mesh_real_features import FaceGraphMeshDataset
        config.batch_size = 1
        config.views_per_sample = 1 if dump_meshes else 8
        config.image_size = 256
        config.render_size = 256
        config.num_mapping_layers = 5
        config.g_channel_base = 32768

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22021833_StyleGAN23D_lowres6K-notwin-lrd1g14-v2m5-256/checkpoints/_epoch=44.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/22021833_StyleGAN23D_lowres6K-notwin-lrd1g14-v2m5-256/checkpoints/ema_000057329.pth"

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [eval_dataset.items[i] for i in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
        E = GraphEncoder(eval_dataset.num_feats).to(device)
        state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
        G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        ema = torch.load(CHECKPOINT_EMA, map_location=device)
        ema.copy_to([p for p in G.parameters() if p.requires_grad])
        E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
        G.eval()
        E.eval()

        for iter_idx, batch in enumerate(tqdm(eval_loader)):
            eval_batch = to_device(batch, device)
            shape = E(batch['x'], batch['graph_data'])
            for z_idx in selected_codes:
                z = torch.load(OUTPUT_DIR_OURS_CODES / f'{selected_shapes[iter_idx]:04d}' / f'{z_idx:04d}.pt').to(device)
                fake = G(eval_batch['graph_data'], z, shape, noise_mode='const')
                if dump_meshes:
                    vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                    mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                    out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                    out_mesh.export(comparisons_dir / "mesh" / f"face6k_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.obj")
                else:
                    fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
                    save_image(fake_render, comparisons_dir / f"face6k_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.jpg", nrow=config.views_per_sample, value_range=(-1, 1), normalize=True)

    with torch.no_grad():
        Ours()
        Ours_face6k()
        if not dump_meshes:
            for i in selected_shapes:
                for j in selected_codes:
                    images = []
                    for m in ["face6k", "ours"]:
                        images.append(np.array(Image.open(comparisons_dir / f"{m}_{i:04d}_{j:04d}.jpg")))
                    Image.fromarray(np.concatenate(images, axis=0)).save(combined_dir / f"{i:04d}_{j:04d}.jpg")


def create_meshes_for_view_ablation():
    import trimesh
    from PIL import Image
    import numpy as np

    device = torch.device("cuda:0")

    selected_shapes = [724, 794, 861, 975] # [0, 1, 2, 5, 7, 8, 9, 11, 13, 15, 24, 41, 48, 55, 61, 97, 99, 107, 119, 123, 159, 195, 211, 218, 279, 287, 314, 322, 356, 379, 405, 474, 628, 631, 696, 708, 724, 726, 762, 776, 781, 792, 794, 814, 842, 861, 868, 873, 887, 888, 889, 891, 914, 930, 975, 988, 1064, 1075, 1097, 1151, 1155, 1168, 1170, 1176, 1188, 1222, 1226, 1230, 1231, 1233, 1237, 1240, 1328, 1337, 1371, 1373, 1445, 1463, 1466, 1469, 1480, 1482, 1502, 1503, 1505, 1510, 1513, 1517, 1603, 1604, 1620, 1625, 1626, 1642, 1690, 1693, 1702, 1707, 1711, 1740, 1790, 1799, 1808, 1821, 1824, 1829, 1840, 1852, 1853, 1871, 1875, 1925, 1927, 1952, 1955, 1959, 1962, 1963, 1972, 2018, 2024, 2067, 2095, 2158, 2198, 2244, 2245, 2250, 2266, 2291, 2311, 2481, 2482, 2620, 2683, 2883, 2891, 2914, 2929, 3018, 3040, 3065, 3146, 3150, 3154, 3322, 3323, 3482, 3581, 3665, 3669, 3762, 3787, 3817, 3874, 3885, 3932, 3950, 3954, 4848, 4883]
    selected_codes = [21, 34, 42, 51, 2, 7, 22, 3, 25] # list(range(40))  # list(range(100))  # [1, 5, 10, 15, 20, 34, 50, 60]

    dump_meshes = True

    comparisons_dir = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/photoshape/compare_view")
    combined_dir = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/photoshape/compare_view/combined")
    comparisons_dir.mkdir(exist_ok=True)
    combined_dir.mkdir(exist_ok=True)
    device = torch.device("cuda:0")

    OUTPUT_DIR_OURS_CODES = OUTPUT_DIR / "ours_vis" / "codes"

    R = DifferentiableRenderer(512, "bounds")

    if dump_meshes:
        for idx in selected_shapes:
            (comparisons_dir / "mesh").mkdir(exist_ok=True, parents=True)

    @hydra.main(config_path='../config', config_name='stylegan2')
    def Ours(config):
        from model.graph_generator_u_deep import Generator
        from model.graph import TwinGraphEncoder
        from dataset.mesh_real_features import FaceGraphMeshDataset

        config.batch_size = 1
        config.views_per_sample = 1 if dump_meshes else 8
        config.image_size = 256
        config.render_size = 256
        config.num_mapping_layers = 5
        config.g_channel_base = 32768
        config.g_channel_max = 768

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/_epoch=119.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/ema_000076439.pth"

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [eval_dataset.items[i] for i in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)
        E = TwinGraphEncoder(eval_dataset.num_feats, 1).to(device)
        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=E.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
        state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
        G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        ema = torch.load(CHECKPOINT_EMA, map_location=device)
        ema.copy_to([p for p in G.parameters() if p.requires_grad])
        E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))

        G.eval()
        E.eval()

        with torch.no_grad():
            for iter_idx, batch in enumerate(tqdm(eval_loader)):
                eval_batch = to_device(batch, device)
                shape = E(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
                for z_idx in selected_codes:
                    z = torch.load(OUTPUT_DIR_OURS_CODES / f'{selected_shapes[iter_idx]:04d}' / f'{z_idx:04d}.pt').to(device)
                    fake = G(eval_batch['graph_data'], z, shape, noise_mode='const', )

                    if dump_meshes:
                        vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                        mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                        out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                        out_mesh.export(comparisons_dir / "mesh" / f"ours_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.obj")
                    else:
                        fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
                        save_image(fake_render, comparisons_dir / f"ours_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.jpg", nrow=config.views_per_sample, value_range=(-1, 1), normalize=True)

    @hydra.main(config_path='../config', config_name='stylegan2')
    def Ours_view1(config):
        from model.graph_generator_u_deep import Generator
        from model.graph import GraphEncoder
        from dataset.mesh_real_features import FaceGraphMeshDataset
        config.batch_size = 1
        config.views_per_sample = 1 if dump_meshes else 8
        config.image_size = 256
        config.render_size = 256
        config.num_mapping_layers = 8
        config.g_channel_base = 32768

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/18020916_StyleGAN23D_fg3bgg-notwin-big-lrd1g14-v1m5-512/checkpoints/_epoch=119.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/18020916_StyleGAN23D_fg3bgg-notwin-big-lrd1g14-v1m5-512/checkpoints/ema_000076439.pth"

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [eval_dataset.items[i] for i in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

        E = GraphEncoder(eval_dataset.num_feats).to(device)
        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=E.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
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
            for z_idx in selected_codes:
                z = torch.load(OUTPUT_DIR_OURS_CODES / f'{selected_shapes[iter_idx]:04d}' / f'{z_idx:04d}.pt').to(device)
                fake = G(eval_batch['graph_data'], z, shape, noise_mode='const')
                if dump_meshes:
                    vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                    mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                    out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                    out_mesh.export(comparisons_dir / "mesh" / f"view1_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.obj")
                else:
                    fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
                    save_image(fake_render, comparisons_dir / f"view1_{selected_shapes[iter_idx]:04d}_{z_idx:04d}.jpg", nrow=config.views_per_sample, value_range=(-1, 1), normalize=True)

    with torch.no_grad():
        Ours()
        Ours_view1()
        if not dump_meshes:
            for i in selected_shapes:
                for j in selected_codes:
                    images = []
                    for m in ["view1", "ours"]:
                        images.append(np.array(Image.open(comparisons_dir / f"{m}_{i:04d}_{j:04d}.jpg")))
                    Image.fromarray(np.concatenate(images, axis=0)).save(combined_dir / f"{i:04d}_{j:04d}.jpg")


def create_latent_samples():
    import trimesh
    from PIL import Image
    import numpy as np

    device = torch.device("cuda:0")

    selected_shapes = [975]# [0] #[631]  # [0, 1, 2, 5, 7, 8, 9, 11, 13, 15, 24, 41, 48, 55, 61, 97, 99, 107, 119, 123, 159, 195, 211, 218, 279, 287, 314, 322, 356, 379, 405, 474, 628, 631, 696, 708, 724, 726, 762, 776, 781, 792, 794, 814, 842, 861, 868, 873, 887, 888, 889, 891, 914, 930, 975, 988, 1064, 1075, 1097, 1151, 1155, 1168, 1170, 1176, 1188, 1222, 1226, 1230, 1231, 1233, 1237, 1240, 1328, 1337, 1371, 1373, 1445, 1463, 1466, 1469, 1480, 1482, 1502, 1503, 1505, 1510, 1513, 1517, 1603, 1604, 1620, 1625, 1626, 1642, 1690, 1693, 1702, 1707, 1711, 1740, 1790, 1799, 1808, 1821, 1824, 1829, 1840, 1852, 1853, 1871, 1875, 1925, 1927, 1952, 1955, 1959, 1962, 1963, 1972, 2018, 2024, 2067, 2095, 2158, 2198, 2244, 2245, 2250, 2266, 2291, 2311, 2481, 2482, 2620, 2683, 2883, 2891, 2914, 2929, 3018, 3040, 3065, 3146, 3150, 3154, 3322, 3323, 3482, 3581, 3665, 3669, 3762, 3787, 3817, 3874, 3885, 3932, 3950, 3954, 4848, 4883]
    selected_codes = [3, 67] #[34, 61]#[5, 12, 16, 22, 24, 29, 34, 41, 42, 49, 56, 60, 61, 73]#[7, 34]  # [1, 2, 4, 5, 7, 22, 24, 31, 34, 41, 86]  # list(range(40)) # list(range(100))  # [1, 5, 10, 15, 20, 34, 50, 60]

    dump_meshes = True

    comparisons_dir = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/photoshape/compare_latent")
    combined_dir = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/photoshape/compare_latent/combined")
    comparisons_dir.mkdir(exist_ok=True)
    combined_dir.mkdir(exist_ok=True)
    device = torch.device("cuda:0")

    OUTPUT_DIR_OURS_CODES = OUTPUT_DIR / "ours_vis" / "codes"

    R = DifferentiableRenderer(512, "bounds")

    if dump_meshes:
        for idx in selected_shapes:
            (comparisons_dir / "mesh").mkdir(exist_ok=True, parents=True)

    @hydra.main(config_path='../config', config_name='stylegan2')
    def Ours(config):
        from model.graph_generator_u_deep import Generator
        from model.graph import TwinGraphEncoder
        from dataset.mesh_real_features import FaceGraphMeshDataset

        config.batch_size = 1
        config.views_per_sample = 1 if dump_meshes else 8
        config.image_size = 256
        config.render_size = 256
        config.num_mapping_layers = 5
        config.g_channel_base = 32768
        config.g_channel_max = 768

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/_epoch=119.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/ema_000076439.pth"

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [eval_dataset.items[i] for i in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)
        E = TwinGraphEncoder(eval_dataset.num_feats, 1).to(device)
        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=E.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
        state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
        G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        ema = torch.load(CHECKPOINT_EMA, map_location=device)
        ema.copy_to([p for p in G.parameters() if p.requires_grad])
        E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))

        G.eval()
        E.eval()

        with torch.no_grad():
            for iter_idx, batch in enumerate(tqdm(eval_loader)):
                eval_batch = to_device(batch, device)
                shape = E(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])

                for z_idx_0 in selected_codes:
                    for z_idx_1 in selected_codes:
                        if z_idx_0 != z_idx_1:

                            z_0 = torch.load(OUTPUT_DIR_OURS_CODES / f'{selected_shapes[iter_idx]:04d}' / f'{z_idx_0:04d}.pt').to(device)
                            z_1 = torch.load(OUTPUT_DIR_OURS_CODES / f'{selected_shapes[iter_idx]:04d}' / f'{z_idx_1:04d}.pt').to(device)
                            for count in range(14):
                                z = z_0 + (z_1 - z_0) * count / 9
                                fake = G(eval_batch['graph_data'], z, shape, noise_mode='const', )
                                if dump_meshes:
                                    vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                                    mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                                    out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                                    out_mesh.export(comparisons_dir / "mesh" / f"ours_{selected_shapes[iter_idx]:04d}_{z_idx_0:04d}_{z_idx_1:04d}_{count:02d}.obj")
                                else:
                                    fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
                                    save_image(fake_render, comparisons_dir / f"ours_{selected_shapes[iter_idx]:04d}_{z_idx_0:04d}_{z_idx_1:04d}_{count:02d}.jpg", nrow=config.views_per_sample, value_range=(-1, 1), normalize=True)

    with torch.no_grad():
        Ours()
        if not dump_meshes:
            for i in selected_shapes:
                for z_0 in selected_codes:
                    for z_1 in selected_codes:
                        if z_0 != z_1:
                            images = []
                            for count in range(14):
                                images.append(np.array(Image.open(comparisons_dir / f"ours_{i:04d}_{z_0:04d}_{z_1:04d}_{count:02d}.jpg")))
                            Image.fromarray(np.concatenate(images, axis=0)).save(combined_dir / f"{i:04d}_{z_0:04d}_{z_1:04d}.jpg")


def create_latent_samples_detailed():
    import trimesh
    from PIL import Image
    import numpy as np

    device = torch.device("cuda:0")

    selected_shapes = [0]# [0] #[631]  # [0, 1, 2, 5, 7, 8, 9, 11, 13, 15, 24, 41, 48, 55, 61, 97, 99, 107, 119, 123, 159, 195, 211, 218, 279, 287, 314, 322, 356, 379, 405, 474, 628, 631, 696, 708, 724, 726, 762, 776, 781, 792, 794, 814, 842, 861, 868, 873, 887, 888, 889, 891, 914, 930, 975, 988, 1064, 1075, 1097, 1151, 1155, 1168, 1170, 1176, 1188, 1222, 1226, 1230, 1231, 1233, 1237, 1240, 1328, 1337, 1371, 1373, 1445, 1463, 1466, 1469, 1480, 1482, 1502, 1503, 1505, 1510, 1513, 1517, 1603, 1604, 1620, 1625, 1626, 1642, 1690, 1693, 1702, 1707, 1711, 1740, 1790, 1799, 1808, 1821, 1824, 1829, 1840, 1852, 1853, 1871, 1875, 1925, 1927, 1952, 1955, 1959, 1962, 1963, 1972, 2018, 2024, 2067, 2095, 2158, 2198, 2244, 2245, 2250, 2266, 2291, 2311, 2481, 2482, 2620, 2683, 2883, 2891, 2914, 2929, 3018, 3040, 3065, 3146, 3150, 3154, 3322, 3323, 3482, 3581, 3665, 3669, 3762, 3787, 3817, 3874, 3885, 3932, 3950, 3954, 4848, 4883]
    selected_codes = [34, 61] #[34, 61]#[5, 12, 16, 22, 24, 29, 34, 41, 42, 49, 56, 60, 61, 73]#[7, 34]  # [1, 2, 4, 5, 7, 22, 24, 31, 34, 41, 86]  # list(range(40)) # list(range(100))  # [1, 5, 10, 15, 20, 34, 50, 60]

    dump_meshes = True

    comparisons_dir = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/photoshape/compare_latent_detailed")
    comparisons_dir.mkdir(exist_ok=True)
    device = torch.device("cuda:0")

    OUTPUT_DIR_OURS_CODES = OUTPUT_DIR / "ours_vis" / "codes"

    R = DifferentiableRenderer(512, "bounds")

    if dump_meshes:
        for idx in selected_shapes:
            (comparisons_dir / "mesh").mkdir(exist_ok=True, parents=True)

    max_count = 5 * 60

    @hydra.main(config_path='../config', config_name='stylegan2')
    def Ours(config):
        from model.graph_generator_u_deep import Generator
        from model.graph import TwinGraphEncoder
        from dataset.mesh_real_features import FaceGraphMeshDataset

        config.batch_size = 1
        config.views_per_sample = 1 if dump_meshes else 8
        config.image_size = 256
        config.render_size = 256
        config.num_mapping_layers = 5
        config.g_channel_base = 32768
        config.g_channel_max = 768

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/_epoch=119.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/15021601_StyleGAN23D_fg3bgg-big-lrd1g14-v2m5-1K512/checkpoints/ema_000076439.pth"

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [eval_dataset.items[i] for i in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)
        E = TwinGraphEncoder(eval_dataset.num_feats, 1).to(device)
        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=E.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
        state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
        G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        ema = torch.load(CHECKPOINT_EMA, map_location=device)
        ema.copy_to([p for p in G.parameters() if p.requires_grad])
        E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))

        G.eval()
        E.eval()
        with torch.no_grad():
            for iter_idx, batch in enumerate(tqdm(eval_loader)):
                eval_batch = to_device(batch, device)
                shape = E(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])

                for itr_0, z_idx_0 in enumerate(selected_codes):
                    for itr_1, z_idx_1 in enumerate(selected_codes):
                        if itr_1 > itr_0:
                            z_0 = torch.load(OUTPUT_DIR_OURS_CODES / f'{selected_shapes[iter_idx]:04d}' / f'{z_idx_0:04d}.pt').to(device)
                            z_1 = torch.load(OUTPUT_DIR_OURS_CODES / f'{selected_shapes[iter_idx]:04d}' / f'{z_idx_1:04d}.pt').to(device)
                            for count in range(max_count):
                                z = z_0 + (z_1 - z_0) * count / (max_count - 1)
                                fake = G(eval_batch['graph_data'], z, shape, noise_mode='const', )
                                if dump_meshes:
                                    vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                                    mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                                    out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                                    out_mesh.export(comparisons_dir / "mesh" / f"ours_{selected_shapes[iter_idx]:04d}_{z_idx_0:04d}_{z_idx_1:04d}_{count:02d}.obj")
                                else:
                                    fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
                                    save_image(fake_render, comparisons_dir / f"ours_{selected_shapes[iter_idx]:04d}_{z_idx_0:04d}_{z_idx_1:04d}_{count:02d}.jpg", nrow=config.views_per_sample, value_range=(-1, 1), normalize=True)

    with torch.no_grad():
        Ours()


@hydra.main(config_path='../config', config_name='stylegan2')
def compute_singularity_stats(config):
    from dataset.mesh_real_features import FaceGraphMeshDataset
    from torch_scatter import scatter_add
    import json
    import trimesh

    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=1, num_workers=0)
    hierarchy_path = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs-manifold-highres/")
    levels = [0] * 5
    vctr = [0] * 5
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        selections = json.loads((hierarchy_path / batch['name'][0] / "selection.json").read_text())
        for lvl, k in enumerate(["96", "384", "1536", "6144", "24576"]):
            mesh = trimesh.load(hierarchy_path / batch['name'][0] / f'quad_{int(k):05d}_{selections[k]:03d}.obj', process=False)
            vertices, faces = torch.from_numpy(mesh.vertices).float(), torch.from_numpy(mesh.faces).long()
            vertex_counts = torch.zeros(vertices.shape[0])
            scatter_add(torch.ones(faces.reshape(-1).shape[0]), index=faces.reshape(-1), out=vertex_counts)
            levels[lvl] += ((vertex_counts == 3).sum() + (vertex_counts == 5).sum()).item()
            vctr[lvl] += vertices.shape[0]
    levels = [levels[i]/vctr[i] for i in range(5)]
    print(levels)


if __name__ == "__main__":
    # compute_singularity_stats()
    # render_mesh()
    # create_meshes_for_rendres_ablation()
    # create_meshes_for_faceres_ablation()
    # create_meshes_for_view_ablation()
    # create_latent_samples()
    # create_latent_samples_detailed()
    # create_meshes_for_representation_ablation()
    create_meshes_for_baselines()
    # render_mesh()
    # ablate_our_gan_6k()
    # evaluate_uv_ours_parameterized_gan()
    # create_real()
    # evaluate_uv_parameterized_gan()
    # evaluate_3d_parameterized_gan()
    # evaluate_our_gan()
    # evaluate_our_gan_texconv()
    # evaluate_triplane_gan()
    # evaluate_texturefields_gan()
    # render_latent()
    # evaluate_uv_normalign_parameterized_gan()
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
