from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from pathlib import Path

from dataset import GraphDataLoader, to_device, to_vertex_colors_scatter
from model.differentiable_renderer import DifferentiableRenderer
import hydra

from util.misc import get_parameters_from_state_dict, compute_fid

OUTPUT_DIR = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/compcars")
REAL_DIR = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/compcars/real")


def render_faces(R, face_colors, batch, render_size, image_size):
    rendered_color = R.render(batch['vertices'], batch['indices'], to_vertex_colors_scatter(face_colors, batch), batch["ranges"].cpu(), resolution=render_size)
    ret_val = rendered_color.permute((0, 3, 1, 2))
    if render_size != image_size:
        ret_val = torch.nn.functional.interpolate(ret_val, (image_size, image_size), mode='bilinear', align_corners=True)
    return ret_val


@hydra.main(config_path='../config', config_name='stylegan2_car')
def create_real(config):
    from PIL import Image
    import numpy as np
    image_size = 256
    REAL_DIR.mkdir(exist_ok=True, parents=True)
    for idx, p in enumerate(tqdm(list(Path(config.image_path).iterdir()))):
        img = Image.open(p)
        width, height = img.size
        result = Image.new(img.mode, (int(width * 1.2), int(height * 1.2)), (255, 255, 255))
        result.paste(img, (int(width * 0.1), int(height * 0.1)))
        result = result.resize((image_size, image_size), resample=Image.LANCZOS)
        mask = np.array(Image.open(Path(config.mask_path) / p.name))
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = Image.fromarray(mask)
        width, height = mask.size
        mask_result = Image.new(mask.mode, (int(width * 1.2), int(height * 1.2)), 0)
        mask_result.paste(mask, (int(width * 0.1), int(height * 0.1)))
        mask_result = mask_result.resize((image_size, image_size), resample=Image.NEAREST)
        mask_arr = (np.array(mask_result) > 128).astype(np.uint8)
        result_arr = (np.array(result) * mask_arr[:, :, None] + np.ones_like(np.array(result)) * 255 * (1 - mask_arr[:, :, None])).astype(np.uint8)

        result = Image.fromarray(result_arr)
        result.save(REAL_DIR / f"{idx:06d}.jpg")


# UV Parameterization - Norm Aligned
@hydra.main(config_path='../config', config_name='stylegan2_car')
def evaluate_uv_normalign_parameterized_gan(config):
    config.views_per_sample = 8
    config.image_size = 128
    config.render_size = 512
    num_latent = 5

    from model.stylegan2.generator import Generator
    from dataset.meshcar_real_features_uv import FaceGraphMeshDataset
    OUTPUT_DIR_UV = OUTPUT_DIR / f"uv_na_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_UV.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/23022321_StyleGAN23D-CompCars_uv_firstview_silhoutte_256/checkpoints/_epoch=199.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/23022321_StyleGAN23D-CompCars_uv_firstview_silhoutte_256/checkpoints/ema_000062799.pth"

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
@hydra.main(config_path='../config', config_name='stylegan2_car')
def evaluate_uv_ours_parameterized_gan(config):
    config.views_per_sample = 8
    config.image_size = 256
    config.render_size = 512
    num_latent = 5

    from model.uv.generator import Generator, NormalEncoder
    from dataset.meshcar_real_features_uv_ours import FaceGraphMeshDataset, split_tensor_into_six
    OUTPUT_DIR_UV = OUTPUT_DIR / f"uv_ours_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_UV.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/23022329_StyleGAN23D-CompCars_uv_firstview_ours_silhoutte_256/checkpoints/_epoch=139.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/23022329_StyleGAN23D-CompCars_uv_firstview_ours_silhoutte_256/checkpoints/ema_000043959.pth"

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


# Triplane Implicit Parameterization
@hydra.main(config_path='../config', config_name='stylegan2_car')
def evaluate_triplane_gan(config):
    from model.eg3d.generator import Generator
    from model.styleganvox import SDFEncoder
    from dataset.meshcar_real_eg3d import FaceGraphMeshDataset
    config.batch_size = 1
    config.views_per_sample = 8
    config.image_size = 256
    config.render_size = 512
    config.num_mapping_layers = 8
    triplane_resolution = 256
    num_latent = 5

    OUTPUT_DIR_EG3D = OUTPUT_DIR / f"eg3d_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_EG3D.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/25021541_StyleGAN23D-CompCars_eg3d_geo_256_condition/checkpoints/_epoch=319.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/25021541_StyleGAN23D-CompCars_eg3d_geo_256_condition/checkpoints/ema_000065939.pth"

    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, triplane_resolution, 96, c_dim=256).to(device)
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


@hydra.main(config_path='../config', config_name='stylegan2_car')
def evaluate_our_gan(config):
    from model.graph_generator_u_deep import Generator
    from model.graph import TwinGraphEncoder
    from dataset.meshcar_real_features_ff2 import FaceGraphMeshDataset
    config.batch_size = 1
    config.views_per_sample = 8
    config.image_size = 256
    config.render_size = 512
    config.num_mapping_layers = 8
    config.g_channel_base = 32768
    config.g_channel_max = 768

    num_latent = 5

    OUTPUT_DIR_OURS = OUTPUT_DIR / f"ours_256_pres49K_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    # CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/24021538_StyleGAN23D-CompCars_bigdtwin-res-lrd1g14-v2m8-1K_128/checkpoints/_epoch=89.ckpt"
    # CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/24021538_StyleGAN23D-CompCars_bigdtwin-res-lrd1g14-v2m8-1K_128/checkpoints/ema_000014129.pth"
    # CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021321_StyleGAN23D-CompCars_moredata-clip_fg3bgg-lrd1g14-v4m5-1K_256/checkpoints/_epoch=349.ckpt"
    # CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021321_StyleGAN23D-CompCars_moredata-clip_fg3bgg-lrd1g14-v4m5-1K_256/checkpoints/ema_000054949.pth"
    # CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021429_StyleGAN23D-CompCars_progressive_clip_fg3bgg-lrd1g14-v4m5-1K_512/checkpoints/_epoch=449.ckpt"
    # CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021429_StyleGAN23D-CompCars_progressive_clip_fg3bgg-lrd1g14-v4m5-1K_512/checkpoints/ema_000070649.pth"

    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/24021538_StyleGAN23D-CompCars_bigdtwin-res-lrd1g14-v2m8-1K_128/checkpoints/_epoch=314.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/24021538_StyleGAN23D-CompCars_bigdtwin-res-lrd1g14-v2m8-1K_128/checkpoints/ema_000049454.pth"

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

    compute_fid(OUTPUT_DIR_OURS, REAL_DIR, OUTPUT_DIR_OURS.parent / f"score_256_pres49K_{config.views_per_sample}_{num_latent}.txt", device)


@hydra.main(config_path='../config', config_name='stylegan2_car')
def create_samples_our_gan(config):
    from model.graph_generator_u_deep import Generator
    from model.graph import TwinGraphEncoder
    from dataset.meshcar_real_features_ff2 import FaceGraphMeshDataset
    config.batch_size = 1
    config.views_per_sample = 8
    config.image_size = 512
    config.render_size = 512
    config.num_mapping_layers = 8
    config.g_channel_base = 32768
    config.g_channel_max = 768

    num_latent = 4

    OUTPUT_DIR_OURS = OUTPUT_DIR / f"ours_vis_check_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    # CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/24021538_StyleGAN23D-CompCars_bigdtwin-res-lrd1g14-v2m8-1K_128/checkpoints/_epoch=89.ckpt"
    # CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/24021538_StyleGAN23D-CompCars_bigdtwin-res-lrd1g14-v2m8-1K_128/checkpoints/ema_000014129.pth"
    # CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021321_StyleGAN23D-CompCars_moredata-clip_fg3bgg-lrd1g14-v4m5-1K_256/checkpoints/_epoch=349.ckpt"
    # CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021321_StyleGAN23D-CompCars_moredata-clip_fg3bgg-lrd1g14-v4m5-1K_256/checkpoints/ema_000054949.pth"

    # progressive_64
    # CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021429_StyleGAN23D-CompCars_progressive_clip_fg3bgg-lrd1g14-v4m5-1K_512/checkpoints/_epoch=449.ckpt"
    # CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021429_StyleGAN23D-CompCars_progressive_clip_fg3bgg-lrd1g14-v4m5-1K_512/checkpoints/ema_000070649.pth"

    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/24021538_StyleGAN23D-CompCars_bigdtwin-res-lrd1g14-v2m8-1K_128/checkpoints/_epoch=314.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/24021538_StyleGAN23D-CompCars_bigdtwin-res-lrd1g14-v2m8-1K_128/checkpoints/ema_000049454.pth"

    subset_ids = [int(x.stem) for x in Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/compcars/ours_progressive_64").iterdir()]

    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config, fixed_view_mode=True)
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
        if iter_idx not in subset_ids:
            continue
        eval_batch = to_device(batch, device)
        shape = E(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
        images = []
        for z_idx in range(num_latent):
            z = torch.randn(config.batch_size, config.latent_dim).to(device)
            fake = G(eval_batch['graph_data'], z, shape, noise_mode='const')
            fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
            images.append(fake_render)
        save_image(torch.cat(images, dim=0), OUTPUT_DIR_OURS / f"{iter_idx:05d}.jpg", nrow=8, value_range=(-1, 1), normalize=True)


# Implicit Parameterization
@hydra.main(config_path='../config', config_name='stylegan2_car')
def evaluate_texturefields_gan(config):
    from dataset.meshcar_real_eg3d import FaceGraphMeshDataset
    import trimesh
    import numpy as np

    OUTPUT_DIR_TEXFIELD = OUTPUT_DIR / f"texfield"
    OUTPUT_DIR_TEXFIELD.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda:0")
    config.batch_size = 1
    config.views_per_sample = 8
    config.image_size = 256
    config.render_size = 512
    R = DifferentiableRenderer(config.render_size, "bounds", config.colorspace)

    car_data_meshes = Path("/rhome/ysiddiqui/texture_fields_car/out/GAN/car/eval_fix/fake")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=0)
    for iter_idx, batch in enumerate(tqdm(eval_loader)):
        try:
            eval_batch = to_device(batch, device)
            fake_mesh = trimesh.load(car_data_meshes / f"{batch['name'][0]}.obj", process=False)
            fake_colors = fake_mesh.visual.vertex_colors[fake_mesh.faces[:, 0], :].astype(np.float32) + fake_mesh.visual.vertex_colors[fake_mesh.faces[:, 1], :].astype(np.float32) \
                          + fake_mesh.visual.vertex_colors[fake_mesh.faces[:, 2], :].astype(np.float32) + fake_mesh.visual.vertex_colors[fake_mesh.faces[:, 3], :].astype(np.float32)
            fake_colors = fake_colors[:, :3] / 4
            fake = torch.from_numpy(fake_colors).to(device) / 127.5 - 1
            fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
            for batch_idx in range(fake_render.shape[0]):
                save_image(fake_render[batch_idx], OUTPUT_DIR_TEXFIELD / f"{iter_idx}_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)
        except Exception as err:
            print("Error Occured:", err)
    compute_fid(OUTPUT_DIR_TEXFIELD, REAL_DIR, OUTPUT_DIR_TEXFIELD.parent / f"score_texfield.txt", device)


# 3D Grid Parameterization
@hydra.main(config_path='../config', config_name='stylegan2_car')
def evaluate_3d_parameterized_gan(config):
    config.batch_size = 1
    config.views_per_sample = 8
    config.image_size = 256
    config.render_size = 512
    num_latent = 5

    from model.styleganvox_sparse import SDFEncoder
    from model.styleganvox_sparse.generator import Generator
    from dataset.meshcar_real_sdfgrid_sparse import SparseSDFGridDataset, Collater
    from MinkowskiEngine.MinkowskiOps import MinkowskiToDenseTensor
    from MinkowskiEngine.MinkowskiSparseTensor import SparseTensor
    import marching_cubes as mc
    import trimesh
    from model.raycast_rgbd.raycast_rgbd import Raycast2DSparseHandler

    OUTPUT_DIR_GRID = OUTPUT_DIR / f"grid_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_GRID.mkdir(exist_ok=True, parents=True)
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/25022319_StyleGAN23D-CompCars_sdfreal-sparse-bgg-lrd1g14-m5-512_128/checkpoints/_epoch=199.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/25022319_StyleGAN23D-CompCars_sdfreal-sparse-bgg-lrd1g14-m5-512_128/checkpoints/ema_000031399.pth"
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
    ema = torch.load(CHECKPOINT_EMA, map_location=device)
    ema.copy_to([p for p in G.parameters() if p.requires_grad])
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
                fake_render = R.raycast_sdf(eval_batch['x_dense'], eval_batch['sparse_data'][0], eval_batch['sparse_data'][1], csdf.contiguous(), eval_batch['view'], eval_batch['intrinsic'])
                fake_render = fake_render.permute((0, 3, 1, 2)).cpu()
            for batch_idx in range(fake_render.shape[0]):
                save_image(fake_render[batch_idx], OUTPUT_DIR_GRID / f"{iter_idx}_{batch_idx}_{z_idx}.jpg", value_range=(-1, 1), normalize=True)

    compute_fid(OUTPUT_DIR_GRID, REAL_DIR, OUTPUT_DIR_GRID.parent / f"score_grid_{config.views_per_sample}_{num_latent}.txt", device)


if __name__ == '__main__':
    create_samples_our_gan()
    # evaluate_3d_parameterized_gan()
    # create_real()
    # evaluate_uv_ours_parameterized_gan()
    # evaluate_uv_normalign_parameterized_gan()
    # evaluate_triplane_gan()
    # evaluate_our_gan()
    # evaluate_texturefields_gan()
