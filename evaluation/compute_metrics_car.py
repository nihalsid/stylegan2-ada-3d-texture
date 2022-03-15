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


# UV Parameterization - First view
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
def evaluate_our_gan_texconv(config):
    from dataset.meshcar_real_features_ff2 import FaceGraphMeshDataset
    from model.graph import TwinGraphEncoder, FaceConv, TextureConv
    from model.graph_generator_u_deep_texconv import Generator
    config.batch_size = 1
    config.views_per_sample = 8
    config.image_size = 256
    config.render_size = 512
    config.num_mapping_layers = 8
    config.g_channel_base = 32768
    config.g_channel_max = 768
    num_latent = 5

    OUTPUT_DIR_OURS = OUTPUT_DIR / f"ours_texconv_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    # CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/24021538_StyleGAN23D-CompCars_bigdtwin-res-lrd1g14-v2m8-1K_128/checkpoints/_epoch=89.ckpt"
    # CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/24021538_StyleGAN23D-CompCars_bigdtwin-res-lrd1g14-v2m8-1K_128/checkpoints/ema_000014129.pth"
    # CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021321_StyleGAN23D-CompCars_moredata-clip_fg3bgg-lrd1g14-v4m5-1K_256/checkpoints/_epoch=349.ckpt"
    # CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021321_StyleGAN23D-CompCars_moredata-clip_fg3bgg-lrd1g14-v4m5-1K_256/checkpoints/ema_000054949.pth"
    # CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021429_StyleGAN23D-CompCars_progressive_clip_fg3bgg-lrd1g14-v4m5-1K_512/checkpoints/_epoch=449.ckpt"
    # CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021429_StyleGAN23D-CompCars_progressive_clip_fg3bgg-lrd1g14-v4m5-1K_512/checkpoints/ema_000070649.pth"

    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/08032136_StyleGAN23D-CompCars_p256_texconv_fg3bgg-lrd1g14-v8m8-1K_256/checkpoints/_epoch=299.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/08032136_StyleGAN23D-CompCars_p256_texconv_fg3bgg-lrd1g14-v8m8-1K_256/checkpoints/ema_000047099.pth"

    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

    E = TwinGraphEncoder(eval_dataset.num_feats, 1, layer_dims=(64, 128, 128, 256, 256, 256, 384, 384), conv_layer=TextureConv).to(device)
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

    compute_fid(OUTPUT_DIR_OURS, REAL_DIR, OUTPUT_DIR_OURS.parent / f"score_texconv_{config.views_per_sample}_{num_latent}.txt", device)


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
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021429_StyleGAN23D-CompCars_progressive_clip_fg3bgg-lrd1g14-v4m5-1K_512/checkpoints/_epoch=449.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021429_StyleGAN23D-CompCars_progressive_clip_fg3bgg-lrd1g14-v4m5-1K_512/checkpoints/ema_000070649.pth"

    # CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/24021538_StyleGAN23D-CompCars_bigdtwin-res-lrd1g14-v2m8-1K_128/checkpoints/_epoch=314.ckpt"
    # CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/24021538_StyleGAN23D-CompCars_bigdtwin-res-lrd1g14-v2m8-1K_128/checkpoints/ema_000049454.pth"

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


@hydra.main(config_path='../config', config_name='stylegan2_car')
def create_interesting_samples_our_gan(config):
    from model.graph_generator_u_deep import Generator
    from model.graph import TwinGraphEncoder
    from dataset.meshcar_real_features_ff2 import FaceGraphMeshDataset
    config.batch_size = 1
    config.views_per_sample = 8
    config.image_size = 512
    config.render_size = 512
    config.num_mapping_layers = 5
    config.g_channel_base = 32768
    config.g_channel_max = 768

    num_latent = 100

    OUTPUT_DIR_OURS_IMAGES = OUTPUT_DIR / "ours_vis_interesting" / "images"
    OUTPUT_DIR_OURS_CODES = OUTPUT_DIR / "ours_vis_interesting" / "codes"


    OUTPUT_DIR_OURS_IMAGES.mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR_OURS_CODES.mkdir(exist_ok=True, parents=True)
    # CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/24021538_StyleGAN23D-CompCars_bigdtwin-res-lrd1g14-v2m8-1K_128/checkpoints/_epoch=89.ckpt"
    # CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/24021538_StyleGAN23D-CompCars_bigdtwin-res-lrd1g14-v2m8-1K_128/checkpoints/ema_000014129.pth"
    # CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021321_StyleGAN23D-CompCars_moredata-clip_fg3bgg-lrd1g14-v4m5-1K_256/checkpoints/_epoch=349.ckpt"
    # CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021321_StyleGAN23D-CompCars_moredata-clip_fg3bgg-lrd1g14-v4m5-1K_256/checkpoints/ema_000054949.pth"

    # progressive_64
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021429_StyleGAN23D-CompCars_progressive_clip_fg3bgg-lrd1g14-v4m5-1K_512/checkpoints/_epoch=449.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021429_StyleGAN23D-CompCars_progressive_clip_fg3bgg-lrd1g14-v4m5-1K_512/checkpoints/ema_000070649.pth"

    # CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/24021538_StyleGAN23D-CompCars_bigdtwin-res-lrd1g14-v2m8-1K_128/checkpoints/_epoch=314.ckpt"
    # CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/24021538_StyleGAN23D-CompCars_bigdtwin-res-lrd1g14-v2m8-1K_128/checkpoints/ema_000049454.pth"

    # subset_ids = [3, 4, 5, 6, 8, 9, 12, 13, 16, 17, 19, 20, 22, 24, 27, 28, 31, 34, 35, 36, 38, 39, 40, 42, 43, 44, 47, 48, 50, 51, 52, 54, 55, 56, 57, 62, 63, 64, 65, 66, 67, 69, 71, 72, 74, 75, 79, 80, 87, 88, 90, 93, 96, 99, 103, 105, 106, 107, 110, 111, 113, 116, 117, 118, 121, 123, 124, 125, 126, 128, 130, 134, 135, 139, 140, 141, 143, 145, 146, 147, 148, 149, 154, 155, 156, 157, 159, 163, 166, 168, 169, 171, 172, 173, 174, 175, 179, 180, 181, 184, 185, 186, 189, 190, 191, 192, 193, 194, 195, 196, 198, 199, 201, 202, 204, 212, 214, 215, 216, 217, 220, 222, 224, 225, 226, 227, 228, 231, 236, 237, 238, 239, 240, 249, 255, 257, 258, 262, 263, 264, 265, 267, 269, 270, 271, 272, 273, 274, 278, 283, 285, 286, 288, 289, 292, 295, 298, 300, 306, 308, 310, 313, 316, 318, 319, 320, 325, 327, 328, 332, 336, 337, 339, 340, 342, 343, 345, 346, 350, 351, 354, 355, 356, 357, 358, 359, 360, 361, 362, 366, 367, 370, 372, 374, 378, 380, 381, 382, 386, 388, 390, 391, 395, 396, 398, 399, 400, 405, 409, 411, 413, 417, 421, 424, 425, 426, 431, 434, 436, 437, 438, 439, 444, 446, 448, 453, 455, 456, 457, 459, 461, 462, 464, 467, 470, 471, 474, 476, 477, 478, 481, 482, 484, 486, 487, 490, 491, 495, 498, 499, 500, 501, 502, 504, 505, 507, 516, 520, 521, 522, 524, 527, 528, 529, 530, 532, 533, 534, 539, 542, 547, 548, 553, 556, 557, 558, 560, 563, 566, 567, 572, 573, 574, 575, 576, 577, 579, 583, 585, 587, 588, 590, 591, 594, 595, 596, 600, 602, 603, 608, 609, 610, 612, 613, 614, 615, 620, 625, 627, 628, 631, 636, 640, 643, 644, 646, 647, 648, 654, 656, 657, 659, 660, 661, 662, 663, 665, 666, 667, 668, 671, 672, 673, 674, 676, 678, 679, 681, 682, 684, 687, 688, 690, 694, 695, 696, 698, 706, 710, 713, 714, 717, 719, 720, 722, 723, 725, 727, 733, 734, 737, 739, 740, 742, 745, 746, 751, 752, 756, 757, 759, 763, 765, 766, 768, 769, 772, 773, 774, 775, 779, 780, 784, 785, 787, 788, 792, 798, 799, 800, 801, 802, 803, 805, 806, 807, 808, 809, 811, 815, 817, 818, 820, 821, 822, 824, 826, 828, 830, 831, 833, 839, 840, 843, 844, 847, 848, 849, 850, 852, 853, 854, 858, 860, 861, 862, 864, 866, 868, 869, 870, 871, 872, 873, 875, 876, 880, 881, 882, 883, 886, 888, 889, 893, 895, 896, 900, 901, 902, 904, 905, 906, 907, 908, 912, 924, 925, 926, 927, 929, 930, 931, 932, 936, 937, 938, 939, 941, 942, 943, 944, 945, 949, 954, 959, 960, 961, 962, 963, 968, 969, 971, 972, 973, 977, 978, 980, 982, 983, 985, 987, 989, 990, 992, 993, 996, 999, 1001, 1002, 1003, 1006, 1008, 1012, 1013, 1014, 1015, 1016, 1020, 1022, 1025, 1029, 1030, 1031, 1033, 1034, 1037, 1038, 1045, 1046, 1048, 1051, 1052, 1053, 1054, 1055, 1056, 1059, 1062, 1064, 1067, 1069, 1070, 1071, 1075, 1076, 1078, 1079, 1081, 1083, 1084, 1086, 1091, 1092, 1093, 1094, 1095, 1098, 1099, 1100, 1101, 1102, 1103, 1106, 1107, 1109, 1110, 1111, 1112, 1123, 1125, 1126, 1132, 1133, 1135, 1136, 1137, 1138, 1141, 1143, 1144, 1146, 1148, 1150, 1151, 1153, 1154, 1155, 1156, 1158, 1159, 1160, 1161, 1163, 1164, 1165, 1167, 1169, 1170, 1174, 1175, 1178, 1182, 1184, 1185, 1187, 1189, 1190, 1191, 1192, 1195, 1196, 1200, 1204, 1205, 1206, 1215, 1216, 1219, 1224, 1225, 1227, 1228, 1232, 1238, 1239, 1240, 1241, 1242, 1244, 1245, 1246, 1248, 1249, 1250, 1253]
    subset_ids = [9, 4, 6, 28, 34, 40, 42, 51, 88, 107, 116, 124, 130, 139, 143, 156, 159, 186, 196, 215, 216, 16, 231, 236, 265, 274, 278, 283, 288, 310, 350, 360, 411, 413, 417, 436, 490, 560, 566, 732, 742, 756, 848, 943, 944, 985, 1001, 1013, 1034, 1045, 1055, 1083, 1091, 1146, 1167, 1169, 1175, 1185, 1240]
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

    z_all = torch.randn(num_latent, config.batch_size, config.latent_dim).to(device)

    for iter_idx, batch in enumerate(tqdm(eval_loader)):
        if iter_idx not in subset_ids:
            continue
        eval_batch = to_device(batch, device)
        shape = E(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
        (OUTPUT_DIR_OURS_CODES / f"{iter_idx:04d}").mkdir(exist_ok=True)
        (OUTPUT_DIR_OURS_IMAGES / f"{iter_idx:04d}").mkdir(exist_ok=True)
        for z_idx in range(num_latent):
            z = z_all[z_idx]
            fake = G(eval_batch['graph_data'], z, shape, noise_mode='const', )
            fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
            torch.save(z.cpu(), OUTPUT_DIR_OURS_CODES / f"{iter_idx:04d}" / f"{z_idx:04d}.pt")
            save_image(fake_render, OUTPUT_DIR_OURS_IMAGES / f"{iter_idx:04d}" / f"{z_idx:04d}.jpg", nrow=config.views_per_sample, value_range=(-1, 1), normalize=True)


@hydra.main(config_path='../config', config_name='stylegan2_car')
def render_mesh(config):
    from model.graph_generator_u_deep import Generator
    from model.graph import TwinGraphEncoder
    from dataset.meshcar_real_features_ff2 import FaceGraphMeshDataset
    import trimesh
    config.batch_size = 1
    config.views_per_sample = 1
    config.image_size = 512
    config.render_size = 512
    config.num_mapping_layers = 5
    config.g_channel_base = 32768
    config.g_channel_max = 768

    OUTPUT_DIR_OURS_MESHES = OUTPUT_DIR / "ours_vis_interesting" / "meshes"
    OUTPUT_DIR_OURS_CODES = OUTPUT_DIR / "ours_vis_interesting" / "codes"

    OUTPUT_DIR_OURS_MESHES.mkdir(exist_ok=True, parents=True)

    # progressive_64
    CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021429_StyleGAN23D-CompCars_progressive_clip_fg3bgg-lrd1g14-v4m5-1K_512/checkpoints/_epoch=449.ckpt"
    CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021429_StyleGAN23D-CompCars_progressive_clip_fg3bgg-lrd1g14-v4m5-1K_512/checkpoints/ema_000070649.pth"

    eval_mesh_ids = [360, 490, 560, 1001, 1013]
    eval_mesh_codes = [[62], [96], [89], [51], [69]]
    # eval_mesh_ids = [1240, 1034, 848, 1146, 944, 742]
    # eval_mesh_codes = [[3, 5, 6, 13, 15, 16, 17, 45, 48, 60, 61, 62, 77, 89, 92, 98]] * len(eval_mesh_ids)

    # eval_mesh_ids = [216] # [265]#[107, 231,]
    # eval_mesh_codes = [[68, 77]]#[[96]]# [[9, 12, 62, 63, 77,]] * len(eval_mesh_ids)

    device = torch.device("cuda:0")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_dataset.items = [eval_dataset.items[i] for i in eval_mesh_ids]
    print(eval_dataset.items)
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


def create_meshes_for_baselines():
    import shutil
    import trimesh

    # selected_shapes = [42]#[1034]# [1240, 1034, 848, 1146, 944, 742]
    selected_shapes = [360, 490, 560, 1001, 1013]
    comparisons_dir = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/compcars/compare_sota")
    device = torch.device("cuda:0")

    for idx in selected_shapes:
        (comparisons_dir / f"{idx:05d}").mkdir(exist_ok=True, parents=True)
    #42->96
    # Texture Field
    @hydra.main(config_path='../config', config_name='stylegan2_car')
    def texture_fields(config):
        from dataset.meshcar_real_eg3d import FaceGraphMeshDataset
        eval_dataset = FaceGraphMeshDataset(config)
        car_data_meshes = Path("/rhome/ysiddiqui/texture_fields_car/out/GAN/car/eval_fix/fake")
        for idx in selected_shapes:
            shutil.copyfile(car_data_meshes / f"{eval_dataset[idx]['name']}.obj", comparisons_dir / f"{idx:05d}" / "texture_fields.obj")

    # LTG
    @hydra.main(config_path='../config', config_name='stylegan2_car')
    def LTG(config):
        from model.stylegan2.generator import Generator
        from dataset.meshcar_real_features_uv import FaceGraphMeshDataset
        config.batch_size = 1
        config.views_per_sample = 1
        config.image_size = 128
        config.render_size = 512
        num_latent = 50
        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/23022321_StyleGAN23D-CompCars_uv_firstview_silhoutte_256/checkpoints/_epoch=199.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/23022321_StyleGAN23D-CompCars_uv_firstview_silhoutte_256/checkpoints/ema_000062799.pth"

        def map_to_vertex(r_texture_atlas, r_batch):
            r_texture_atlas = r_texture_atlas.reshape((-1, r_texture_atlas.shape[2], r_texture_atlas.shape[3], r_texture_atlas.shape[4]))
            vertices_mapped = r_texture_atlas[r_batch["uv"][:, 0].long(), :, (r_batch["uv"][:, 1] * config.image_size).long(), (r_batch["uv"][:, 2] * config.image_size).long()]
            vertices_mapped = torch.cat([vertices_mapped, torch.ones((vertices_mapped.shape[0], 1), device=vertices_mapped.device)], dim=-1)
            return vertices_mapped

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [eval_dataset.items[x] for x in selected_shapes]
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
    @hydra.main(config_path='../config', config_name='stylegan2_car')
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
        from dataset.meshcar_real_sdfgrid_sparse import SparseSDFGridDataset
        from dataset.meshcar_real_sdfgrid_sparse import Collater
        from dataset.meshcar_real_features_uv import FaceGraphMeshDataset
        import marching_cubes as mc

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/25022319_StyleGAN23D-CompCars_sdfreal-sparse-bgg-lrd1g14-m5-512_128/checkpoints/_epoch=199.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/25022319_StyleGAN23D-CompCars_sdfreal-sparse-bgg-lrd1g14-m5-512_128/checkpoints/ema_000031399.pth"

        eval_dataset_ltg = FaceGraphMeshDataset(config)
        eval_dataset_ltg.items = [eval_dataset_ltg.items[x] for x in selected_shapes]
        eval_dataset = SparseSDFGridDataset(config)
        eval_dataset.items = eval_dataset_ltg.items
        eval_loader = DataLoader(eval_dataset, config.batch_size, shuffle=False, drop_last=True, num_workers=0, collate_fn=Collater([], []))

        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, 64, 3).to(device)
        E = SDFEncoder(1).to(device)
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
            for z_idx in range(num_latent):
                z = torch.randn(config.batch_size, config.latent_dim).to(device)
                csdf = G(z, eval_batch['sparse_data_064'][0].long(), eval_batch['sparse_data'][0].long(), shape, noise_mode='const')
                csdf_dense = torch.clamp(to_dense(SparseTensor(csdf, eval_batch['sparse_data'][0].int())).squeeze(0).permute((1, 2, 3, 0)), -1.0, 1.0).cpu().numpy() * 0.5 + 0.5
                tsdf_dense = batch['x_dense'].squeeze(0).squeeze(0).permute((2, 1, 0)).cpu().numpy()
                vertices, triangles = mc.marching_cubes_color(tsdf_dense, csdf_dense, 0)
                vertices[:, :3] = vertices[:, :3] / 128 - 0.5
                mc.export_obj(vertices, triangles, comparisons_dir / f"{selected_shapes[iter_idx]:05d}" / f"grid_{z_idx:02d}.obj")

    # EG3D
    @hydra.main(config_path='../config', config_name='stylegan2_car')
    def EG3D(config):
        from model.eg3d.generator import Generator
        from model.styleganvox import SDFEncoder
        from dataset.meshcar_real_eg3d import FaceGraphMeshDataset
        config.batch_size = 1
        config.views_per_sample = 1
        config.image_size = 128
        config.render_size = 512
        config.num_mapping_layers = 8
        num_latent = 10

        # CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/25021541_StyleGAN23D-CompCars_eg3d_geo_256_condition/checkpoints/_epoch=319.ckpt"
        # CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/25021541_StyleGAN23D-CompCars_eg3d_geo_256_condition/checkpoints/ema_000065939.pth"

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/23021451_StyleGAN23D-CompCars_eg3d_geo_condition/checkpoints/_epoch=119.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/23021451_StyleGAN23D-CompCars_eg3d_geo_condition/checkpoints/ema_000037679.pth"

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [eval_dataset.items[x] for x in selected_shapes]
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
    @hydra.main(config_path='../config', config_name='stylegan2_car')
    def OursUV(config):
        config.batch_size = 1
        config.views_per_sample = 1
        config.image_size = 256
        config.render_size = 512
        num_latent = 10

        from model.uv.generator import Generator, NormalEncoder
        from dataset.meshcar_real_features_uv_ours import FaceGraphMeshDataset, split_tensor_into_six

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/23022329_StyleGAN23D-CompCars_uv_firstview_ours_silhoutte_256/checkpoints/_epoch=139.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/23022329_StyleGAN23D-CompCars_uv_firstview_ours_silhoutte_256/checkpoints/ema_000043959.pth"

        def map_to_vertex(r_texture_atlas, r_batch):
            r_texture_atlas = split_tensor_into_six(r_texture_atlas)
            vertices_mapped = r_texture_atlas[r_batch["uv"][:, 0].long(), :, (r_batch["uv"][:, 1] * (config.image_size // 2)).long(), (r_batch["uv"][:, 2] * (config.image_size // 3)).long()]
            vertices_mapped = torch.cat([vertices_mapped, torch.ones((vertices_mapped.shape[0], 1), device=vertices_mapped.device)], dim=-1)
            return vertices_mapped

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [eval_dataset.items[x] for x in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, shuffle=False, drop_last=True, num_workers=0)
        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.image_size, 3).to(device)
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

    @hydra.main(config_path='../config', config_name='stylegan2_car')
    def OursTConv(config):
        from dataset.meshcar_real_features_ff2 import FaceGraphMeshDataset
        from model.graph import TwinGraphEncoder, FaceConv, TextureConv
        from model.graph_generator_u_deep_texconv import Generator
        config.batch_size = 1
        config.views_per_sample = 1
        config.image_size = 256
        config.render_size = 512
        config.num_mapping_layers = 8
        config.g_channel_base = 32768
        config.g_channel_max = 768
        num_latent = 1

        OUTPUT_DIR_OURS_MESHES = OUTPUT_DIR / "ours_vis_interesting" / "meshes"
        OUTPUT_DIR_OURS_CODES = OUTPUT_DIR / "ours_vis_interesting" / "codes"

        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/08032136_StyleGAN23D-CompCars_p256_texconv_fg3bgg-lrd1g14-v8m8-1K_256/checkpoints/_epoch=299.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/08032136_StyleGAN23D-CompCars_p256_texconv_fg3bgg-lrd1g14-v8m8-1K_256/checkpoints/ema_000047099.pth"

        selected_shapes = [42, 265, 742, 848, 944, 1034, 1146, 1240]

        eval_dataset = FaceGraphMeshDataset(config)
        eval_dataset.items = [x for idx, x in enumerate(eval_dataset.items) if idx in selected_shapes]
        eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

        E = TwinGraphEncoder(eval_dataset.num_feats, 1, layer_dims=(64, 128, 128, 256, 256, 256, 384, 384), conv_layer=TextureConv).to(device)
        G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=E.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
        state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
        G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        ema = torch.load(CHECKPOINT_EMA, map_location=device)
        ema.copy_to([p for p in G.parameters() if p.requires_grad])
        E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
        G.eval()
        E.eval()

        eval_mesh_codes = [[27, 36, 42, 82, 64], [73], [34], [24], [34], [27, 36, 42, 82, 64], [27, 36, 42, 82, 64], [27, 36, 42, 82, 64], [27, 36, 42, 82, 64], [27, 36, 42, 82, 64], ]

        with torch.no_grad():
            for iter_idx, batch in enumerate(tqdm(eval_loader)):
                eval_batch = to_device(batch, device)
                shape = E(eval_batch['x'],  eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
                for z_idx in eval_mesh_codes[iter_idx]:
                    z = torch.load(OUTPUT_DIR_OURS_CODES / f'{selected_shapes[iter_idx]:04d}' / f'{z_idx:04d}.pt').to(device)
                    fake = G(eval_batch['graph_data'], z, shape, noise_mode='const', )
                    vertex_colors = ((torch.clamp(to_vertex_colors_scatter(fake, batch), -1, 1) * 0.5 + 0.5) * 255).int()
                    mesh = trimesh.load(Path(config.mesh_path) / batch['name'][0] / "model_normalized.obj", process=False)
                    out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
                    out_mesh.export(comparisons_dir / f"{selected_shapes[iter_idx]:05d}" / f"ours_tconv_{z_idx:02d}.obj")

    with torch.no_grad():
        texture_fields()
        # OursTConv()
        SPSG()
        LTG()
        EG3D()
        OursUV()


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


def create_latent_samples():
    import trimesh
    from PIL import Image
    import numpy as np

    device = torch.device("cuda:0")

    selected_shapes = [216]#[186]  # [0, 1, 2, 5, 7, 8, 9, 11, 13, 15, 24, 41, 48, 55, 61, 97, 99, 107, 119, 123, 159, 195, 211, 218, 279, 287, 314, 322, 356, 379, 405, 474, 628, 631, 696, 708, 724, 726, 762, 776, 781, 792, 794, 814, 842, 861, 868, 873, 887, 888, 889, 891, 914, 930, 975, 988, 1064, 1075, 1097, 1151, 1155, 1168, 1170, 1176, 1188, 1222, 1226, 1230, 1231, 1233, 1237, 1240, 1328, 1337, 1371, 1373, 1445, 1463, 1466, 1469, 1480, 1482, 1502, 1503, 1505, 1510, 1513, 1517, 1603, 1604, 1620, 1625, 1626, 1642, 1690, 1693, 1702, 1707, 1711, 1740, 1790, 1799, 1808, 1821, 1824, 1829, 1840, 1852, 1853, 1871, 1875, 1925, 1927, 1952, 1955, 1959, 1962, 1963, 1972, 2018, 2024, 2067, 2095, 2158, 2198, 2244, 2245, 2250, 2266, 2291, 2311, 2481, 2482, 2620, 2683, 2883, 2891, 2914, 2929, 3018, 3040, 3065, 3146, 3150, 3154, 3322, 3323, 3482, 3581, 3665, 3669, 3762, 3787, 3817, 3874, 3885, 3932, 3950, 3954, 4848, 4883]
    selected_codes = [62, 40]#[70, 87, 57, 39, 6, 4, 98]#[5, 17, 40] #[0, 3, 5, 6, 8, 13, 14, 15, 16, 17, 23, 28, 40, 45, 46, 61, 62, 86, 89, 96]  # list(range(40)) # list(range(100))  # [1, 5, 10, 15, 20, 34, 50, 60]

    dump_meshes = True

    comparisons_dir = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/compcars/compare_latent_styleconsistency")
    combined_dir = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/compcars/compare_latent_styleconsistency/combined")
    comparisons_dir.mkdir(exist_ok=True)
    combined_dir.mkdir(exist_ok=True)
    device = torch.device("cuda:0")

    OUTPUT_DIR_OURS_CODES = OUTPUT_DIR / "ours_vis_interesting" / "codes"

    R = DifferentiableRenderer(512, "bounds")

    if dump_meshes:
        for idx in selected_shapes:
            (comparisons_dir / "mesh").mkdir(exist_ok=True, parents=True)

    @hydra.main(config_path='../config', config_name='stylegan2_car')
    def Ours(config):
        from model.graph_generator_u_deep import Generator
        from model.graph import TwinGraphEncoder
        from dataset.meshcar_real_features_ff2 import FaceGraphMeshDataset
        config.batch_size = 1
        config.views_per_sample = 8 if not dump_meshes else 1
        config.image_size = 512
        config.render_size = 512
        config.num_mapping_layers = 5
        config.g_channel_base = 32768
        config.g_channel_max = 768

        # progressive_64
        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021429_StyleGAN23D-CompCars_progressive_clip_fg3bgg-lrd1g14-v4m5-1K_512/checkpoints/_epoch=449.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021429_StyleGAN23D-CompCars_progressive_clip_fg3bgg-lrd1g14-v4m5-1K_512/checkpoints/ema_000070649.pth"
        eval_dataset = FaceGraphMeshDataset(config, fixed_view_mode=True)
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

    selected_shapes = [186]#[186]  # [0, 1, 2, 5, 7, 8, 9, 11, 13, 15, 24, 41, 48, 55, 61, 97, 99, 107, 119, 123, 159, 195, 211, 218, 279, 287, 314, 322, 356, 379, 405, 474, 628, 631, 696, 708, 724, 726, 762, 776, 781, 792, 794, 814, 842, 861, 868, 873, 887, 888, 889, 891, 914, 930, 975, 988, 1064, 1075, 1097, 1151, 1155, 1168, 1170, 1176, 1188, 1222, 1226, 1230, 1231, 1233, 1237, 1240, 1328, 1337, 1371, 1373, 1445, 1463, 1466, 1469, 1480, 1482, 1502, 1503, 1505, 1510, 1513, 1517, 1603, 1604, 1620, 1625, 1626, 1642, 1690, 1693, 1702, 1707, 1711, 1740, 1790, 1799, 1808, 1821, 1824, 1829, 1840, 1852, 1853, 1871, 1875, 1925, 1927, 1952, 1955, 1959, 1962, 1963, 1972, 2018, 2024, 2067, 2095, 2158, 2198, 2244, 2245, 2250, 2266, 2291, 2311, 2481, 2482, 2620, 2683, 2883, 2891, 2914, 2929, 3018, 3040, 3065, 3146, 3150, 3154, 3322, 3323, 3482, 3581, 3665, 3669, 3762, 3787, 3817, 3874, 3885, 3932, 3950, 3954, 4848, 4883]
    selected_codes = [5, 17]#[70, 87, 57, 39, 6, 4, 98]#[5, 17, 40] #[0, 3, 5, 6, 8, 13, 14, 15, 16, 17, 23, 28, 40, 45, 46, 61, 62, 86, 89, 96]  # list(range(40)) # list(range(100))  # [1, 5, 10, 15, 20, 34, 50, 60]

    dump_meshes = True

    comparisons_dir = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/compcars/compare_latent_detailed")
    comparisons_dir.mkdir(exist_ok=True)
    device = torch.device("cuda:0")

    max_count = 5 * 60

    OUTPUT_DIR_OURS_CODES = OUTPUT_DIR / "ours_vis_interesting" / "codes"

    R = DifferentiableRenderer(512, "bounds")

    if dump_meshes:
        for idx in selected_shapes:
            (comparisons_dir / "mesh").mkdir(exist_ok=True, parents=True)

    @hydra.main(config_path='../config', config_name='stylegan2_car')
    def Ours(config):
        from model.graph_generator_u_deep import Generator
        from model.graph import TwinGraphEncoder
        from dataset.meshcar_real_features_ff2 import FaceGraphMeshDataset
        config.batch_size = 1
        config.views_per_sample = 8 if not dump_meshes else 1
        config.image_size = 512
        config.render_size = 512
        config.num_mapping_layers = 5
        config.g_channel_base = 32768
        config.g_channel_max = 768

        # progressive_64
        CHECKPOINT = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021429_StyleGAN23D-CompCars_progressive_clip_fg3bgg-lrd1g14-v4m5-1K_512/checkpoints/_epoch=449.ckpt"
        CHECKPOINT_EMA = "/cluster_HDD/gondor/ysiddiqui/stylegan2-ada-3d-texture/runs/21021429_StyleGAN23D-CompCars_progressive_clip_fg3bgg-lrd1g14-v4m5-1K_512/checkpoints/ema_000070649.pth"
        eval_dataset = FaceGraphMeshDataset(config, fixed_view_mode=True)
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


@hydra.main(config_path='../config', config_name='stylegan2_car')
def compute_singularity_stats(config):
    from dataset.meshcar_real_features_ff2 import FaceGraphMeshDataset
    from torch_scatter import scatter_add
    import json
    import trimesh

    dataset = FaceGraphMeshDataset(config, fixed_view_mode=True)
    dataloader = GraphDataLoader(dataset, batch_size=1, num_workers=0)
    hierarchy_path = Path("/cluster/gimli/ysiddiqui/CADTextures/CompCars-model/manifold_combined/")
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


if __name__ == '__main__':
    # create_meshes_for_baselines()
    pass
    # create_latent_samples()
    # create_latent_samples_detailed()
    # create_interesting_samples_our_gan()
    # compute_singularity_stats()
    render_mesh()
    # create_meshes_for_baselines()
    # evaluate_3d_parameterized_gan()
    # create_real()
    # evaluate_uv_ours_parameterized_gan()
    # evaluate_uv_normalign_parameterized_gan()
    # evaluate_triplane_gan()
    # evaluate_our_gan()
    # evaluate_our_gan_texconv()
    # evaluate_texturefields_gan()
