from pathlib import Path

import hydra
from torchvision.utils import save_image
from tqdm import tqdm
import torch
import numpy as np

from dataset.mesh_real_features import FaceGraphMeshDataset
from dataset import GraphDataLoader, to_device, to_vertex_colors_scatter, to_vertex_shininess_scatter
from model.differentiable_renderer_light import DifferentiableRenderer
from trainer.train_stylegan_real_feature_light import get_light_directions, sample_light_directions


@hydra.main(config_path='../config', config_name='stylegan2')
def test_specular(config):
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=4, num_workers=0)
    render_helper = DifferentiableRenderer(config.image_size, 'standard', config.colorspace).cuda()
    Path("runs/images_light").mkdir(exist_ok=True)
    ctr = 0
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, torch.device("cuda:0"))
        lightdir = torch.randn(size=[3], device=batch['vertices'].device)
        lightdir /= lightdir.norm() + 1e-8
        rendered_color_gt = render_helper.render(batch['vertices'], batch['indices'], to_vertex_colors_scatter(batch["y"][:, :3], batch), batch["ranges"].cpu(), batch['bg']).permute((0, 3, 1, 2))
        rendered_specular = render_helper.render_light(lightdir, batch['view_vector'], batch['vertices'], batch['indices'], to_vertex_colors_scatter(batch["x"], batch)[:, :3], batch["ranges"].cpu()).permute((0, 3, 1, 2)).expand(-1, 3, -1,
                                                                                                                                                                                                                                    -1)
        rendered_specular = torch.max(torch.zeros_like(rendered_specular), rendered_specular) ** 30
        batch['real'] = dataset.cspace_convert_back(batch['real'])
        rendered_color_gt = dataset.cspace_convert_back(rendered_color_gt)
        save_image(torch.cat([rendered_color_gt, rendered_specular]), f"runs/images_light/test_view_{batch_idx:04d}.png", nrow=4, value_range=(-1, 1), normalize=True)
        ctr += 1
        if ctr == 5:
            break


@hydra.main(config_path='../config', config_name='stylegan2')
def test_specular_bounds(config):
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=4, num_workers=0)
    render_helper = DifferentiableRenderer(config.image_size, 'bounds', config.colorspace).cuda()
    Path("runs/images_light").mkdir(exist_ok=True)
    ctr = 0
    shininess = 28
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, torch.device("cuda:0"))
        lightdirs = get_light_directions(3, torch.device("cuda:0"))
        # lightdirs = sample_light_directions(lightdirs)
        rendered_color_diffuse, rendered_color_spec = render_helper.render(batch['vertices'], batch['indices'],
                                                                           to_vertex_colors_scatter(batch["y"][:, :3], batch),
                                                                           batch['normals'],
                                                                           to_vertex_shininess_scatter(batch["y"][:, 3:4], batch),
                                                                           lightdirs, batch['view_vector'], shininess,
                                                                           batch["ranges"].cpu(), batch['bg'])
        rendered_color_diffuse = rendered_color_diffuse.permute((0, 3, 1, 2))
        rendered_color_spec = rendered_color_spec.permute((0, 3, 1, 2)).expand(-1, 3, -1, -1)
        rendered_color = rendered_color_diffuse + rendered_color_spec
        batch['real'] = dataset.cspace_convert_back(batch['real'])
        rendered_color = dataset.cspace_convert_back(rendered_color)
        save_image(torch.cat([rendered_color, rendered_color_diffuse, rendered_color_spec]), f"runs/images_light/test_view_{batch_idx:04d}.png", nrow=4, value_range=(-1, 1), normalize=True)
        ctr += 1
        if ctr == 10:
            break


if __name__ == '__main__':
    # ld = get_light_directions(3, torch.device("cpu"))
    # all_lights = []
    # for _sid in range(640):
    #     ld = sample_light_directions(ld)
    #     all_lights.append(ld)
    # ld = torch.cat(all_lights, dim=0)
    # Path(f"lights.obj").write_text("\n".join([f"v {ld[i][0]} {ld[i][1]} {ld[i][2]}\n" for i in range(len(ld))]))
    test_specular_bounds()
