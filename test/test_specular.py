from pathlib import Path

import hydra
from torchvision.utils import save_image
from tqdm import tqdm
import torch
import numpy as np

from dataset.mesh_real_features import FaceGraphMeshDataset
from dataset import GraphDataLoader, to_device, to_vertex_colors_scatter
from model.differentiable_renderer_light import DifferentiableRenderer


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
        rendered_specular = render_helper.render_light(lightdir, batch['view_vector'], batch['vertices'], batch['indices'], to_vertex_colors_scatter(batch["x"], batch)[:, :3], batch["ranges"].cpu()).permute((0, 3, 1, 2)).expand(-1, 3, -1, -1)
        rendered_specular = torch.max(torch.zeros_like(rendered_specular), rendered_specular) ** 30
        batch['real'] = dataset.cspace_convert_back(batch['real'])
        rendered_color_gt = dataset.cspace_convert_back(rendered_color_gt)
        save_image(torch.cat([rendered_color_gt, rendered_specular]), f"runs/images_light/test_view_{batch_idx:04d}.png", nrow=4, value_range=(-1, 1), normalize=True)
        ctr += 1
        if ctr == 5:
            break


def get_light_directions(num_lights, device):
    angles = torch.linspace(0, 2, steps=num_lights + 1, device=device)[:num_lights]
    phi = np.pi * angles
    theta = np.pi * torch.tensor([1.0 / 3.0] * num_lights, device=device)
    light_dirs = []
    for i in range(angles.shape[0]):
        xp = torch.sin(theta[i]) * torch.cos(phi[i])
        zp = torch.sin(theta[i]) * torch.sin(phi[i])
        yp = torch.cos(theta[i])
        z = torch.tensor([xp, yp, zp], device=device)
        z = z / (torch.linalg.norm(z) + 1e-8)
        light_dirs.append(-z)
    return light_dirs


@hydra.main(config_path='../config', config_name='stylegan2')
def test_specular_bounds(config):
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=4, num_workers=0)
    render_helper = DifferentiableRenderer(config.image_size, 'bounds', config.colorspace).cuda()
    Path("runs/images_light").mkdir(exist_ok=True)
    ctr = 0
    shininess = 21
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, torch.device("cuda:0"))
        lightdirs = get_light_directions(3, torch.device("cuda:0"))
        rendered_color_diffuse, rendered_color_spec = render_helper.render(batch['vertices'], batch['indices'],
                                                                         to_vertex_colors_scatter(batch["y"][:, :3], batch),
                                                                         to_vertex_colors_scatter(batch["x"], batch)[:, :3],
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
    # Path("lights.obj").write_text("\n".join([f"v {ld[i][0]} {ld[i][1]} {ld[i][2]}\n" for i in range(len(ld))]))
    test_specular_bounds()
