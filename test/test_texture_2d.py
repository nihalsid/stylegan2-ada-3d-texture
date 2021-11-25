import hydra
from torchvision.utils import save_image
from tqdm import tqdm
import torch

from dataset.mesh import FaceGraphMeshDataset, GraphDataLoader, to_device, to_vertex_colors_scatter
from model.differentiable_renderer import DifferentiableRenderer


@hydra.main(config_path='../config', config_name='stylegan2')
def test_texture_to_face(config):
    batch_size = 8
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=batch_size, num_workers=0)
    render_helper = DifferentiableRenderer(64).cuda()
    level_mask = torch.tensor(list(range(batch_size))).long().cuda()
    level_mask = level_mask.unsqueeze(-1).expand(-1, config.num_faces[0]).reshape(-1)
    face_color_idx = torch.tensor(dataset.indices_src * batch_size).long().cuda() + level_mask * len(dataset.indices_src)
    face_batch_idx = level_mask
    indices_img_i = torch.tensor(dataset.indices_dest_i * batch_size).long().cuda()
    indices_img_j = torch.tensor(dataset.indices_dest_j * batch_size).long().cuda()
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, torch.device("cuda:0"))
        texture = dataset.to_image(batch['y'], level_mask)
        save_image(texture, "test_texture.png", nrow=4, value_range=(-1, 1), normalize=True)
        face_colors = dataset.to_face_colors(texture, face_color_idx, face_batch_idx, indices_img_i, indices_img_j)
        rendered_color_t2f = render_helper.render(batch['vertices'], batch['indices'], to_vertex_colors_scatter(face_colors, batch), batch["ranges"].cpu())
        rendered_color_gt = render_helper.render(batch['vertices'], batch['indices'], to_vertex_colors_scatter(batch['y'], batch), batch["ranges"].cpu())
        save_image(rendered_color_t2f.permute((0, 3, 1, 2)), "test_texture_to_face.png", nrow=4, value_range=(-1, 1), normalize=True)
        save_image(rendered_color_gt.permute((0, 3, 1, 2)), "test_face.png", nrow=4, value_range=(-1, 1), normalize=True)
        break


if __name__ == '__main__':
    test_texture_to_face()
