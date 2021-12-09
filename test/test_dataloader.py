import hydra
from torchvision.utils import save_image
from tqdm import tqdm
import torch

from dataset.mesh import FaceGraphMeshDataset, GraphDataLoader, to_device, to_vertex_colors_scatter
from model.differentiable_renderer import DifferentiableRenderer
from model.graph import pool, unpool


@hydra.main(config_path='../config', config_name='stylegan2')
def test_dataloader(config):
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=1, num_workers=0)
    render_helper = DifferentiableRenderer(128).cuda()
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # sanity test render + target colors
        batch = to_device(batch, torch.device("cuda:0"))
        print(batch['name'])
        rendered_color_gt = render_helper.render(batch['vertices'], batch['indices'], to_vertex_colors_scatter(batch["y"], batch), batch["ranges"].cpu())
        save_image(rendered_color_gt.permute((0, 3, 1, 2)), "test_dataloader.png", nrow=4, value_range=(-1, 1), normalize=True)
        # sanity test graph counts and pool maps
        x_0 = pool(batch['y'], batch['graph_data']['node_counts'][0], batch['graph_data']['pool_maps'][0])
        x_1 = pool(x_0, batch['graph_data']['node_counts'][1], batch['graph_data']['pool_maps'][1])
        x_1 = unpool(x_1, batch['graph_data']['pool_maps'][1])
        x_0 = unpool(x_1, batch['graph_data']['pool_maps'][0])
        # works only if uv's are present
        # save_image(dataset.to_image(batch["y"], batch["graph_data"]["level_masks"][0]), "test_target.png", nrow=4, value_range=(-1, 1), normalize=True)
        # save_image(dataset.to_image(x_0, batch["graph_data"]["level_masks"][0]), "test_pooled.png", nrow=4, value_range=(-1, 1), normalize=True)
        # break


if __name__ == '__main__':
    test_dataloader()
