from pathlib import Path

import hydra
import torch_scatter
import trimesh
from tqdm import tqdm
import torch

from dataset import GraphDataLoader, to_device
import numpy as np
import json
from model.graph import pool

mesh_directory = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs-manifold-highres/")


def export_semantics_at_level(semantics, batch, num_faces):

    for bid in range(len(batch['name'])):
        selections = json.loads((mesh_directory / batch['name'][bid] / "selection.json").read_text())
        mesh_file = f"quad_{num_faces:05d}_{selections[str(num_faces)]:03d}.obj"
        mesh = trimesh.load(mesh_directory / batch['name'][bid] / mesh_file, process=False)
        vertex_colors = torch.zeros(mesh.vertices.shape).to(semantics.device)
        torch_scatter.scatter_mean(semantics[num_faces * bid: num_faces * (bid + 1), :].unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3),
                                   torch.from_numpy(mesh.faces).to(semantics.device).reshape(-1).long(), dim=0, out=vertex_colors)
        out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
        out_mesh.export(f"runs/semantics/{batch['name'][bid]}_{mesh_file}")


@hydra.main(config_path='../config', config_name='stylegan2')
def test_dataloader(config):
    from dataset.mesh_real_2features import FaceGraphMeshDataset
    hex_to_rgb = lambda x: [int(x[i:i + 2], 16) for i in (1, 3, 5)]
    distinct_colors = ['#ff0000', '#ffff00', '#c71585', '#00fa9a', '#0000ff', '#1e90ff', '#ffdab9']
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=4, num_workers=0)
    Path("runs/semantics").mkdir(exist_ok=True)
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, torch.device("cuda:0"))
        x = batch['x_1']
        for level in range(5):
            colored_semantics = torch.from_numpy(np.array([hex_to_rgb(distinct_colors[label]) for label in torch.argmax(x, dim=1).cpu().numpy().tolist()])).cuda().float()
            export_semantics_at_level(colored_semantics, batch, config.num_faces[level])
            x = pool(x, batch['graph_data']['node_counts'][level], batch['graph_data']['pool_maps'][level], batch['graph_data']['lateral_maps'][level], pool_op='mean')
        break


@hydra.main(config_path='../config', config_name='stylegan2')
def test_semantics_from_data(config):
    from dataset.mesh_real_features_patch_spool import FaceGraphMeshDataset
    hex_to_rgb = lambda x: [int(x[i:i + 2], 16) for i in (1, 3, 5)]
    distinct_colors = ['#ff0000', '#ffff00', '#c71585', '#00fa9a', '#0000ff', '#1e90ff', '#ffdab9']
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=4, num_workers=0)
    Path("runs/semantics").mkdir(exist_ok=True)
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, torch.device("cuda:0"))
        for level in range(5):
            x = batch['graph_data']['semantic_maps'][level]
            colored_semantics = torch.from_numpy(np.array([hex_to_rgb(distinct_colors[label]) for label in x.squeeze().cpu().numpy().tolist()])).cuda().float()
            export_semantics_at_level(colored_semantics, batch, config.num_faces[level])
        break


if __name__ == '__main__':
    # test_dataloader()
    test_semantics_from_data()
