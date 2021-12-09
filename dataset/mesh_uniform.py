import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch.utils.data.dataloader import default_collate
from scipy.spatial.transform import Rotation

from dataset import get_default_perspective_cam
from model.differentiable_renderer import intrinsic_to_projection
from util.misc import EasyDict


def generate_random_camera(loc):
    y_angles = list(range(90, 270, 5))
    weight_fn = lambda x: 0.5 + 0.125 * math.cos(2 * (math.pi / 45) * x)
    weights_y = [weight_fn(y) for y in y_angles]
    y_angle = random.choices(y_angles, weights_y, k=1)[0]
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = Rotation.from_euler('y', y_angle, degrees=True).as_matrix() @ Rotation.from_euler('z', 180, degrees=True).as_matrix() @ Rotation.from_euler('x', 180, degrees=True).as_matrix()
    camera_translation = camera_pose[:3, :3] @ np.array([0, 0, 1.250]) + loc
    camera_pose[:3, 3] = camera_translation
    return camera_pose


class FaceGraphMeshDataset(torch.utils.data.Dataset):

    def __init__(self, config, limit_dataset_size=None):
        self.dataset_directory = Path(config.dataset_path)
        self.mesh_directory = Path(config.mesh_path)
        self.items = list(x.stem for x in Path(config.dataset_path).iterdir())[:limit_dataset_size]
        self.target_name = "model_normalized.obj"
        self.mask = lambda x, bs: torch.ones((x.shape[0],)).float().to(x.device)
        self.indices_src, self.indices_dest_i, self.indices_dest_j, self.faces_to_uv = [], [], [], None
        self.projection_matrix = intrinsic_to_projection(get_default_perspective_cam()).float()
        self.generate_camera = generate_random_camera
        self.views_per_sample = config.views_per_sample

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        selected_item = self.items[idx]
        pt_arxiv = torch.load(os.path.join(self.dataset_directory, f'{selected_item}.pt'))
        edge_index = pt_arxiv['conv_data'][0][0].long()
        num_sub_vertices = [pt_arxiv['conv_data'][i][0].shape[0] for i in range(1, len(pt_arxiv['conv_data']))]
        pad_sizes = [pt_arxiv['conv_data'][i][2].shape[0] for i in range(len(pt_arxiv['conv_data']))]
        sub_edges = [pt_arxiv['conv_data'][i][0].long() for i in range(1, len(pt_arxiv['conv_data']))]
        pool_maps = pt_arxiv['pool_locations']
        is_pad = [pt_arxiv['conv_data'][i][4].bool() for i in range(len(pt_arxiv['conv_data']))]
        level_masks = [torch.zeros(pt_arxiv['conv_data'][i][0].shape[0]).long() for i in range(len(pt_arxiv['conv_data']))]

        # noinspection PyTypeChecker
        mesh = trimesh.load(self.mesh_directory / selected_item / self.target_name, process=False)
        mvp = torch.stack([torch.matmul(self.projection_matrix, torch.from_numpy(np.linalg.inv(self.generate_camera((mesh.bounds[0] + mesh.bounds[1]) / 2))).float())
                           for _ in range(self.views_per_sample)], dim=0)
        vertices = torch.from_numpy(mesh.vertices).float()
        indices = torch.from_numpy(mesh.faces).int()
        tri_indices = torch.cat([indices[:, [0, 1, 2]], indices[:, [0, 2, 3]]], 0)
        vctr = torch.tensor(list(range(vertices.shape[0]))).long()
        r, g, b = random.randint(0, 255) / 255. - 0.5, random.randint(0, 255) / 255. - 0.5, random.randint(0, 255) / 255. - 0.5
        pt_arxiv['target_colors'][:, 0] = r
        pt_arxiv['target_colors'][:, 1] = g
        pt_arxiv['target_colors'][:, 2] = b
        return {
            "name": selected_item,
            "y": pt_arxiv['target_colors'].float() * 2,
            "vertex_ctr": vctr,
            "vertices": vertices,
            "indices_quad": indices,
            "mvp": mvp,
            "indices": tri_indices,
            "ranges": torch.tensor([0, tri_indices.shape[0]]).int(),
            "graph_data": self.get_item_as_graphdata(edge_index, sub_edges, pad_sizes, num_sub_vertices, pool_maps, is_pad, level_masks)
        }

    @staticmethod
    def get_item_as_graphdata(edge_index, sub_edges, pad_sizes, num_sub_vertices, pool_maps, is_pad, level_masks):
        return EasyDict({
            'face_neighborhood': edge_index,
            'sub_neighborhoods': sub_edges,
            'pads': pad_sizes,
            'node_counts': num_sub_vertices,
            'pool_maps': pool_maps,
            'is_pad': is_pad,
            'level_masks': level_masks
        })

    def visualize_graph_with_predictions(self, name, prediction, output_dir, output_suffix):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        # noinspection PyTypeChecker
        mesh = trimesh.load(Path(self.raw_dir, name) / self.target_name, force='mesh', process=False)
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, face_colors=prediction + 0.5, process=False)
        mesh.export(output_dir / f"{name}_{output_suffix}.obj")

    def to_image(self, face_colors, level_mask):
        batch_size = level_mask.max() + 1
        image = torch.zeros((batch_size, 3, self.mesh_resolution, self.mesh_resolution), device=face_colors.device)
        indices_dest_i = torch.tensor(self.indices_dest_i * batch_size, device=face_colors.device).long()
        indices_dest_j = torch.tensor(self.indices_dest_j * batch_size, device=face_colors.device).long()
        indices_src = torch.tensor(self.indices_src * batch_size, device=face_colors.device).long()
        image[level_mask, :, indices_dest_i, indices_dest_j] = face_colors[indices_src + level_mask * len(self.indices_src), :]
        return image

    @staticmethod
    def batch_mask(t, graph_data, idx, level=0):
        return t[graph_data['level_masks'][level] == idx]
