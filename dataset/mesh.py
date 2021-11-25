import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch_scatter
import trimesh
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Dataset
from collections.abc import Mapping, Sequence
from scipy.spatial.transform import Rotation

from model.differentiable_renderer import intrinsic_to_projection, transform_pos_mvp
from util.misc import EasyDict


def to_device(batch, device):
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    for k in batch['graph_data'].keys():
        if isinstance(batch['graph_data'][k], torch.Tensor):
            batch['graph_data'][k] = batch['graph_data'][k].to(device)
        elif isinstance(batch['graph_data'][k], list):
            for m in range(len(batch['graph_data'][k])):
                if isinstance(batch['graph_data'][k][m], torch.Tensor):
                    batch['graph_data'][k][m] = batch['graph_data'][k][m].to(device)
    return batch


def to_device_graph_data(batch, device):
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
        elif isinstance(batch[k], list):
            for m in range(len(batch[k])):
                if isinstance(batch[k][m], torch.Tensor):
                    batch[k][m] = batch[k][m].to(device)
    return batch


def generate_random_camera(loc):
    x_angles, y_angles = list(range(90, 270, 5)), list(range(0, 360, 5))
    weight_fn = lambda x: 0.5 + 0.125 * math.cos(2 * (math.pi / 45) * x)
    weights_x, weights_y = [weight_fn(x) for x in x_angles], [weight_fn(y) for y in y_angles]
    x_angle, y_angle = random.choices(x_angles, weights_x, k=1)[0], random.choices(y_angles, weights_y, k=1)[0]
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = Rotation.from_euler('y', y_angle, degrees=True).as_matrix() @ Rotation.from_euler('z', 180, degrees=True).as_matrix() @ Rotation.from_euler('x', x_angle, degrees=True).as_matrix()
    # camera_translation = camera_pose[:3, :3] @ np.array([0, 0, 1.025]) + loc
    camera_translation = camera_pose[:3, :3] @ np.array([0, 0, 1.925]) + loc
    camera_pose[:3, 3] = camera_translation
    return camera_pose


def generate_fixed_camera(loc):
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = Rotation.from_euler('y', 0, degrees=True).as_matrix() @ Rotation.from_euler('z', 180, degrees=True).as_matrix() @ Rotation.from_euler('x', 0, degrees=True).as_matrix()
    # camera_translation = camera_pose[:3, :3] @ np.array([0, 0, 1.025]) + loc
    camera_translation = camera_pose[:3, :3] @ np.array([0, 0, 1.925]) + loc
    camera_pose[:3, 3] = camera_translation
    return camera_pose


def get_default_perspective_cam():
    camera = np.array([[886.81, 0., 512., 0.],
                       [0., 886.81, 512., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]], dtype=np.float32)
    return camera


def to_vertex_colors_scatter(face_colors, batch):
    vertex_colors = torch.zeros((batch["vertices"].shape[0] // batch["mvp"].shape[1], face_colors.shape[1])).to(face_colors.device)
    torch_scatter.scatter_mean(face_colors.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3), batch["indices_quad"].reshape(-1).long(), dim=0, out=vertex_colors)
    return vertex_colors[batch['vertex_ctr'], :]


class FaceGraphMeshDataset(torch.utils.data.Dataset):

    def __init__(self, config, limit_dataset_size=None):
        self.dataset_directory = Path(config.dataset_path)
        self.mesh_directory = Path(config.mesh_path)
        self.items = list(x.stem for x in Path(config.dataset_path).iterdir())[:limit_dataset_size]
        self.target_name = "model_normalized.obj"
        self.mask = lambda x, bs: torch.ones((x.shape[0],)).float().to(x.device)
        self.indices_src, self.indices_dest_i, self.indices_dest_j, self.faces_to_uv = [], [], [], None
        self.mesh_resolution = config.mesh_resolution
        self.setup_cube_texture_fast_visualization_buffers()
        self.projection_matrix = intrinsic_to_projection(get_default_perspective_cam()).float()
        self.plane = "Plane" in self.dataset_directory.name
        self.generate_camera = generate_fixed_camera if self.plane else generate_random_camera
        self.views_per_sample = 1 if self.plane else config.views_per_sample
        print("Plane Rendering: ", self.plane)

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
        mesh = trimesh.load(self.mesh_directory / '_'.join(selected_item.split('_')[:-2]) / self.target_name, process=False)
        mvp = torch.stack([torch.matmul(self.projection_matrix, torch.from_numpy(np.linalg.inv(self.generate_camera((mesh.bounds[0] + mesh.bounds[1]) / 2))).float())
                           for _ in range(self.views_per_sample)], dim=0)
        vertices = torch.from_numpy(mesh.vertices).float()
        indices = torch.from_numpy(mesh.faces).int()
        tri_indices = torch.cat([indices[:, [0, 1, 2]], indices[:, [0, 2, 3]]], 0)
        vctr = torch.tensor(list(range(vertices.shape[0]))).long()
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
        mesh = trimesh.load(Path(self.raw_dir, "_".join(name.split('_')[:-2])) / self.target_name, force='mesh', process=False)
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

    @staticmethod
    def to_face_colors(texture, n_faces):
        b, c, h, w = texture.shape
        return texture.reshape((b, c, h * w)).permute((0, 2, 1))[:, :n_faces, :].reshape(-1, 3)

    def setup_cube_texture_fast_visualization_buffers(self):
        # noinspection PyTypeChecker
        mesh = trimesh.load(self.mesh_directory / "coloredbrodatz_D48_COLORED" / self.target_name, process=False)
        vertex_to_uv = np.array(mesh.visual.uv)
        faces_to_vertices = np.array(mesh.faces)
        a = vertex_to_uv[faces_to_vertices[:, 0], :]
        b = vertex_to_uv[faces_to_vertices[:, 1], :]
        c = vertex_to_uv[faces_to_vertices[:, 2], :]
        d = vertex_to_uv[faces_to_vertices[:, 3], :]
        self.faces_to_uv = (a + b + c + d) / 4
        for v_idx in range(self.faces_to_uv.shape[0]):
            j = int(round(self.faces_to_uv[v_idx][0] * (self.mesh_resolution - 1)))
            i = (self.mesh_resolution - 1) - int(round(self.faces_to_uv[v_idx][1] * (self.mesh_resolution - 1)))
            self.indices_dest_i.append(i)
            self.indices_dest_j.append(j)
            self.indices_src.append(v_idx)


class GraphDataLoader(torch.utils.data.DataLoader):

    def __init__(
            self,
            dataset: Dataset,
            batch_size: int = 1,
            shuffle: bool = False,
            follow_batch=None,
            exclude_keys=None,
            **kwargs,
    ):
        if exclude_keys is None:
            exclude_keys = []
        if follow_batch is None:
            follow_batch = []
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning...
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(dataset, batch_size, shuffle, collate_fn=Collater(follow_batch, exclude_keys), **kwargs)


class Collater(object):

    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    @staticmethod
    def cat_collate(batch, dim=0):
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                # noinspection PyProtectedMember
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.cat(batch, dim, out=out)
        raise NotImplementedError

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, EasyDict):
            if 'face_neighborhood' in elem:  # face conv data
                face_neighborhood, sub_neighborhoods, pads, node_counts, pool_maps, is_pad, level_masks = [], [], [], [], [], [], []
                pad_sum = 0

                for b_i in range(len(batch)):
                    face_neighborhood.append(batch[b_i].face_neighborhood + pad_sum)
                    pad_sum += batch[b_i].is_pad[0].shape[0]

                for sub_i in range(len(elem.pads)):
                    is_pad_n = []
                    pad_n = 0
                    for b_i in range(len(batch)):
                        is_pad_n.append(batch[b_i].is_pad[sub_i])
                        batch[b_i].level_masks[sub_i][:] = b_i
                        pad_n += batch[b_i].pads[sub_i]
                    is_pad.append(self.cat_collate(is_pad_n))
                    level_masks.append(self.cat_collate([batch[b_i].level_masks[sub_i] for b_i in range(len(batch))]))
                    pads.append(pad_n)

                for sub_i in range(len(elem.sub_neighborhoods)):
                    sub_n = []
                    pool_n = []
                    node_count_n = 0
                    pad_sum = 0
                    for b_i in range(len(batch)):
                        sub_n.append(batch[b_i].sub_neighborhoods[sub_i] + pad_sum)
                        pool_n.append(batch[b_i].pool_maps[sub_i] + node_count_n)
                        pad_sum += batch[b_i].is_pad[sub_i + 1].shape[0]
                        node_count_n += batch[b_i].node_counts[sub_i]
                    sub_neighborhoods.append(self.cat_collate(sub_n))
                    node_counts.append(node_count_n)
                    pool_maps.append(self.cat_collate(pool_n))

                batch_dot_dict = {
                    'face_neighborhood': self.cat_collate(face_neighborhood),
                    'sub_neighborhoods': sub_neighborhoods,
                    'pads': pads,
                    'node_counts': node_counts,
                    'pool_maps': pool_maps,
                    'is_pad': is_pad,
                    'level_masks': level_masks
                }

                return batch_dot_dict
            else:  # graph conv data
                edge_index, sub_edges, node_counts, pool_maps, level_masks = [], [], [], [], []
                pad_sum = 0
                for b_i in range(len(batch)):
                    edge_index.append(batch[b_i].edge_index + pad_sum)
                    pad_sum += batch[b_i].pool_maps[0].shape[0]

                for sub_i in range(len(elem.sub_edges) + 1):
                    for b_i in range(len(batch)):
                        batch[b_i].level_masks[sub_i][:] = b_i
                    level_masks.append(self.cat_collate([batch[b_i].level_masks[sub_i] for b_i in range(len(batch))]))

                for sub_i in range(len(elem.sub_edges)):
                    sub_n = []
                    pool_n = []
                    node_count_n = 0
                    for b_i in range(len(batch)):
                        sub_n.append(batch[b_i].sub_edges[sub_i] + node_count_n)
                        pool_n.append(batch[b_i].pool_maps[sub_i] + node_count_n)
                        node_count_n += batch[b_i].node_counts[sub_i]
                    sub_edges.append(self.cat_collate(sub_n, dim=1))
                    node_counts.append(node_count_n)
                    pool_maps.append(self.cat_collate(pool_n))

                batch_dot_dict = {
                    'edge_index': self.cat_collate(edge_index, dim=1),
                    'sub_edges': sub_edges,
                    'node_counts': node_counts,
                    'pool_maps': pool_maps,
                    'level_masks': level_masks
                }

                return batch_dot_dict
        elif isinstance(elem, Mapping):
            retdict = {}
            for key in elem:
                if key in ['x', 'y']:
                    retdict[key] = self.cat_collate([d[key] for d in batch])
                elif key == 'vertices':
                    retdict[key] = self.cat_collate([transform_pos_mvp(d['vertices'], d['mvp']) for d in batch])
                elif key == 'indices':
                    num_vertex = 0
                    indices = []
                    for b_i in range(len(batch)):
                        for view_i in range(batch[b_i]['mvp'].shape[0]):
                            indices.append(batch[b_i][key] + num_vertex)
                            num_vertex += batch[b_i]['vertices'].shape[0]
                    retdict[key] = self.cat_collate(indices)
                elif key == 'indices_quad':
                    num_vertex = 0
                    indices_quad = []
                    for b_i in range(len(batch)):
                        indices_quad.append(batch[b_i][key] + num_vertex)
                        num_vertex += batch[b_i]['vertices'].shape[0]
                    retdict[key] = self.cat_collate(indices_quad)
                elif key == 'ranges':
                    ranges = []
                    start_index = 0
                    for b_i in range(len(batch)):
                        for view_i in range(batch[b_i]['mvp'].shape[0]):
                            ranges.append(torch.tensor([start_index, batch[b_i]['indices'].shape[0]]).int())
                            start_index += batch[b_i]['indices'].shape[0]
                    retdict[key] = self.collate(ranges)
                elif key == 'vertex_ctr':
                    vertex_counts = []
                    num_vertex = 0
                    for b_i in range(len(batch)):
                        for view_i in range(batch[b_i]['mvp'].shape[0]):
                            vertex_counts.append(batch[b_i]['vertex_ctr'] + num_vertex)
                        num_vertex += batch[b_i]['vertices'].shape[0]
                    retdict[key] = self.cat_collate(vertex_counts)
                else:
                    retdict[key] = self.collate([d[key] for d in batch])
            return retdict
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            # noinspection PyArgumentList
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self.collate(s) for s in zip(*batch)]
        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)
