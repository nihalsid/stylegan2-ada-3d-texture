import torch
import torch_scatter
import numpy as np
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Dataset
from collections.abc import Mapping, Sequence

from model.differentiable_renderer import transform_pos_mvp
from util.misc import EasyDict


def get_default_perspective_cam():
    camera = np.array([[886.81, 0., 512., 0.],
                       [0., 886.81, 512., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]], dtype=np.float32)
    return camera


def to_vertex_colors_scatter(face_colors, batch):
    vertex_colors = torch.zeros((batch["vertices"].shape[0] // batch["mvp"].shape[1], face_colors.shape[1] + 1)).to(face_colors.device)
    torch_scatter.scatter_mean(face_colors.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3), batch["indices_quad"].reshape(-1).long(), dim=0, out=vertex_colors)
    vertex_colors[:, 3] = 1
    return vertex_colors[batch['vertex_ctr'], :]


def to_device(batch, device):
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    if 'graph_data' in batch:
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
                face_neighborhood, sub_neighborhoods, pads, node_counts, pool_maps, lateral_maps, semantic_maps, is_pad, level_masks = [], [], [], [], [], [], [], [], []
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

                if 'semantic_maps' in elem:
                    for sub_i in range(len(elem.pads)):
                        sem = []
                        for b_i in range(len(batch)):
                            sem.append(batch[b_i].semantic_maps[sub_i])
                        semantic_maps.append(self.cat_collate(sem))

                for sub_i in range(len(elem.sub_neighborhoods)):
                    sub_n = []
                    pool_n = []
                    lateral_n = []
                    node_count_n = 0
                    pad_sum = 0
                    for b_i in range(len(batch)):
                        sub_n.append(batch[b_i].sub_neighborhoods[sub_i] + pad_sum)
                        pool_n.append(batch[b_i].pool_maps[sub_i] + node_count_n)
                        lateral_n.append(batch[b_i].lateral_maps[sub_i] + node_count_n)
                        pad_sum += batch[b_i].is_pad[sub_i + 1].shape[0]
                        node_count_n += batch[b_i].node_counts[sub_i]
                    sub_neighborhoods.append(self.cat_collate(sub_n))
                    node_counts.append(node_count_n)
                    pool_maps.append(self.cat_collate(pool_n))
                    lateral_maps.append(self.cat_collate(lateral_n))

                batch_dot_dict = {
                    'face_neighborhood': self.cat_collate(face_neighborhood),
                    'sub_neighborhoods': sub_neighborhoods,
                    'pads': pads,
                    'node_counts': node_counts,
                    'pool_maps': pool_maps,
                    'lateral_maps': lateral_maps,
                    'is_pad': is_pad,
                    'semantic_maps': semantic_maps,
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
                if key in ['x', 'y', 'bg', 'x_0', 'x_1']:
                    retdict[key] = self.cat_collate([d[key] for d in batch])
                elif key == 'vertices':
                    retdict[key] = self.cat_collate([transform_pos_mvp(d['vertices'], d['mvp']) for d in batch])
                    retdict['view_vector'] = self.cat_collate([(d['vertices'].unsqueeze(0).expand(d['cam_position'].shape[0], -1, -1) - d['cam_position'].unsqueeze(1).expand(-1, d['vertices'].shape[0], -1)).reshape(-1, 3) for d in batch])
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
                elif key in ['real', 'mask', 'patch', 'real_hres', 'mask_hres']:
                    retdict[key] = self.cat_collate([d[key] for d in batch])
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