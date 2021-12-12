import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from torchvision.io import read_image
from tqdm import tqdm
import json

from dataset import get_default_perspective_cam
from model.differentiable_renderer import intrinsic_to_projection
from util.camera import spherical_coord_to_cam
from util.misc import EasyDict
import torchvision.transforms as T


class FaceGraphMeshDataset(torch.utils.data.Dataset):

    def __init__(self, config, limit_dataset_size=None):
        self.dataset_directory = Path(config.dataset_path)
        self.mesh_directory = Path(config.mesh_path)
        self.image_size = config.image_size
        self.real_images = {x.name.split('.')[0]: x for x in Path(config.image_path).iterdir() if x.name.endswith('.jpg') or x.name.endswith('.png')}
        self.items = list(x.stem for x in Path(config.dataset_path).iterdir())[:limit_dataset_size]
        self.target_name = "model_normalized.obj"
        self.generate_camera = generate_random_camera
        self.views_per_sample = config.views_per_sample
        self.pair_meta = self.load_pair_meta(config.pairmeta_path)
        self.real_images_preloaded = {}
        if config.preload:
            self.preload_real_images()

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
        vertices = torch.from_numpy(mesh.vertices).float()
        indices = torch.from_numpy(mesh.faces).int()
        tri_indices = torch.cat([indices[:, [0, 1, 2]], indices[:, [0, 2, 3]]], 0)
        vctr = torch.tensor(list(range(vertices.shape[0]))).long()

        real_sample, mvp = self.get_image_and_view(selected_item)

        return {
            "name": selected_item,
            "y": pt_arxiv['target_colors'].float() * 2,
            "vertex_ctr": vctr,
            "vertices": vertices,
            "indices_quad": indices,
            "mvp": mvp,
            "real": real_sample,
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

    @staticmethod
    def batch_mask(t, graph_data, idx, level=0):
        return t[graph_data['level_masks'][level] == idx]

    def get_image_and_view(self, shape):
        meta_to_pair = lambda c: f'shape{c["shape_id"]:05d}_rank{(c["rank"] - 1):02d}_pair{c["id"]}'
        shape_id = int(shape.split('_')[0].split('shape')[1])
        candidates = self.pair_meta[shape_id]
        candidates = [c for c in candidates if meta_to_pair(c) in self.real_images.keys()]
        if len(candidates) < self.views_per_sample:
            while len(candidates) < self.views_per_sample:
                meta = self.pair_meta[random.choice(list(self.pair_meta.keys()))]
                meta = [c for c in meta if meta_to_pair(c) in self.real_images.keys()]
                candidates.extend(meta[:self.views_per_sample - len(candidates)])
        else:
            candidates = random.sample(candidates, self.views_per_sample)
        images, cameras = [], []
        for c in candidates:
            images.append(self.get_real_image(meta_to_pair(c)))
            perspective_cam = spherical_coord_to_cam(c['fov'], c['azimuth'], c['elevation'])
            # projection_matrix = intrinsic_to_projection(get_default_perspective_cam()).float()
            projection_matrix = torch.from_numpy(perspective_cam.projection_mat()).float()
            # view_matrix = torch.from_numpy(np.linalg.inv(generate_camera(np.zeros(3), c['azimuth'], c['elevation']))).float()
            view_matrix = torch.from_numpy(perspective_cam.view_mat()).float()
            cameras.append(torch.matmul(projection_matrix, view_matrix))
        image = torch.cat(images, dim=0)
        mvp = torch.stack(cameras, dim=0)
        return image, mvp

    def get_real_image(self, name):
        if name not in self.real_images_preloaded.keys():
            resize = T.Resize(size=(self.image_size, self.image_size))
            pad = T.Pad(padding=(100, 100), fill=1)
            image = read_image(str(self.real_images[name])).float() / 127.5 - 1
            return resize(pad(image)).unsqueeze(0)
        else:
            return self.real_images_preloaded[name]

    def process_real_image(self, path):
        resize = T.Resize(size=(self.image_size, self.image_size))
        pad = T.Pad(padding=(100, 100), fill=1)
        return resize(pad(read_image(str(path)).float() / 127.5 - 1)).unsqueeze(0)

    @staticmethod
    def load_pair_meta(pairmeta_path):
        loaded_json = json.loads(Path(pairmeta_path).read_text())
        ret_dict = defaultdict(list)
        for k in loaded_json.keys():
            ret_dict[loaded_json[k]['shape_id']].append(loaded_json[k])
        return ret_dict

    def preload_real_images(self):
        for ri in tqdm(self.real_images.keys(), desc='preload'):
            self.real_images_preloaded[ri] = self.process_real_image(self.real_images[ri])
