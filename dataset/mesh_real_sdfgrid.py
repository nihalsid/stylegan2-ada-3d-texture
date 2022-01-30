import json
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import math

import torch
import torchvision.transforms as T
from torchvision.io import read_image
from tqdm import tqdm

from scipy.spatial.transform import Rotation


class SDFGridDataset(torch.utils.data.Dataset):

    def __init__(self, config, limit_dataset_size=None):
        self.dataset_directory = Path(config.dataset_path)
        self.mesh_directory = Path(config.mesh_path)
        self.image_size = config.image_size
        self.real_images = {x.name.split('.')[0]: x for x in Path(config.image_path).iterdir() if x.name.endswith('.jpg') or x.name.endswith('.png')}
        self.items = list(x.stem for x in Path(config.dataset_path).iterdir())[:limit_dataset_size]
        self.views_per_sample = config.views_per_sample
        self.pair_meta, self.all_views = self.load_pair_meta(config.pairmeta_path)
        self.real_images_preloaded = {}
        if config.preload:
            self.preload_real_images()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        selected_item = self.items[idx]
        sdf_grid = torch.from_numpy(np.load(self.dataset_directory / selected_item / "064.npy")).unsqueeze(0) - 0.01
        color_grid = 2 * torch.from_numpy(np.load(self.dataset_directory / selected_item / "064_color.npy")).permute((3, 0, 1, 2)) / 255.0 - 1
        real_sample, view_matrices, projection_matrices = self.get_image_and_view(selected_item)
        return {
            "name": selected_item,
            "x": sdf_grid.float(),
            "y": color_grid.float(),
            "view": view_matrices,
            "intrinsic": projection_matrices,
            "real": real_sample,
        }

    def get_image_and_view(self, shape):
        shape_id = int(shape.split('_')[0].split('shape')[1])
        image_selections = self.get_image_selections(shape_id)
        view_selections = random.sample(self.all_views, self.views_per_sample)
        images, view_matrices, projection_matrices = [], [], []
        for c_i, c_v in zip(image_selections, view_selections):
            images.append(self.get_real_image(self.meta_to_pair(c_i)))
            view_matrix, projection_matrix = self.get_camera(c_v['fov'], c_v['azimuth'], c_v['elevation'])
            view_matrices.append(view_matrix)
            projection_matrices.append(projection_matrix)
        image = torch.cat(images, dim=0)
        return image, torch.stack(view_matrices, dim=0), torch.stack(projection_matrices, dim=0)

    def get_real_image(self, name):
        if name not in self.real_images_preloaded.keys():
            return self.process_real_image(self.real_images[name])
        else:
            return self.real_images_preloaded[name]

    def get_image_selections(self, shape_id):
        candidates = self.pair_meta[shape_id]
        if len(candidates) < self.views_per_sample:
            while len(candidates) < self.views_per_sample:
                meta = self.pair_meta[random.choice(list(self.pair_meta.keys()))]
                candidates.extend(meta[:self.views_per_sample - len(candidates)])
        else:
            candidates = random.sample(candidates, self.views_per_sample)
        return candidates

    def process_real_image(self, path):
        resize = T.Resize(size=(self.image_size, self.image_size))
        pad = T.Pad(padding=(100, 100), fill=1)
        t_image = resize(pad(read_image(str(path)).float() / 127.5 - 1))
        return t_image.unsqueeze(0)

    def load_pair_meta(self, pairmeta_path):
        loaded_json = json.loads(Path(pairmeta_path).read_text())
        ret_dict = defaultdict(list)
        ret_views = []
        for k in loaded_json.keys():
            if self.meta_to_pair(loaded_json[k]) in self.real_images.keys():
                ret_dict[loaded_json[k]['shape_id']].append(loaded_json[k])
                ret_views.append(loaded_json[k])
        return ret_dict, ret_views

    def preload_real_images(self):
        for ri in tqdm(self.real_images.keys(), desc='preload'):
            self.real_images_preloaded[ri] = self.process_real_image(self.real_images[ri])

    @staticmethod
    def meta_to_pair(c):
        return f'shape{c["shape_id"]:05d}_rank{(c["rank"] - 1):02d}_pair{c["id"]}'

    def get_camera(self, fov, azimuth, elevation):
        y_angle = azimuth * 180 / math.pi
        x_angle = 90 - elevation * 180 / math.pi
        z_angle = 180
        camera_rot = np.eye(4)
        camera_rot[:3, :3] = Rotation.from_euler('x', x_angle, degrees=True).as_matrix() @ Rotation.from_euler('z', z_angle, degrees=True).as_matrix() @ Rotation.from_euler('y', y_angle, degrees=True).as_matrix()
        camera_translation = np.eye(4)
        camera_translation[:3, 3] = np.array([0, 0, 1.75])
        camera_pose = camera_translation @ camera_rot
        translate = torch.tensor([[1.0, 0, 0, 48], [0, 1.0, 0, 48], [0, 0, 1.0, 48], [0, 0, 0, 1.0]]).float()
        scale = torch.tensor([[48.0, 0, 0, 0], [0, 48.0, 0, 0], [0, 0, 48.0, 0], [0, 0, 0, 1.0]]).float()
        world2grid = translate @ scale
        view_matrix = world2grid @ torch.linalg.inv(torch.from_numpy(camera_pose).float())
        camera_intrinsics = torch.zeros((4,))
        f = self.image_size / (2 * np.tan(fov * np.pi / 180 / 2.0))
        camera_intrinsics[0] = f
        camera_intrinsics[1] = f
        camera_intrinsics[2] = self.image_size / 2
        camera_intrinsics[3] = self.image_size / 2
        return view_matrix, camera_intrinsics
