import json
import random
import trimesh
from collections import defaultdict
from pathlib import Path
import numpy as np

import torch
import torchvision.transforms as T
from torchvision.io import read_image
from tqdm import tqdm

from util.camera import spherical_coord_to_cam


class SDFGridDataset(torch.utils.data.Dataset):

    def __init__(self, config, limit_dataset_size=None):
        self.dataset_directory = Path(config.dataset_path)
        self.mesh_directory = Path(config.dataset_path)
        try:
            self.image_size = config.img_size
        except Exception as err:
            self.image_size = config.image_size
        self.bg_color = config.random_bg
        self.real_images = {x.name.split('.')[0]: x for x in Path(config.image_path).iterdir() if x.name.endswith('.jpg') or x.name.endswith('.png')}
        self.masks = {x: Path(config.mask_path) / self.real_images[x].name for x in self.real_images}
        self.items = sorted(list(x.stem for x in Path(config.dataset_path).iterdir())[:limit_dataset_size])
        problem_files = {"shape03283_rank00_pair26092", "shape04324_rank00_pair46014", "shape04142_rank00_pair284841", "shape03765_rank01_pair164915", "shape03704_rank02_pair249529", "shape06471_rank00_pair221396"}
        self.items = [x for x in self.items if x not in problem_files]
        self.views_per_sample = 1
        self.erode = config.erode
        self.real_pad = 100
        self.color_generator = random_color if config.random_bg == 'color' else (random_grayscale if config.random_bg == 'grayscale' else white)
        self.pair_meta, self.all_views = self.load_pair_meta(config.pairmeta_path)
        self.real_images_preloaded, self.masks_preloaded = {}, {}
        if config.preload:
            self.preload_real_images()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        selected_item = self.items[idx]
        mesh = trimesh.load(self.dataset_directory / selected_item / "model_normalized.obj", process=False)
        faces = torch.from_numpy(mesh.triangles.mean(1)).float()
        vertices = torch.from_numpy(mesh.vertices).float()
        vctr = torch.tensor(list(range(vertices.shape[0]))).long()
        indices = torch.from_numpy(mesh.faces).int()
        sdf_grid = torch.from_numpy(np.load(self.dataset_directory / selected_item / "064.npy")).unsqueeze(0) - 0.0075
        color_grid = torch.from_numpy(np.load(self.dataset_directory / selected_item / "064_if.npy")).permute((3, 0, 1, 2)) / 127.5 - 1
        tri_indices = torch.cat([indices[:, [0, 1, 2]], indices[:, [0, 2, 3]]], 0)
        real_sample, masks, mvp = self.get_image_and_view(selected_item)
        background = self.color_generator(self.views_per_sample)
        return {
            "name": selected_item,
            "sdf_x": sdf_grid.float(),
            "csdf_y": color_grid.float(),
            "faces": faces,
            "vertex_ctr": vctr,
            "vertices": vertices,
            "indices_quad": indices,
            "indices": tri_indices,
            "ranges": torch.tensor([0, tri_indices.shape[0]]).int(),
            "mvp": mvp,
            "real": real_sample[0].unsqueeze(0),
            "mask": masks[0].unsqueeze(0),
            "bg": torch.cat([background, torch.ones([background.shape[0], 1, 1, 1])], dim=1)
        }

    def get_image_and_view(self, shape):
        shape_id = int(shape.split('_')[0].split('shape')[1])
        image_selections = self.get_image_selections(shape_id)
        view_selections = random.sample(self.all_views, self.views_per_sample)
        images, masks, cameras = [], [], []
        for c_i, c_v in zip(image_selections, view_selections):
            images.append(self.get_real_image(self.meta_to_pair(c_i)))
            masks.append(self.get_real_mask(self.meta_to_pair(c_i)))
            perspective_cam = spherical_coord_to_cam(c_v['fov'], c_v['azimuth'], c_v['elevation'])
            projection_matrix = torch.from_numpy(perspective_cam.projection_mat()).float()
            view_matrix = torch.from_numpy(perspective_cam.view_mat()).float()
            cameras.append(torch.matmul(projection_matrix, view_matrix))
        image = torch.cat(images, dim=0)
        masks = torch.cat(masks, dim=0)
        mvp = torch.stack(cameras, dim=0)
        return image, masks, mvp

    def get_real_image(self, name):
        if name not in self.real_images_preloaded.keys():
            return self.process_real_image(self.real_images[name])
        else:
            return self.real_images_preloaded[name]

    def get_real_mask(self, name):
        if name not in self.masks_preloaded.keys():
            return self.process_real_mask(self.masks[name])
        else:
            return self.masks_preloaded[name]

    @staticmethod
    def erode_mask(mask):
        import cv2 as cv
        mask = mask.squeeze(0).numpy().astype(np.uint8)
        kernel_size = 3
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1), (kernel_size, kernel_size))
        mask = cv.erode(mask, element)
        return torch.from_numpy(mask).unsqueeze(0)

    def process_real_mask(self, path):
        resize = T.Resize(size=(self.image_size, self.image_size))
        pad = T.Pad(padding=(self.real_pad, self.real_pad), fill=0)
        if self.erode:
            eroded_mask = self.erode_mask(read_image(str(path)))
        else:
            eroded_mask = read_image(str(path))
        t_mask = resize(pad((eroded_mask > 0).float()))
        return t_mask.unsqueeze(0)

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
        pad = T.Pad(padding=(self.real_pad, self.real_pad), fill=1)
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
            self.masks_preloaded[ri] = self.process_real_mask(self.masks[ri])

    @staticmethod
    def meta_to_pair(c):
        return f'shape{c["shape_id"]:05d}_rank{(c["rank"] - 1):02d}_pair{c["id"]}'


def random_color(num_views):
    randoms = []
    for i in range(num_views):
        r, g, b = random.randint(0, 255) / 127.5 - 1, random.randint(0, 255) / 127.5 - 1, random.randint(0, 255) / 127.5 - 1
        randoms.append(torch.from_numpy(np.array([r, g, b]).reshape((1, 3, 1, 1))).float())
    return torch.cat(randoms, dim=0)


def random_grayscale(num_views):
    randoms = []
    for i in range(num_views):
        c = random.randint(0, 255) / 127.5 - 1
        randoms.append(torch.from_numpy(np.array([c, c, c]).reshape((1, 3, 1, 1))).float())
    return torch.cat(randoms, dim=0)


def white(num_views):
    return torch.from_numpy(np.array([1, 1, 1]).reshape((1, 3, 1, 1))).expand(num_views, -1, -1, -1).float()
