import json
import os
import random
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import trimesh
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from util.camera import spherical_coord_to_cam


class FaceGraphMeshDataset(torch.utils.data.Dataset):

    def __init__(self, config, limit_dataset_size=None, single_mode=False):
        self.dataset_directory = Path(config.dataset_path)
        self.mesh_directory = Path(config.mesh_path)
        self.image_size = config.image_size
        self.random_views = config.random_views
        self.camera_noise = config.camera_noise
        self.uv_directory = Path(config.uv_path)
        self.normals_directory = Path(config.normals_path)
        self.real_images = {x.name.split('.')[0]: x for x in Path(config.image_path).iterdir() if x.name.endswith('.jpg') or x.name.endswith('.png')}
        self.masks = {x: Path(config.mask_path) / self.real_images[x].name for x in self.real_images}
        self.erode = config.erode
        if not single_mode:
            self.items = list(x.stem for x in Path(config.dataset_path).iterdir())[:limit_dataset_size]
        else:
            self.items = [config.shape_id]
            if limit_dataset_size is None:
                self.items = self.items * config.epoch_steps
            else:
                self.items = self.items * limit_dataset_size
        # self.items = [x for x in self.items if (self.uv_directory / f'{x}.npy').exists()]
        self.target_name = "model_normalized.obj"
        self.views_per_sample = config.views_per_sample
        self.color_generator = random_color if config.random_bg == 'color' else (random_grayscale if config.random_bg == 'grayscale' else white)
        self.real_images_preloaded, self.masks_preloaded = {}, {}
        if config.preload:
            self.preload_real_images()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        selected_item = self.items[idx]
        pt_arxiv = torch.load(os.path.join(self.dataset_directory, f'{selected_item}.pt'))
        vertices = pt_arxiv['vertices'].float()
        indices = pt_arxiv['indices'].int()
        uv = torch.from_numpy(np.load(str(Path(self.uv_directory, f'{selected_item}.npy')))).float()
        normals = np.array(Image.open(self.normals_directory / f'{selected_item}.jpg').resize((self.image_size, self.image_size), resample=Image.NEAREST))
        # noinspection PyTypeChecker
        tri_indices = torch.cat([indices[:, [0, 1, 2]], indices[:, [0, 2, 3]]], 0)
        vctr = torch.tensor(list(range(vertices.shape[0]))).long()

        real_sample, real_mask, mvp, cam_positions = self.get_image_and_view()
        background = self.color_generator(self.views_per_sample)

        normals = (torch.from_numpy(normals) / 127.5 - 1).permute((2, 0, 1))
        return {
            "name": selected_item,
            "vertex_ctr": vctr,
            "uv_vertex_ctr": vertices.shape[0],
            "vertices": vertices,
            "indices_quad": indices,
            "mvp": mvp,
            "cam_position": cam_positions,
            "real": real_sample,
            "mask": real_mask,
            "bg": torch.cat([background, torch.ones([background.shape[0], 1, 1, 1])], dim=1),
            "indices": tri_indices,
            "ranges": torch.tensor([0, tri_indices.shape[0]]).int(),
            "uv": uv,
            "uv_normals": normals,
        }

    def get_image_and_view(self):
        total_selections = len(self.real_images.keys()) // 8
        available_views = get_car_views()
        view_indices = random.sample(list(range(8)), self.views_per_sample)
        sampled_view = [available_views[vidx] for vidx in view_indices]
        image_indices = random.sample(list(range(total_selections)), self.views_per_sample)
        image_selections = [f'{(iidx * 8 + vidx):05d}' for (iidx, vidx) in zip(image_indices, view_indices)]
        images, masks, cameras, cam_positions = [], [], [], []
        for c_i, c_v in zip(image_selections, sampled_view):
            images.append(self.get_real_image(c_i))
            masks.append(self.get_real_mask(c_i))
            azimuth = c_v['azimuth'] + (random.random() - 0.5) * self.camera_noise
            elevation = c_v['elevation'] + (random.random() - 0.5) * self.camera_noise
            perspective_cam = spherical_coord_to_cam(c_v['fov'], azimuth, elevation)
            projection_matrix = torch.from_numpy(perspective_cam.projection_mat()).float()
            cam_position = torch.from_numpy(np.linalg.inv(perspective_cam.view_mat())[:3, 3]).float()
            view_matrix = torch.from_numpy(perspective_cam.view_mat()).float()
            cameras.append(torch.matmul(projection_matrix, view_matrix))
            cam_positions.append(cam_position)
        image = torch.cat(images, dim=0)
        mask = torch.cat(masks, dim=0)
        mvp = torch.stack(cameras, dim=0)
        cam_positions = torch.stack(cam_positions, dim=0)
        image = image * mask.expand(-1, 3, -1, -1) + (1 - mask).expand(-1, 3, -1, -1) * torch.ones_like(image)
        return image, mask, mvp, cam_positions

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

    def process_real_image(self, path):
        pad_size = int(self.image_size * 0.1)
        resize = T.Resize(size=(self.image_size - 2 * pad_size, self.image_size - 2 * pad_size), interpolation=InterpolationMode.BICUBIC)
        pad = T.Pad(padding=(pad_size, pad_size), fill=1)
        t_image = pad(torch.from_numpy(np.array(resize(Image.open(str(path)))).transpose((2, 0, 1))).float() / 127.5 - 1)
        return t_image.unsqueeze(0)

    def export_mesh(self, name, vertex_colors, output_path):
        try:
            mesh = trimesh.load(self.mesh_directory / name / self.target_name, process=False)
            out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.cpu().numpy(), process=False)
            out_mesh.export(output_path)
        except Exception as err:
            print("Failed exporting mesh", err)

    @staticmethod
    def erode_mask(mask):
        import cv2 as cv
        mask = mask.squeeze(0).numpy().astype(np.uint8)
        kernel_size = 1
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1), (kernel_size, kernel_size))
        mask = cv.erode(mask, element)
        return torch.from_numpy(mask).unsqueeze(0)

    def process_real_mask(self, path):
        pad_size = int(self.image_size * 0.1)
        resize = T.Resize(size=(self.image_size - 2 * pad_size, self.image_size - 2 * pad_size), interpolation=InterpolationMode.NEAREST)
        pad = T.Pad(padding=(pad_size, pad_size), fill=0)
        mask_im = read_image(str(path))[:1, :, :]
        if self.erode:
            eroded_mask = self.erode_mask(mask_im)
        else:
            eroded_mask = mask_im
        t_mask = pad(resize((eroded_mask > 128).float()))
        return t_mask.unsqueeze(0)

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

    @staticmethod
    def get_color_bg_real(batch):
        real_sample = batch['real'] * batch['mask'].expand(-1, 3, -1, -1) + (1 - batch['mask']).expand(-1, 3, -1, -1) * batch['bg'][:, :3, :, :]
        return real_sample


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


def get_semi_random_views(num_views):
    elevation_params = [1.407, 0.207, 0.785, 1.767]
    azimuth = random.sample(np.arange(0, 2 * math.pi).tolist(), num_views)
    elevation = [min(max(x, elevation_params[2]), elevation_params[3])
                 for x in np.random.normal(loc=elevation_params[0], scale=elevation_params[1], size=num_views).tolist()]
    return [{'azimuth': a, 'elevation': e} for a, e in zip(azimuth, elevation)]


def get_random_views(num_views):
    elevation_params = [1.407, 0.207, 0.785, 1.767]
    azimuth = random.sample(np.arange(0, 2 * math.pi).tolist(), num_views)
    elevation = np.random.uniform(low=elevation_params[2], high=elevation_params[3], size=num_views).tolist()
    return [{'azimuth': a, 'elevation': e} for a, e in zip(azimuth, elevation)]


def split_tensor_into_six(t_arr):
    # B, C, H, W
    h, w = t_arr.shape[2:]
    h_, w_ = h // 2, w // 3
    split_array = []
    for b in range(t_arr.shape[0]):
        for i in range(2):
            for j in range(3):
                split_array.append(t_arr[b: b + 1, :, i * h_: (i + 1) * h_, j * w_: (j + 1) * w_])
    return torch.cat(split_array, dim=0)


def get_car_views():
    # front, back, right, left, front_right, front_left, back_right, back_left
    azimuth = [3 * math.pi / 2, math.pi / 2,
               0, math.pi,
               math.pi + math.pi / 3, 0 - math.pi / 3,
               math.pi / 2 + math.pi / 6, math.pi / 2 - math.pi / 6]
    azimuth_noise = [0, 0,
                     0, 0,
                     (random.random() - 0.5) * math.pi / 8, (random.random() - 0.5) * math.pi / 8,
                     (random.random() - 0.5) * math.pi / 8, (random.random() - 0.5) * math.pi / 8,]
    elevation = [math.pi / 2, math.pi / 2,
                 math.pi / 2, math.pi / 2,
                 math.pi / 2 - math.pi / 48, math.pi / 2 - math.pi / 48,
                 math.pi / 2 - math.pi / 48, math.pi / 2 - math.pi / 48]
    elevation_noise = [-random.random() * math.pi / 70, -random.random() * math.pi / 70,
                       0, 0,
                       -random.random() * math.pi / 32, -random.random() * math.pi / 32,
                       0, 0]
    return [{'azimuth': a + an, 'elevation': e + en, 'fov': 40} for a, an, e, en in zip(azimuth, azimuth_noise, elevation, elevation_noise)]

