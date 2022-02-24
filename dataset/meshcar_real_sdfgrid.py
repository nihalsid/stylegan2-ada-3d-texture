import random
from pathlib import Path
import numpy as np
import math

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from scipy.spatial.transform import Rotation


class SDFGridDataset(torch.utils.data.Dataset):

    def __init__(self, config, limit_dataset_size=None):
        self.dataset_directory = Path(config.dataset_path)
        self.mesh_directory = Path(config.mesh_path)
        self.image_size = config.image_size
        self.bg_color = config.random_bg
        self.real_images = {x.name.split('.')[0]: x for x in Path(config.image_path).iterdir() if x.name.endswith('.jpg') or x.name.endswith('.png')}
        self.masks = {x: Path(config.mask_path) / self.real_images[x].name for x in self.real_images}
        self.items = sorted(list(x.stem for x in Path(config.dataset_path).iterdir())[:limit_dataset_size])
        self.views_per_sample = 1
        self.erode = config.erode
        self.real_pad_factor = 0.1
        self.real_images_preloaded, self.masks_preloaded = {}, {}
        if config.preload:
            self.preload_real_images()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        selected_item = self.items[idx]
        sdf_grid = torch.from_numpy(np.load(self.mesh_directory / selected_item / "064.npy")).unsqueeze(0) - (0.03125 / 2)
        normal_grid = compute_normals_from_sdf_dense(sdf_grid)
        real_sample, masks, view_matrices, projection_matrices = self.get_image_and_view()
        if self.bg_color == 'white':
            bg = torch.tensor(1.).float()
        else:
            bg = torch.tensor(random.random() * 2 - 1).float()
        return {
            "name": selected_item,
            "x": sdf_grid.float(),
            "y": normal_grid.float(),
            "view": view_matrices[0],
            "intrinsic": projection_matrices[0],
            "real": real_sample[0],
            "mask": masks[0],
            "bg": bg
        }

    def get_image_and_view(self):
        total_selections = len(self.real_images.keys()) // 8
        available_views = get_car_views()
        view_indices = random.sample(list(range(8)), self.views_per_sample)
        sampled_view = [available_views[vidx] for vidx in view_indices]
        image_indices = random.sample(list(range(total_selections)), self.views_per_sample)
        image_selections = [f'{(iidx * 8 + vidx):05d}' for (iidx, vidx) in zip(image_indices, view_indices)]
        images, masks, view_matrices, projection_matrices = [], [], [], []
        for c_i, c_v in zip(image_selections, sampled_view):
            images.append(self.get_real_image(c_i))
            masks.append(self.get_real_mask(c_i))
            view_matrix, projection_matrix = self.get_camera(c_v['fov'], c_v['azimuth'], c_v['elevation'])
            view_matrices.append(view_matrix)
            projection_matrices.append(projection_matrix)
        image = torch.cat(images, dim=0)
        masks = torch.cat(masks, dim=0)
        image = image * masks.expand(-1, 3, -1, -1) + (1 - masks).expand(-1, 3, -1, -1) * torch.ones_like(image)
        return image, masks, torch.stack(view_matrices, dim=0), torch.stack(projection_matrices, dim=0)

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
        kernel_size = 1
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1), (kernel_size, kernel_size))
        mask = cv.erode(mask, element)
        return torch.from_numpy(mask).unsqueeze(0)

    def process_real_mask(self, path):
        pad_size = int(self.image_size * self.real_pad_factor)
        resize = T.Resize(size=(self.image_size - 2 * pad_size, self.image_size - 2 * pad_size), interpolation=InterpolationMode.NEAREST)
        pad = T.Pad(padding=(pad_size, pad_size), fill=0)
        mask_im = read_image(str(path))[:1, :, :]
        if self.erode:
            eroded_mask = self.erode_mask(mask_im)
        else:
            eroded_mask = mask_im
        t_mask = pad(resize((eroded_mask > 128).float()))
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
        pad_size = int(self.image_size * self.real_pad_factor)
        resize = T.Resize(size=(self.image_size - 2 * pad_size, self.image_size - 2 * pad_size), interpolation=InterpolationMode.BICUBIC)
        pad = T.Pad(padding=(pad_size, pad_size), fill=1)
        t_image = pad(torch.from_numpy(np.array(resize(Image.open(str(path)))).transpose((2, 0, 1))).float() / 127.5 - 1)
        return t_image.unsqueeze(0)

    def preload_real_images(self):
        for ri in tqdm(self.real_images.keys(), desc='preload'):
            self.real_images_preloaded[ri] = self.process_real_image(self.real_images[ri])
            self.masks_preloaded[ri] = self.process_real_mask(self.masks[ri])

    def get_camera(self, fov, azimuth, elevation):
        y_angle = azimuth * 180 / math.pi
        x_angle = 90 - elevation * 180 / math.pi
        z_angle = 180
        camera_rot = np.eye(4)
        camera_rot[:3, :3] = Rotation.from_euler('x', x_angle, degrees=True).as_matrix() @ Rotation.from_euler('z', z_angle, degrees=True).as_matrix() @ Rotation.from_euler('y', y_angle, degrees=True).as_matrix()
        camera_translation = np.eye(4)
        camera_translation[:3, 3] = np.array([0, 0, 1.75])
        camera_pose = camera_translation @ camera_rot
        translate = torch.tensor([[1.0, 0, 0, 32], [0, 1.0, 0, 32], [0, 0, 1.0, 32], [0, 0, 0, 1.0]]).float()
        scale = torch.tensor([[32.0, 0, 0, 0], [0, 32.0, 0, 0], [0, 0, 32.0, 0], [0, 0, 0, 1.0]]).float()
        world2grid = translate @ scale
        view_matrix = world2grid @ torch.linalg.inv(torch.from_numpy(camera_pose).float())
        camera_intrinsics = torch.zeros((4,))
        f = self.image_size / (2 * np.tan(fov * np.pi / 180 / 2.0))
        camera_intrinsics[0] = f
        camera_intrinsics[1] = f
        camera_intrinsics[2] = self.image_size / 2
        camera_intrinsics[3] = self.image_size / 2
        return view_matrix, camera_intrinsics


def get_car_views():
    # front, back, right, left, front_right, front_left, back_right, back_left
    azimuth = [3 * math.pi / 2, math.pi / 2,
               0, math.pi,
               math.pi + math.pi / 3, 0 - math.pi / 3,
               math.pi / 2 + math.pi / 6, math.pi / 2 - math.pi / 6]
    azimuth_noise = [0, 0,
                     0, 0,
                     (random.random() - 0.5) * math.pi / 8, (random.random() - 0.5) * math.pi / 8,
                     (random.random() - 0.5) * math.pi / 8, (random.random() - 0.5) * math.pi / 8, ]
    elevation = [math.pi / 2, math.pi / 2,
                 math.pi / 2, math.pi / 2,
                 math.pi / 2 - math.pi / 48, math.pi / 2 - math.pi / 48,
                 math.pi / 2 - math.pi / 48, math.pi / 2 - math.pi / 48]
    elevation_noise = [-random.random() * math.pi / 70, -random.random() * math.pi / 70,
                       0, 0,
                       -random.random() * math.pi / 32, -random.random() * math.pi / 32,
                       0, 0]
    return [{'azimuth': a + an, 'elevation': e + en, 'fov': 50} for a, an, e, en in zip(azimuth, azimuth_noise, elevation, elevation_noise)]


def compute_normals_from_sdf_dense(sdf):
    sdf = sdf.unsqueeze(0)
    dims = sdf.shape[2:]
    sdfx = sdf[:, :, 1:dims[0] - 1, 1:dims[1] - 1, 2:dims[2]] - sdf[:, :, 1:dims[0] - 1, 1:dims[1] - 1, 0:dims[2] - 2]
    sdfy = sdf[:, :, 1:dims[0] - 1, 2:dims[1], 1:dims[2] - 1] - sdf[:, :, 1:dims[0] - 1, 0:dims[1] - 2, 1:dims[2] - 1]
    sdfz = sdf[:, :, 2:dims[0], 1:dims[1] - 1, 1:dims[2] - 1] - sdf[:, :, 0:dims[0] - 2, 1:dims[1] - 1, 1:dims[2] - 1]
    normals = torch.cat([sdfx, sdfy, sdfz], 1)
    normals = torch.nn.functional.pad(normals, [1, 1, 1, 1, 1, 1], value=0)
    normals = -torch.nn.functional.normalize(normals, p=2, dim=1, eps=1e-5, out=None)
    return normals.squeeze(0)
