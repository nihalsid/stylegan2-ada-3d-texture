import json
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import math
from collections.abc import Mapping, Sequence

import trimesh
import torch
import torchvision.transforms as T
from torch.utils.data.dataloader import default_collate
from torchvision.io import read_image
from tqdm import tqdm
import MinkowskiEngine as ME

from scipy.spatial.transform import Rotation

from model.differentiable_renderer import transform_pos_mvp
from util.camera import spherical_coord_to_cam


class SparseSDFGridDataset(torch.utils.data.Dataset):

    def __init__(self, config, limit_dataset_size=None):
        self.dataset_directory = Path(config.dataset_path)
        self.mesh_directory = Path(config.mesh_path)
        self.image_size = config.image_size
        self.bg_color = config.random_bg
        self.real_images = {x.name.split('.')[0]: x for x in Path(config.image_path).iterdir() if x.name.endswith('.jpg') or x.name.endswith('.png')}
        self.masks = {x: Path(config.mask_path) / self.real_images[x].name for x in self.real_images}
        self.items = sorted(list(x.stem for x in Path(config.dataset_path).iterdir())[:limit_dataset_size])
        self.items = [x for x in self.items if Path(self.mesh_directory, x ,"model_normalized.obj").exists()]
        self.vox_sizes = (0.03125, 0.015625)
        self.views_per_sample = config.views_per_sample
        self.erode = config.erode
        self.pair_meta, self.all_views = self.load_pair_meta(config.pairmeta_path)
        self.real_images_preloaded, self.masks_preloaded = {}, {}
        if config.preload:
            self.preload_real_images()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        selected_item = self.items[idx]
        sdf_grid_064 = torch.from_numpy(np.load(self.dataset_directory / selected_item / "064.npy")) - self.vox_sizes[0] / 2
        sdf_grid_128 = torch.from_numpy(np.load(self.dataset_directory / selected_item / "128.npy")) - self.vox_sizes[1] / 2
        color_grid = 2 * torch.from_numpy(np.load(self.dataset_directory / selected_item / "128_if.npy")).permute((3, 0, 1, 2)) / 255.0 - 1
        real_sample, masks, view_matrices, projection_matrices, mvp = self.get_image_and_view(selected_item)
        locs_064 = torch.nonzero((torch.abs(sdf_grid_064) < self.vox_sizes[0] * 5))
        locs_064 = torch.cat([locs_064[:, 2:3], locs_064[:, 1:2], locs_064[:, 0:1]], 1).contiguous()
        locs_128 = torch.nonzero((torch.abs(sdf_grid_128) < self.vox_sizes[1] * 5))
        locs_128 = torch.cat([locs_128[:, 2:3], locs_128[:, 1:2], locs_128[:, 0:1]], 1).contiguous()
        sparse_sdf = sdf_grid_128[locs_128[:, 2], locs_128[:, 1], locs_128[:, 0]].unsqueeze(-1).contiguous()
        sparse_color = color_grid[:, locs_128[:, 2], locs_128[:, 1], locs_128[:, 0]].T.contiguous()
        mesh = trimesh.load(self.mesh_directory / selected_item / "model_normalized.obj", process=False)
        faces = torch.from_numpy(mesh.triangles.mean(1)).float()
        vertices = torch.from_numpy(mesh.vertices).float()
        vctr = torch.tensor(list(range(vertices.shape[0]))).long()
        indices = torch.from_numpy(mesh.faces).int()
        tri_indices = torch.cat([indices[:, [0, 1, 2]], indices[:, [0, 2, 3]]], 0)

        if self.bg_color == 'white':
            bg = torch.tensor(1.).float()
        else:
            bg = torch.tensor(random.random() * 2 - 1).float()
        return {
            "name": selected_item,
            "faces": faces,
            "vertex_ctr": vctr,
            "vertices": vertices,
            "indices_quad": indices,
            "indices": tri_indices,
            "ranges": torch.tensor([0, tri_indices.shape[0]]).int(),
            "mvp": mvp,
            "x_dense": sdf_grid_128.unsqueeze(0).float(),
            "x_loc_064": locs_064.long(),
            "x_loc_128": locs_128.long(),
            "x_val": sparse_sdf.float(),
            "x_dense_064": sdf_grid_064.unsqueeze(0).float(),
            "y_val": sparse_color.float(),
            "view": view_matrices[0],
            "intrinsic": projection_matrices[0],
            "real": real_sample[0],
            "mask": masks[0],
            "bg": bg
        }

    def get_image_and_view(self, shape):
        shape_id = int(shape.split('_')[0].split('shape')[1])
        image_selections = self.get_image_selections(shape_id)
        view_selections = random.sample(self.all_views, self.views_per_sample)
        images, masks, view_matrices, projection_matrices, cameras = [], [], [], [], []
        for c_i, c_v in zip(image_selections, view_selections):
            images.append(self.get_real_image(self.meta_to_pair(c_i)))
            masks.append(self.get_real_mask(self.meta_to_pair(c_i)))
            view_matrix, projection_matrix = self.get_camera(c_v['fov'], c_v['azimuth'], c_v['elevation'])
            view_matrices.append(view_matrix)
            projection_matrices.append(projection_matrix)
            perspective_cam = spherical_coord_to_cam(c_v['fov'], c_v['azimuth'], c_v['elevation'])
            mvp_projection_matrix = torch.from_numpy(perspective_cam.projection_mat()).float()
            mvp_view_matrix = torch.from_numpy(perspective_cam.view_mat()).float()
            cameras.append(torch.matmul(mvp_projection_matrix, mvp_view_matrix))
        image = torch.cat(images, dim=0)
        masks = torch.cat(masks, dim=0)
        mvp = torch.stack(cameras, dim=0)
        return image, masks, torch.stack(view_matrices, dim=0), torch.stack(projection_matrices, dim=0), mvp

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
        pad = T.Pad(padding=(150, 150), fill=0)
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
        pad = T.Pad(padding=(150, 150), fill=1)
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

    def get_camera(self, fov, azimuth, elevation):
        y_angle = azimuth * 180 / math.pi
        x_angle = 90 - elevation * 180 / math.pi
        z_angle = 180
        camera_rot = np.eye(4)
        camera_rot[:3, :3] = Rotation.from_euler('x', x_angle, degrees=True).as_matrix() @ Rotation.from_euler('z', z_angle, degrees=True).as_matrix() @ Rotation.from_euler('y', y_angle, degrees=True).as_matrix()
        camera_translation = np.eye(4)
        camera_translation[:3, 3] = np.array([0, 0, 1.75])
        camera_pose = camera_translation @ camera_rot
        translate = torch.tensor([[1.0, 0, 0, 64], [0, 1.0, 0, 64], [0, 0, 1.0, 64], [0, 0, 0, 1.0]]).float()
        scale = torch.tensor([[64.0, 0, 0, 0], [0, 64.0, 0, 0], [0, 0, 64.0, 0], [0, 0, 0, 1.0]]).float()
        world2grid = translate @ scale
        view_matrix = world2grid @ torch.linalg.inv(torch.from_numpy(camera_pose).float())
        camera_intrinsics = torch.zeros((4,))
        f = self.image_size / (2 * np.tan(fov * np.pi / 180 / 2.0))
        camera_intrinsics[0] = f
        camera_intrinsics[1] = f
        camera_intrinsics[2] = self.image_size / 2
        camera_intrinsics[3] = self.image_size / 2
        return view_matrix, camera_intrinsics


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
        elif isinstance(elem, Mapping):
            retdict = {'sparse_data': ME.utils.sparse_collate([d["x_loc_128"] for d in batch], [d["x_val"] for d in batch], [d["y_val"] for d in batch]),
                       'sparse_data_064': ME.utils.sparse_collate([d["x_loc_064"] for d in batch], [torch.zeros_like(d["x_loc_064"]) for d in batch])}
            for key in elem:
                if 'cam_position' in elem:
                    retdict['view_vector'] = self.cat_collate([(d['vertices'].unsqueeze(0).expand(d['cam_position'].shape[0], -1, -1) - d['cam_position'].unsqueeze(1).expand(-1, d['vertices'].shape[0], -1)).reshape(-1, 3) for d in batch])
                if key == 'vertices':
                    retdict[key] = self.cat_collate([transform_pos_mvp(d['vertices'], d['mvp']) for d in batch])
                elif key == 'normals':
                    retdict[key] = self.cat_collate([d['normals'].unsqueeze(0).expand(d['cam_position'].shape[0], -1, -1).reshape(-1, 3) for d in batch])
                elif key == 'uv':
                    uvs = []
                    for b_i in range(len(batch)):
                        batch[b_i][key][:, 0] += b_i * 6
                        uvs.append(batch[b_i][key].unsqueeze(0).expand(batch[b_i]['cam_position'].shape[0], -1, -1).reshape(-1, 3))
                    retdict[key] = self.cat_collate(uvs)
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
                elif key == 'uv_vertex_ctr':
                    retdict[key] = [d[key] for d in batch]
                elif key not in ["x_loc", "x_loc_128", "x_loc_064", "x_val", "y_val"]:
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
