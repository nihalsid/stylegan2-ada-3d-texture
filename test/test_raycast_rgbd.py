from model.styleganvox.raycast_rgbd.raycast_rgbd import Raycast2DHandler
import numpy as np
import torch
import random
import math
from PIL import Image
from scipy.spatial.transform import Rotation
from pathlib import Path
import json
import time
from util.camera import spherical_coord_to_cam


def get_random_views(num_views):
    elevation_params = [1.407, 0.207, 0.785, 1.767]
    azimuth = random.sample(np.arange(0, 2 * math.pi).tolist(), num_views)
    elevation = np.random.uniform(low=elevation_params[2], high=elevation_params[3], size=num_views).tolist()
    return [{'fov': 50, 'azimuth': a, 'elevation': e} for a, e in zip(azimuth, elevation)]


def meta_to_pair(c):
    return f'shape{c["shape_id"]:05d}_rank{(c["rank"] - 1):02d}_pair{c["id"]}'


def load_pair_meta_views(image_path, pairmeta_path):
    dataset_images = [x.stem for x in image_path.iterdir()]
    loaded_json = json.loads(Path(pairmeta_path).read_text())
    ret_dict = {}
    for k in loaded_json.keys():
        if meta_to_pair(loaded_json[k]) in dataset_images:
            ret_dict[meta_to_pair(loaded_json[k])] = loaded_json[k]
    return ret_dict


def test_raycast():
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    voxel_size = 0.020834
    dims = (96, 96, 96)
    batch_size, render_shape = 4, (256, 256)
    trunc = 5 * voxel_size
    pairmeta_path = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/metadata/pairs.json")
    image_path = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape/exemplars")

    sdf = torch.from_numpy(np.load("/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs/shape02320_rank00_pair83185/096.npy")).cuda()
    color = torch.from_numpy(np.load("/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs/shape02320_rank00_pair83185/096_if.npy")).cuda()

    sdf = sdf.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1, -1).float() - 0.01
    color = color.permute((3, 0, 1, 2)).unsqueeze(0).expand(batch_size, -1, -1, -1, -1).float()

    raycast_handler = Raycast2DHandler(batch_size, dims, render_shape, voxel_size, trunc)

    views_photoshape = load_pair_meta_views(image_path, pairmeta_path)
    view_keys = sorted(list(views_photoshape.keys()))

    repeats = 10

    for _ in range(repeats):
        projections, views = [], []

        for b_idx in range(batch_size):
            c_v = views_photoshape[random.choice(view_keys)]
            # c_v = views_photoshape[view_keys[0]]
            perspective_cam = spherical_coord_to_cam(c_v['fov'], c_v['azimuth'], c_v['elevation'])

            y_angle = c_v['azimuth'] * 180 / math.pi
            x_angle = 90 - c_v['elevation'] * 180 / math.pi
            z_angle = 180 #random.random() * 90
            camera_rot = np.eye(4)
            camera_rot[:3, :3] = Rotation.from_euler('x', x_angle, degrees=True).as_matrix() @ Rotation.from_euler('z', z_angle, degrees=True).as_matrix() @ Rotation.from_euler('y', y_angle, degrees=True).as_matrix()
            camera_translation = np.eye(4)
            camera_translation[:3, 3] = np.array([0, 0, 1.75])
            camera_pose = camera_translation @ camera_rot

            translate = torch.tensor([[1.0, 0, 0, 48], [0, 1.0, 0, 48], [0, 0, 1.0, 48], [0, 0, 0, 1.0]]).to(sdf.device).float()
            scale = torch.tensor([[48.0, 0, 0, 0], [0, 48.0, 0, 0], [0, 0, 48.0, 0], [0, 0, 0, 1.0]]).to(sdf.device).float()
            world2grid = translate @ scale
            translate_0 = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0.0], [0, 0, 1.0, -0.5], [0, 0, 0, 1.0]]).to(sdf.device).float()
            # view_matrix = torch.from_numpy(perspective_cam.view_mat()).to(sdf.device).float() @ scale @ translate
            # view_matrix = translate @ scale @ translate_0
            # view_matrix = world2grid @ translate_0
            # view_matrix = world2grid @ torch.from_numpy(perspective_cam.view_mat()).to(sdf.device).float()
            view_matrix = world2grid @ torch.linalg.inv(torch.from_numpy(camera_pose).to(sdf.device).float())
            # views.append(torch.linalg.inv(view_matrix))
            views.append(view_matrix)
            camera_intrinsics = torch.zeros((4,)).to(sdf.device)
            f = render_shape[1] / (2 * np.tan(c_v['fov'] * np.pi / 180 / 2.0))
            camera_intrinsics[0] = f
            camera_intrinsics[1] = f
            camera_intrinsics[2] = render_shape[0] / 2
            camera_intrinsics[3] = render_shape[0] / 2
            # camera_intrinsics = torch.ones((4,)).to(sdf.device)
            projections.append(camera_intrinsics)

        r_color, r_depth, r_normals = raycast_handler.raycast_sdf(sdf, color, torch.stack(views, dim=0), torch.stack(projections, dim=0))
        print(r_color.max(), r_color.min(), r_color.shape)
        for idx in range(r_color.shape[0]):
            color_i = r_color[idx].cpu().numpy()
            color_i[color_i == -float('inf')] = 0
            Image.fromarray(color_i.astype(np.uint8)).save(f"render_{idx}.jpg")

        time.sleep(2)


def test_views():
    import trimesh
    import pyrender
    from pyrender import RenderFlags
    mesh = trimesh.load("/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs/shape02320_rank00_pair83185/096.obj", process=True)
    for idx, c_v in enumerate(get_random_views(4)):
        spherical_camera = spherical_coord_to_cam(c_v['fov'], c_v['azimuth'], c_v['elevation'])
        translate_0 = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0.00], [0, 0, 1.0, -1.75], [0, 0, 0, 1.0]])
        translate = np.array([[1.0, 0, 0, -48], [0, 1.0, 0, -48], [0, 0, 1.0, -48], [0, 0, 0, 1.0]])
        scale = np.array([[1.0 / 48, 0, 0, 0], [0, 1.0 / 48, 0, 0], [0, 0, 1.0 / 48, 0], [0, 0, 0, 1.0]])
        # view = spherical_camera.view_mat() @ scale @ translate
        view = translate_0 @ scale @ translate
        # view = spherical_camera.view_mat()
        camera_pose = np.linalg.inv(view)
        r = pyrender.OffscreenRenderer(256, 256)
        camera = pyrender.PerspectiveCamera(yfov=np.pi * c_v['fov'] / 180, aspectRatio=1.0, znear=0.001)
        camera_intrinsics = np.eye(4, dtype=np.float32)
        camera_intrinsics[0, 0] = camera_intrinsics[1, 1] = 256 / (2 * np.tan(camera.yfov / 2.0))
        camera_intrinsics[0, 2] = camera_intrinsics[1, 2] = camera_intrinsics[2, 2] = -256 / 2
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=[0.5, 0.5, 0.5])
        geo = pyrender.Mesh.from_trimesh(mesh)
        scene.add(geo)
        scene.add(camera, pose=camera_pose)
        color_flat, depth = r.render(scene, flags=RenderFlags.FLAT | RenderFlags.SKIP_CULL_FACES)
        Image.fromarray(color_flat).save(f"render_{idx}.jpg")


if __name__ == "__main__":
    # test_views()
    test_raycast()
