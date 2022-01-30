import json
from pathlib import Path
import numpy as np
import random
from util.camera import spherical_coord_to_cam
import math
from scipy.spatial.transform import Rotation

pairmeta_path = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/metadata/pairs.json")
image_path = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape/exemplars")


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


views_photoshape = load_pair_meta_views(image_path, pairmeta_path)
view_keys = sorted(list(views_photoshape.keys()))


def test_camera_distribution():
    positions = []
    print(len(view_keys))
    for vk in view_keys:
        c_v = views_photoshape[vk]
        noise_azimuth = (random.random() - 0.5) * 0.075
        noise_elevation = (random.random() - 0.5) * 0.075
        perspective_cam = spherical_coord_to_cam(c_v['fov'], c_v['azimuth'] + noise_azimuth, c_v['elevation'] + noise_elevation)
        positions.append(perspective_cam.position)
    Path("camera_distribution.obj").write_text("\n".join([f"v {p[0]} {p[1]} {p[2]}" for p in positions]))


def test_camera_distribution_sdf():
    positions = []
    print(len(view_keys))
    for vk in view_keys:
        c_v = views_photoshape[vk]
        y_angle = c_v['azimuth'] * 180 / math.pi
        x_angle = 90 - c_v['elevation'] * 180 / math.pi
        z_angle = 180  # random.random() * 90
        camera_rot = np.eye(4)
        camera_rot[:3, :3] = Rotation.from_euler('x', x_angle, degrees=True).as_matrix() @ Rotation.from_euler('z', z_angle, degrees=True).as_matrix() @ Rotation.from_euler('y', y_angle, degrees=True).as_matrix()
        camera_translation = np.eye(4)
        camera_translation[:3, 3] = np.array([0, 0, 1.75])
        camera_pose = np.linalg.inv(camera_translation @ camera_rot)
        noise = np.array([random.random(), random.random(), random.random()]) * 0.12 * 0
        positions.append(camera_pose[:3, 3] + noise)
    Path("camera_distribution_sdf.obj").write_text("\n".join([f"v {p[0]} {p[1]} {p[2]}" for p in positions]))


if __name__ == '__main__':
    test_camera_distribution()
