import torch
import numpy as np
from pathlib import Path
import marching_cubes as mc
import trimesh
import hydra


class DistanceFieldDataset(torch.utils.data.Dataset):
    
    def __init__(self, config, limit=None, overfit=False):
        self.items = [x / "032.npy" for x in Path(config.df_path).iterdir()][:limit]
        if overfit:
            self.items = self.items * 160
        self.trunc = config.df_trunc
        self.vox_size = config.df_size
        self.mean = config.df_mean
        self.std = config.df_std
        self.weight_occupied = 8

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        selected_item = self.items[idx]
        return {
            'df': torch.from_numpy(np.load(str(selected_item))).unsqueeze(0),
            'name': selected_item.parent.name.split('.')[0]
        }

    def augment_batch_data(self, batch):
        weights = torch.ones_like(batch['df']) * (1 + (batch['df'] < self.trunc).float() * (self.weight_occupied - 1)).float()
        empty = (batch['df'] >= self.trunc)
        batch['weights'] = weights
        batch['empty'] = empty

    def normalize(self, df):
        return (df - self.mean) / self.std

    def visualize_as_mesh(self, grid, output_path):
        vertices, triangles = mc.marching_cubes(grid, self.vox_size)
        mc.export_obj(vertices, triangles, output_path)

    def visualize_sdf_as_voxels(self, sdf, output_path):
        from util.misc import to_point_list
        point_list = to_point_list(sdf <= self.vox_size)
        if point_list.shape[0] > 0:
            base_mesh = trimesh.voxel.ops.multibox(centers=point_list, pitch=1)
            base_mesh.export(output_path)

    @staticmethod
    def visualize_occupancy_as_voxels(occupancy, output_path):
        from util.misc import to_point_list
        point_list = to_point_list(occupancy)
        if point_list.shape[0] > 0:
            base_mesh = trimesh.voxel.ops.multibox(centers=point_list, pitch=1)
            base_mesh.export(output_path)

    @staticmethod
    def visualize_float_grid(grid, ignore_val, minval, maxval, output_path):
        from matplotlib import cm
        jetmap = cm.get_cmap('jet')
        norm_grid = (grid - minval) / (maxval - minval)
        f = open(output_path, "w")
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                for z in range(grid.shape[2]):
                    if grid[x, y, z] > ignore_val:
                        c = (np.array(jetmap(norm_grid[x, y, z])) * 255).astype(np.uint8)
                        f.write('v %f %f %f %f %f %f\n' % (x + 0.5, y + 0.5, z + 0.5, c[0], c[1], c[2]))
        f.close()


@hydra.main(config_path='../config', config_name='stylegan2')
def test_distancefield_dataset(config):
    from tqdm import tqdm
    from dataset import to_device
    dataset = DistanceFieldDataset(config=config, limit=4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, torch.device("cuda:0"))
        dataset.augment_batch_data(batch)
        print(batch['name'], batch['df'].shape, batch['df'].max(), batch['df'].min())
        for idx in range(batch['df'].shape[0]):
            dataset.visualize_as_mesh(batch['df'][idx].squeeze(0).cpu().numpy(), f"{batch['name'][idx]}.obj")
            dataset.visualize_occupancy_as_voxels(1 - batch["empty"][idx].squeeze(0).cpu().numpy(), f"{batch['name'][idx]}_nempty.obj")
            dataset.visualize_float_grid(batch['weights'][idx].squeeze(0).cpu().numpy(), 0.0, 0.0, 2.0, f"{batch['name'][idx]}_weight.obj")


@hydra.main(config_path='../config', config_name='stylegan2')
def get_dataset_mean_std(config):
    from tqdm import tqdm
    import math
    dataset = DistanceFieldDataset(config=config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    ctr, mean, var = 0., 0., 0.
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        mean += batch['df'].mean().item()
        var += batch['df'].std().item() ** 2
        ctr += 1
    print(mean / ctr, math.sqrt((var / ctr)))


if __name__ == "__main__":
    get_dataset_mean_std()
