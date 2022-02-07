"""
Contains code for logging approximate FID scores during training.
If you want to output ground-truth images from the training dataset, you can
run this file as a script.
"""

import os
import shutil
import torch
import copy
import gc
import argparse

from torchvision.utils import save_image
from pytorch_fid import fid_score
from tqdm import tqdm

from dataset import GraphDataLoader, to_device
from dataset.mesh_real_pigan import SDFGridDataset
from util.misc import EasyDict


def output_real_images(dataloader, num_imgs, real_dir):
    img_counter = 0
    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
    for i in range(num_imgs//batch_size):
        real_imgs = next(dataloader)['real']
        for img in real_imgs:
            save_image(img, os.path.join(real_dir, f'{img_counter:0>5}.jpg'), normalize=True)
            img_counter += 1


def setup_evaluation(config, generated_dir, target_size=128, num_imgs=5000):
    # Only make real images if they haven't been made yet
    real_dir = os.path.join('runs/EvalImages', '_real_images_' + str(target_size))
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        dataloader = GraphDataLoader(SDFGridDataset(EasyDict(config)))
        print('outputting real images...')
        output_real_images(dataloader, num_imgs, real_dir)
        print('...done')

    if generated_dir is not None:
        os.makedirs(generated_dir, exist_ok=True)
    return real_dir


def output_images(render, generator, encoder, dataloader, rank, output_dir, device, num_imgs=2048):
    generator.eval()
    img_counter = rank

    if rank == 0: pbar = tqdm("generating images", total = num_imgs)
    with torch.no_grad():
        while img_counter < num_imgs:
            torch.cuda.empty_cache()
            gc.collect()
            print('=====')
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
                except:
                    pass
            batch = next(dataloader)
            batch = to_device(batch, device)
            z = torch.randn((batch['real'].shape[0], generator.z_dim), device=device)
            faces = batch['faces']
            shape = encoder(batch['sdf_x'])[4].mean((2, 3, 4))
            generated_imgs = render(generator(faces, z, shape), batch, 128).cpu()

            for img in generated_imgs:
                save_image(img, os.path.join(output_dir, f'{img_counter:0>5}.jpg'), normalize=True)
                if rank == 0: pbar.update(1)
    if rank == 0: pbar.close()


def calculate_fid(dataset_name, generated_dir, target_size=256):
    real_dir = os.path.join('runs/EvalImages', '_real_images_' + str(target_size))
    fid = fid_score.calculate_fid_given_paths([real_dir, generated_dir], 128, 'cuda', 2048)
    torch.cuda.empty_cache()

    return fid


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CelebA')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--num_imgs', type=int, default=8000)

    opt = parser.parse_args()

    real_images_dir = setup_evaluation(opt.dataset, None, target_size=opt.img_size, num_imgs=opt.num_imgs)