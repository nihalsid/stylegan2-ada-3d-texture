import time
import torch
from tqdm import tqdm
import hydra
from dataset.mesh import to_device, GraphDataLoader, FaceGraphMeshDataset
from model import modulated_conv2d
from model.generator import Generator
from model.graph import modulated_face_conv


@hydra.main(config_path='../config', config_name='stylegan2')
def test_generator(config):
    batch_size = 16
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=batch_size, num_workers=0)
    G = Generator(512, 512, 2, [6144, 1536, 384, 96, 24], 3).cuda()
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # sanity test forward pass
        batch = to_device(batch, torch.device("cuda:0"))
        z = torch.randn(batch_size, 512).to(torch.device("cuda:0"))
        w = G.mapping(z)
        t0 = time.time()
        fake = G.synthesis(batch['graph_data'], w)
        print('Time for fake:', time.time() - t0, ', shape:', fake.shape)
        # sanity test backwards
        loss = torch.abs(fake - torch.rand_like(fake)).mean()
        t0 = time.time()
        loss.backward()
        print('Time for backwards:', time.time() - t0)
        print('backwards done')
        break


def test_modulated_face_conv():
    x_2d = torch.randn(4, 3, 16, 16)
    w_2d = torch.rand(8, 3, 3, 3)
    s = torch.rand(4, 3)
    x_2d_patches = x_2d.unfold(2, 3, 1).unfold(3, 3, 1).reshape(4, 3, 14, 14, 9).permute(0, 2, 3, 1, 4).reshape(4 * 14 * 14, 3, 9).unsqueeze(-2)
    a = modulated_conv2d(x_2d, w_2d, s).permute((0, 2, 3, 1)).reshape(4 * 14 * 14, 8)
    b = modulated_face_conv(x_2d_patches, w_2d.reshape(8, 3, 9, 1).permute((0, 1, 3, 2)), s)
    print(torch.abs(a - b).mean(), a.mean(), a.std(), b.mean(), b.std())


if __name__ == '__main__':
    test_generator()
