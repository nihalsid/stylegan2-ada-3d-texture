import time
import torch
from tqdm import tqdm
import hydra

from dataset import GraphDataLoader, to_device
from dataset.mesh_real import FaceGraphMeshDataset
from model import modulated_conv2d
from model.graph_generator import Generator
from model.graph import modulated_face_conv, GraphEncoder
from util.misc import print_module_summary, print_model_parameter_count


@hydra.main(config_path='../config', config_name='stylegan2')
def test_generator(config):
    batch_size = 16
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=batch_size, num_workers=0)
    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3).cuda()

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
        print_module_summary(G.synthesis, [batch['graph_data'], w])
        break


@hydra.main(config_path='../config', config_name='stylegan2')
def test_generator_u(config):
    from model.graph_generator_u import Generator
    batch_size = 16
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=batch_size, num_workers=0)
    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3).cuda()
    E = GraphEncoder(3).cuda()
    print_model_parameter_count(E)
    normals = torch.randn(6144 * batch_size, 3).cuda()
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # sanity test forward pass
        batch = to_device(batch, torch.device("cuda:0"))
        shape = E(normals, batch['graph_data'])
        z = torch.randn(batch_size, 512).to(torch.device("cuda:0"))
        w = G.mapping(z)
        t0 = time.time()
        fake = G.synthesis(batch['graph_data'], w, shape)
        print('Time for fake:', time.time() - t0, ', shape:', fake.shape)
        # sanity test backwards
        loss = torch.abs(fake - torch.rand_like(fake)).mean()
        t0 = time.time()
        loss.backward()
        print('Time for backwards:', time.time() - t0)
        print('backwards done')
        print_module_summary(G.synthesis, [batch['graph_data'], w, shape])
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
    test_generator_u()
