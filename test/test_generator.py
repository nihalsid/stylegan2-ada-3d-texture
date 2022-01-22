import time
import torch
from tqdm import tqdm
import hydra

from dataset import GraphDataLoader, to_device
from dataset.mesh_real_features import FaceGraphMeshDataset
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
    batch_size = 4
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=batch_size, num_workers=0)
    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, channel_base=config.g_channel_base, channel_max=config.g_channel_max).cuda()
    E = GraphEncoder(dataset.num_feats).cuda()
    print_model_parameter_count(G)
    print_model_parameter_count(E)
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # sanity test forward pass
        batch = to_device(batch, torch.device("cuda:0"))
        shape = E(batch['x'], batch['graph_data'])
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


@hydra.main(config_path='../config', config_name='stylegan2')
def test_generator_u2(config):
    from model.graph_generator_u2 import Generator
    from dataset.mesh_real_2features import FaceGraphMeshDataset
    batch_size = 2
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=batch_size, num_workers=0)
    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, channel_base=config.g_channel_base, channel_max=config.g_channel_max).cuda()
    layer_dims = (32, 64, 64, 96, 128, 128, 192, 192)
    E = GraphEncoder(dataset.num_feats_0, layer_dims=layer_dims).cuda()
    S = GraphEncoder(dataset.num_feats_1, layer_dims=layer_dims).cuda()
    print_model_parameter_count(G)
    print_model_parameter_count(E)
    print_model_parameter_count(S)
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # sanity test forward pass
        batch = to_device(batch, torch.device("cuda:0"))
        shape = E(batch['x_0'], batch['graph_data'])
        semantics = S(batch['x_1'], batch['graph_data'])
        z = torch.randn(batch_size, 512).to(torch.device("cuda:0"))
        w = G.mapping(z)
        t0 = time.time()
        fake = G.synthesis(batch['graph_data'], w, shape, semantics)
        print('Time for fake:', time.time() - t0, ', shape:', fake.shape)
        # sanity test backwards
        loss = torch.abs(fake - torch.rand_like(fake)).mean()
        t0 = time.time()
        loss.backward()
        print('Time for backwards:', time.time() - t0)
        print('backwards done')
        print_module_summary(G.synthesis, [batch['graph_data'], w, shape, semantics])
        break


@hydra.main(config_path='../config', config_name='stylegan2')
def test_generator_u_spool(config):
    from model.graph_generator_u_spool import Generator
    from dataset.mesh_real_features_patch_spool import FaceGraphMeshDataset
    batch_size = 4
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=batch_size, num_workers=0)
    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, 7, channel_base=config.g_channel_base, channel_max=config.g_channel_max).cuda()
    E = GraphEncoder(dataset.num_feats).cuda()
    print_model_parameter_count(G)
    print_model_parameter_count(E)
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # sanity test forward pass
        batch = to_device(batch, torch.device("cuda:0"))
        shape = E(batch['x'], batch['graph_data'])
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


@hydra.main(config_path='../config', config_name='stylegan2')
def test_spade_block(config):
    from model.graph_generator_u_spade import SPADEResnetBlock
    from dataset.mesh_real_features_patch_spool import FaceGraphMeshDataset
    batch_size = 4
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=batch_size, num_workers=0)
    model = SPADEResnetBlock(3, 128, 7).cuda()
    print_model_parameter_count(model)
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, torch.device("cuda:0"))
        print(model(batch['x'], batch['graph_data']['face_neighborhood'], batch['graph_data']['is_pad'][0],
              batch['graph_data']['pads'][0], batch['graph_data']['semantic_maps'][0]).shape)
        break


@hydra.main(config_path='../config', config_name='stylegan2')
def test_texture_conv(config):
    from model.graph import TextureConv
    from dataset.mesh_real_features import FaceGraphMeshDataset
    batch_size = 4
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=batch_size, num_workers=0)
    model = TextureConv(3, 128, 'max').cuda()
    print_model_parameter_count(model)
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, torch.device("cuda:0"))
        print(model(batch['x'], batch['graph_data']['face_neighborhood'], batch['graph_data']['is_pad'][0], batch['graph_data']['pads'][0]).shape)
        break


@hydra.main(config_path='../config', config_name='stylegan2')
def test_generator_u_spade(config):
    from model.graph_generator_u_spade import Generator
    from dataset.mesh_real_features_patch_spool import FaceGraphMeshDataset
    batch_size = 4
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=batch_size, num_workers=0)
    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, 7, channel_base=config.g_channel_base, channel_max=config.g_channel_max).cuda()
    E = GraphEncoder(dataset.num_feats).cuda()
    print_model_parameter_count(G)
    print_model_parameter_count(E)
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # sanity test forward pass
        batch = to_device(batch, torch.device("cuda:0"))
        shape = E(batch['x'], batch['graph_data'])
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


@hydra.main(config_path='../config', config_name='stylegan2')
def test_generator_u_textureconv(config):
    from model.graph_generator_u_texconv import Generator
    batch_size = 4
    dataset = FaceGraphMeshDataset(config)
    dataloader = GraphDataLoader(dataset, batch_size=batch_size, num_workers=0)
    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, channel_base=config.g_channel_base, channel_max=config.g_channel_max, aggregation_function='max').cuda()
    E = GraphEncoder(dataset.num_feats).cuda()
    print_model_parameter_count(G)
    print_model_parameter_count(E)
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # sanity test forward pass
        batch = to_device(batch, torch.device("cuda:0"))
        shape = E(batch['x'], batch['graph_data'])
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


if __name__ == '__main__':
    test_generator_u()
