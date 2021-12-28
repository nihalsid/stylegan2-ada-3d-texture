import torch
import torch_geometric
import torch_scatter
import math
from torch.nn import init


def pool(x, node_count, pool_map, pool_op='max'):
    if pool_op == 'max':
        x_pooled = torch.ones((node_count, x.shape[1]), dtype=x.dtype).to(x.device) * (x.min().detach() - 1e-3)
        torch_scatter.scatter_max(x, pool_map, dim=0, out=x_pooled)
    elif pool_op == 'mean':
        x_pooled = torch.zeros((node_count, x.shape[1]), dtype=x.dtype).to(x.device)
        torch_scatter.scatter_mean(x, pool_map, dim=0, out=x_pooled)
    else:
        raise NotImplementedError
    return x_pooled


def unpool(x, pool_map):
    x_unpooled = x[pool_map, :]
    return x_unpooled


def create_faceconv_input(x, kernel_size, face_neighborhood, face_is_pad, pad_size):
    padded_x = torch.zeros((pad_size, x.shape[1]), dtype=x.dtype, device=x.device)
    padded_x[torch.logical_not(face_is_pad), :] = x
    f_ = [padded_x[face_neighborhood[:, i]] for i in range(kernel_size)]
    conv_input = torch.cat([f.unsqueeze(-1).unsqueeze(-1) for f in f_], dim=3)
    return conv_input


def modulated_face_conv(x, weight, styles, demodulate=True):
    batch_size = styles.shape[0]
    num_faces = x.shape[0] // batch_size
    out_channels, in_channels, kh, kw = weight.shape

    w = weight.unsqueeze(0)
    w = w * styles.reshape(batch_size, 1, -1, 1, 1)
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)

    x = x.reshape(batch_size, num_faces, *x.shape[1:]).permute((1, 0, 2, 3, 4)).reshape(num_faces, batch_size * in_channels, 1, weight.shape[-1])
    w = w.reshape(-1, in_channels, kh, kw)
    x = torch.nn.functional.conv2d(x, w, groups=batch_size)
    x = x.reshape(num_faces, batch_size * out_channels).reshape(num_faces, batch_size, out_channels).permute((1, 0, 2)).reshape((batch_size * num_faces, out_channels))
    return x


class SmoothUpsample(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.blur = Blur()

    def forward(self, x, face_neighborhood, face_is_pad, pad_size, pool_map):
        x = unpool(x, pool_map)
        x = self.blur(x, face_neighborhood, face_is_pad, pad_size)
        return x


class Blur(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.ones((1, 1, 1, 9)).float())
        self.register_buffer("blur_filter", torch.tensor([1/4, 1/8, 1/16, 1/8, 1/16, 1/8, 1/16, 1/8, 1/16]).float())
        self.weight[:, :, 0, :] = self.blur_filter

    def forward(self, x, face_neighborhood, face_is_pad, pad_size):
        padded_x = torch.zeros((pad_size, x.shape[1]), dtype=x.dtype, device=x.device)
        padded_x[torch.logical_not(face_is_pad), :] = x
        f_ = [padded_x[face_neighborhood[:, i]] for i in range(9)]
        conv_input = torch.cat([f.unsqueeze(-1).unsqueeze(-1) for f in f_], dim=3).reshape((-1, 1, 1, 9))
        correction_factor = ((1 - face_is_pad[face_neighborhood].float()) * self.blur_filter.unsqueeze(0).expand(face_neighborhood.shape[0], -1)).sum(-1)
        correction_factor = correction_factor.unsqueeze(1).expand(-1, x.shape[1])
        return torch.nn.functional.conv2d(conv_input, self.weight).squeeze(-1).squeeze(-1).reshape(x.shape[0], x.shape[1]) / correction_factor


class FaceConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.neighborhood_size = 8
        self.conv = torch.nn.Conv2d(in_channels, out_channels, (1, self.neighborhood_size + 1), padding=0)

    def forward(self, x, face_neighborhood, face_is_pad, pad_size):
        return self.conv(create_faceconv_input(x, self.neighborhood_size + 1, face_neighborhood, face_is_pad, pad_size)).squeeze(-1).squeeze(-1)


class SymmetricFaceConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_0 = torch.nn.Parameter(torch.empty((out_channels, in_channels, 1, 1)))
        self.weight_1 = torch.nn.Parameter(torch.empty((out_channels, in_channels, 1, 1)))
        self.weight_2 = torch.nn.Parameter(torch.empty((out_channels, in_channels, 1, 1)))
        self.neighborhood_size = 8
        self.bias = torch.nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_0, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_1, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_2, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(torch.empty((self.out_channels, self.in_channels, 1, 9)))
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def create_symmetric_conv_filter(self):
        return torch.cat([self.weight_0, self.weight_1, self.weight_2,
                          self.weight_1, self.weight_2, self.weight_1,
                          self.weight_2, self.weight_1, self.weight_2], dim=-1)

    def forward(self, x, face_neighborhood, face_is_pad, pad_size):
        conv_input = create_faceconv_input(x, self.neighborhood_size + 1, face_neighborhood, face_is_pad, pad_size)
        return torch.nn.functional.conv2d(conv_input, self.create_symmetric_conv_filter(), self.bias).squeeze(-1).squeeze(-1)


class GraphEncoder(torch.nn.Module):

    def __init__(self, in_channels, conv_layer=SymmetricFaceConv, norm=torch_geometric.nn.BatchNorm):

        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.enc_conv_in = torch.nn.Linear(in_channels, 32)
        self.down_0_block_0 = FResNetBlock(32, 64, conv_layer, norm, self.activation)
        self.down_0_block_1 = FResNetBlock(64, 64, conv_layer, norm, self.activation)
        self.down_1_block_0 = FResNetBlock(64, 128, conv_layer, norm, self.activation)
        self.down_2_block_0 = FResNetBlock(128, 128, conv_layer, norm, self.activation)
        self.down_3_block_0 = FResNetBlock(128, 128, conv_layer, norm, self.activation)
        self.down_4_block_0 = FResNetBlock(128, 256, conv_layer, norm, self.activation)
        self.enc_mid_block_0 = FResNetBlock(256, 256, conv_layer, norm, self.activation)
        self.enc_out_conv = conv_layer(256, 256)
        self.enc_out_norm = norm(256)

    def forward(self, x, graph_data):
        level_feats = []
        pool_ctr = 0
        x = self.enc_conv_in(x)
        x = self.down_0_block_0(x, graph_data['face_neighborhood'], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = self.down_0_block_1(x, graph_data['face_neighborhood'], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        level_feats.append(x)
        x = pool(x, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.down_1_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        level_feats.append(x)
        x = pool(x, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.down_2_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        level_feats.append(x)
        x = pool(x, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.down_3_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        level_feats.append(x)
        x = pool(x, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.down_4_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        level_feats.append(x)
        x = pool(x, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.enc_mid_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = self.enc_out_norm(x)
        x = self.activation(x)
        x = self.enc_out_conv(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])

        level_feats.append(x)

        return level_feats


class FResNetBlock(torch.nn.Module):

    def __init__(self, nf_in, nf_out, conv_layer, norm, activation):
        super().__init__()
        self.nf_in = nf_in
        self.nf_out = nf_out
        self.norm_0 = norm(nf_in)
        self.conv_0 = conv_layer(nf_in, nf_out)
        self.norm_1 = norm(nf_out)
        self.conv_1 = conv_layer(nf_out, nf_out)
        self.activation = activation
        if nf_in != nf_out:
            self.nin_shortcut = conv_layer(nf_in, nf_out)

    def forward(self, x, face_neighborhood, face_is_pad, num_pad):
        h = x
        h = self.norm_0(h)
        h = self.activation(h)
        h = self.conv_0(h, face_neighborhood, face_is_pad, num_pad)

        h = self.norm_1(h)
        h = self.activation(h)
        h = self.conv_1(h, face_neighborhood, face_is_pad, num_pad)

        if self.nf_in != self.nf_out:
            x = self.nin_shortcut(x, face_neighborhood, face_is_pad, num_pad)

        return x + h
