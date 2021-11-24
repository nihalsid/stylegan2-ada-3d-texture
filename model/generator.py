import torch
import numpy as np
from model import activation_funcs, FullyConnectedLayer, clamp_gain, normalize_2nd_moment, identity
from model.graph import create_faceconv_input, SmoothUpsample, modulated_face_conv


class Generator(torch.nn.Module):

    def __init__(self, z_dim, w_dim, w_num_layers, num_faces, color_channels):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_faces = num_faces
        self.color_channels = color_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, num_faces=num_faces, color_channels=color_channels)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, w_dim=w_dim, num_ws=self.num_ws, num_layers=w_num_layers)

    def forward(self, graph_data, z, truncation_psi=1, truncation_cutoff=None, noise_mode='random'):
        ws = self.mapping(z, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(graph_data, ws, noise_mode)
        return img


class SynthesisNetwork(torch.nn.Module):

    def __init__(self, w_dim, num_faces, color_channels, channel_base=16384, channel_max=512):
        super().__init__()
        self.num_ws = 10
        self.w_dim = w_dim
        self.num_faces = num_faces
        self.color_channels = color_channels
        block_pow_2 = [2 ** i for i in range(2, len(self.num_faces) + 2)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in block_pow_2}
        self.blocks = torch.nn.ModuleList()
        block_level = len(block_pow_2) - 1
        self.first_block = SynthesisPrologue(channels_dict[block_pow_2[0]], w_dim=w_dim, num_face=num_faces[block_level], color_channels=color_channels)
        for cdict_key in block_pow_2[1:]:
            block_level -= 1
            in_channels = channels_dict[cdict_key // 2]
            out_channels = channels_dict[cdict_key]
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim,
                                   num_face=num_faces[block_level],
                                   color_channels=color_channels)
            self.blocks.append(block)

    def forward(self, graph_data, ws, noise_mode='random'):
        split_ws = [ws[:, 0:2, :], ws[:, 1:4, :], ws[:, 3:6, :], ws[:, 5:8, :], ws[:, 7:10, :]]
        neighborhoods = [graph_data['face_neighborhood']] + graph_data['sub_neighborhoods']
        appended_pool_maps = [None] + graph_data['pool_maps']
        block_level = len(self.num_faces) - 1
        x, face_colors = self.first_block(neighborhoods[block_level],
                                          graph_data['is_pad'][block_level],
                                          graph_data['pads'][block_level],
                                          appended_pool_maps[block_level],
                                          split_ws[0], noise_mode)
        for i in range(len(self.num_faces) - 1):
            block_level -= 1
            sub_neighborhoods, is_pad, pads, pool_maps = [neighborhoods[block_level + 1], neighborhoods[block_level]], \
                                                         [graph_data['is_pad'][block_level + 1], graph_data['is_pad'][block_level]], \
                                                         [graph_data['pads'][block_level + 1], graph_data['pads'][block_level]], \
                                                         [appended_pool_maps[block_level + 1], appended_pool_maps[block_level]]
            x, face_colors = self.blocks[i](sub_neighborhoods, is_pad, pads, pool_maps, x, face_colors, split_ws[i + 1], noise_mode)

        return face_colors


class SynthesisPrologue(torch.nn.Module):

    def __init__(self, out_channels, w_dim, num_face, color_channels):
        super().__init__()
        self.w_dim = w_dim
        self.num_face = num_face
        self.color_channels = color_channels
        self.const = torch.nn.Parameter(torch.randn([num_face, out_channels]))
        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, num_face=num_face)
        self.torgb = ToRGBLayer(out_channels, color_channels, w_dim=w_dim, num_face=num_face)

    def forward(self, face_neighborhood, face_is_pad, pad_size, pool_map, ws, noise_mode):
        w_iter = iter(ws.unbind(dim=1))
        x = self.const.unsqueeze(0).repeat([ws.shape[0], 1, 1]).reshape((self.num_face * ws.shape[0], -1))

        x = self.conv1(x, [face_neighborhood], [face_is_pad], [pad_size], pool_map, next(w_iter), noise_mode=noise_mode)
        img = self.torgb(x, face_neighborhood, face_is_pad, pad_size, next(w_iter))
        return x, img


class SynthesisBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, num_face, color_channels):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.num_face = num_face
        self.color_channels = color_channels
        self.num_conv = 0
        self.num_torgb = 0
        self.resampler = SmoothUpsample()
        self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, num_face=num_face, resampler=self.resampler)
        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, num_face=num_face)
        self.torgb = ToRGBLayer(out_channels, color_channels, w_dim=w_dim, num_face=num_face)

    def forward(self, face_neighborhood, face_is_pad, pad_size, pool_map, x, img, ws, noise_mode):
        w_iter = iter(ws.unbind(dim=1))

        x = self.conv0(x, face_neighborhood, face_is_pad, pad_size, pool_map[0], next(w_iter), noise_mode=noise_mode)
        x = self.conv1(x, face_neighborhood[1:], face_is_pad[1:], pad_size[1:], pool_map[1], next(w_iter), noise_mode=noise_mode)

        y = self.torgb(x, face_neighborhood[1], face_is_pad[1], pad_size[1], next(w_iter))
        img = self.resampler(img, face_neighborhood[1], face_is_pad[1], pad_size[1], pool_map[0])
        img = img.add_(y)

        return x, img


class ToRGBLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, num_face, kernel_size=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_face = num_face
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, 1, kernel_size ** 2]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, face_neighborhood, face_is_pad, pad_size, w):
        styles = self.affine(w) * self.weight_gain
        x = create_faceconv_input(x, self.kernel_size ** 2, face_neighborhood, face_is_pad, pad_size)
        x = modulated_face_conv(x=x, weight=self.weight, styles=styles, demodulate=False)
        return torch.clamp(x + self.bias[None, :], -256, 256)


class SynthesisLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, num_face, kernel_size=3, resampler=None, activation='lrelu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_face = num_face
        self.resampler = resampler
        self.activation = activation_funcs[activation]['fn']
        self.activation_gain = activation_funcs[activation]['def_gain']
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, 1,  kernel_size ** 2]))

        self.register_buffer('noise_const', torch.randn([num_face, ]))
        self.noise_strength = torch.nn.Parameter(torch.zeros([1]))

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, face_neighborhood, face_is_pad, pad_size, pool_map, w, noise_mode, gain=1):
        styles = self.affine(w)

        noise = None
        if noise_mode == 'random':
            noise = torch.randn([styles.shape[0] * self.num_face, 1], device=x.device) * self.noise_strength
        elif noise_mode == 'const':
            noise = self.noise_const.unsqueeze(0).repeat([styles.shape[0], 1]).reshape([styles.shape[0] * self.num_face, 1]) * self.noise_strength

        x = create_faceconv_input(x, self.kernel_size ** 2, face_neighborhood[0], face_is_pad[0], pad_size[0])
        x = modulated_face_conv(x=x, weight=self.weight, styles=styles)
        if self.resampler is not None:
            x = self.resampler(x, face_neighborhood[1], face_is_pad[1], pad_size[1], pool_map)
        x = x + noise

        return clamp_gain(self.activation(x + self.bias[None, :]), self.activation_gain * gain, 256 * gain)


class MappingNetwork(torch.nn.Module):

    def __init__(self, z_dim, w_dim, num_ws, num_layers=8, activation='lrelu', lr_multiplier=0.01, w_avg_beta=0.995):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        features_list = [z_dim] + [w_dim] * num_layers

        self.layers = torch.nn.ModuleList()
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            self.layers.append(FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier))

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = normalize_2nd_moment(z)

        # Main layers.
        for idx in range(self.num_layers):
            x = self.layers[idx](x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            if self.num_ws is None or truncation_cutoff is None:
                x = self.w_avg.lerp(x, truncation_psi)
            else:
                x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)

        return x
