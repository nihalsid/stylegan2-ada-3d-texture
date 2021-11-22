import torch
import numpy as np
from model import activation_funcs, FullyConnectedLayer, clamp_gain, modulated_conv2d, SmoothUpsample, normalize_2nd_moment, identity


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

    def forward(self, z, truncation_psi=1, truncation_cutoff=None, noise_mode='random'):
        ws = self.mapping(z, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, noise_mode)
        return img


class SynthesisNetwork(torch.nn.Module):

    def __init__(self, w_dim, num_faces, color_channels, channel_base=16384, channel_max=512):
        super().__init__()
        self.num_ws = 10
        self.w_dim = w_dim
        self.num_faces = num_faces
        self.color_channels = color_channels
        block_pow_2 = [2 ** i for i in range(2, len(self.num_faces) + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in block_pow_2}
        self.blocks = torch.nn.ModuleList()
        self.first_block = SynthesisPrologue(channels_dict[block_pow_2[0]], w_dim=w_dim, num_face=num_faces[0], color_channels=color_channels)
        for cidx, cdict_key in enumerate(block_pow_2[1:]):
            in_channels = channels_dict[cdict_key // 2]
            out_channels = channels_dict[cdict_key]
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, num_face=num_faces[cidx], color_channels=color_channels)
            self.blocks.append(block)

    def forward(self, ws, noise_mode='random'):
        split_ws = [ws[:, 0:2, :], ws[:, 1:4, :], ws[:, 3:6, :], ws[:, 5:8, :], ws[:, 7:10, :]]
        x, img = self.first_block(split_ws[0], noise_mode)
        for i in range(len(self.num_faces) - 1):
            x, img = self.blocks[i](x, img, split_ws[i + 1], noise_mode)
        return img


class SynthesisPrologue(torch.nn.Module):

    def __init__(self, out_channels, w_dim, num_face, color_channels):
        super().__init__()
        self.w_dim = w_dim
        self.num_face = num_face
        self.color_channels = color_channels
        self.const = torch.nn.Parameter(torch.randn([out_channels, num_face]))
        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, num_face=num_face)
        self.torgb = ToRGBLayer(out_channels, color_channels, w_dim=w_dim)

    def forward(self, ws, noise_mode):
        w_iter = iter(ws.unbind(dim=1))
        x = self.const.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        x = self.conv1(x, next(w_iter), noise_mode=noise_mode)
        img = self.torgb(x, next(w_iter))
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
        self.torgb = ToRGBLayer(out_channels, color_channels, w_dim=w_dim)

    def forward(self, x, img, ws, noise_mode):
        w_iter = iter(ws.unbind(dim=1))

        x = self.conv0(x, next(w_iter), noise_mode=noise_mode)
        x = self.conv1(x, next(w_iter), noise_mode=noise_mode)

        y = self.torgb(x, next(w_iter))
        img = self.resampler(img)
        img = img.add_(y)

        return x, img


class ToRGBLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1):
        super().__init__()
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False)
        return torch.clamp(x + self.bias[None, :, None, None], -256, 256)


class SynthesisLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, num_face, kernel_size=3, resampler=identity, activation='lrelu'):
        super().__init__()
        self.num_face = num_face
        self.resampler = resampler
        self.activation = activation_funcs[activation]['fn']
        self.activation_gain = activation_funcs[activation]['def_gain']
        self.padding = kernel_size // 2
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))

        self.register_buffer('noise_const', torch.randn([num_face,]))
        self.noise_strength = torch.nn.Parameter(torch.zeros([1]))

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode, gain=1):
        styles = self.affine(w)

        noise = None
        if noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.num_face], device=x.device) * self.noise_strength
        if noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, padding=self.padding)
        x = self.resampler(x)
        x = x + noise

        return clamp_gain(self.activation(x + self.bias[None, :, None, None]), self.activation_gain * gain, 256 * gain)


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


if __name__ == '__main__':
    from util.misc import print_model_parameter_count, print_module_summary

    model = Generator(512, 512, 2, 64, 3)
    print_module_summary(model, (torch.randn((16, 512)), ))
    print_model_parameter_count(model)
