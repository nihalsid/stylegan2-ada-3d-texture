import torch
import numpy as np
from model.styleganvox import activation_funcs, FullyConnectedLayer, clamp_gain, modulated_conv3d, SmoothUpsample, normalize_2nd_moment, identity, SDFEncoder
from model.styleganvox_sparse import MinkowskiModulatedConvolution, SmoothUpsampleSparse
import MinkowskiEngine as ME


class Generator(torch.nn.Module):

    def __init__(self, z_dim, w_dim, w_num_layers, img_resolution, img_channels, synthesis_layer='stylegan2'):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, synthesis_layer=synthesis_layer)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, w_dim=w_dim, num_ws=self.num_ws, num_layers=w_num_layers)

    def forward(self, z, coord_064, coord_128, shape, truncation_psi=1, truncation_cutoff=None, noise_mode='random'):
        ws = self.mapping(z, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, coord_064, coord_128, shape, noise_mode)
        return img


class SynthesisNetwork(torch.nn.Module):

    def __init__(self, w_dim, img_resolution, img_channels, synthesis_layer='stylegan2'):
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        self.num_ws = 2 * (len(self.block_resolutions) + 1)
        channels_dict = {4: 512, 8: 512, 16: 512, 32: 256, 64: 128, 128: 128}
        channels_dict_geo = {4: 256, 8: 256, 16: 128, 32: 128, 64: 0, 128: 0}
        self.blocks = torch.nn.ModuleList()
        self.first_block = SynthesisPrologue(channels_dict[self.block_resolutions[0]], w_dim=w_dim, geo_channels=channels_dict_geo[4],
                                             resolution=self.block_resolutions[0], img_channels=img_channels,
                                             synthesis_layer=synthesis_layer)
        for res in self.block_resolutions[1:]:
            in_channels = channels_dict[res // 2]
            geo_channels = channels_dict_geo[res]
            out_channels = channels_dict[res]
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, geo_channels=geo_channels, resolution=res, img_channels=img_channels, synthesis_layer=synthesis_layer)
            self.blocks.append(block)

        self.final_block = SynthesisBlockSparse(channels_dict[64], channels_dict[128], w_dim=w_dim, img_channels=img_channels)

    def forward(self, ws, coord_064, coord_128, shape, noise_mode='random'):
        split_ws = [ws[:, 0:2, :]] + [ws[:, 2 * n + 1: 2 * n + 4, :] for n in range(len(self.block_resolutions))]
        x, img = self.first_block(split_ws[0], noise_mode)
        for i in range(len(self.block_resolutions) - 1):
            x, img = self.blocks[i](x, img, split_ws[i + 1], shape[len(shape) - 1 - i], noise_mode)
        # x = Bx3x64x64x64
        # img = Bx3x64x64x64
        # convert to sparse
        x_feat = x[coord_064[:, 0], :, coord_064[:, 1], coord_064[:, 2], coord_064[:, 3]]
        img_feat = img[coord_064[:, 0], :, coord_064[:, 1], coord_064[:, 2], coord_064[:, 3]]
        x_feat, img_feat = self.final_block(x_feat, coord_064, coord_128, img_feat, ws)
        return img_feat.contiguous()


class SynthesisPrologue(torch.nn.Module):

    def __init__(self, out_channels, w_dim, geo_channels, resolution, img_channels, synthesis_layer):
        super().__init__()
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.img_channels = img_channels
        self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution, resolution]))
        self.conv1 = SynthesisLayer(out_channels, out_channels - geo_channels, w_dim=w_dim, resolution=resolution)
        self.torgb = ToRGBLayer(out_channels - geo_channels, img_channels, w_dim=w_dim)

    def forward(self, ws, noise_mode):
        w_iter = iter(ws.unbind(dim=1))
        x = self.const.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1, 1])
        x = self.conv1(x, next(w_iter), noise_mode=noise_mode)
        img = self.torgb(x, next(w_iter))
        return x, img


class SynthesisBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, geo_channels, resolution, img_channels, synthesis_layer):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.num_conv = 0
        self.num_torgb = 0
        self.resampler = SmoothUpsample()
        self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, resampler=self.resampler)
        self.conv1 = SynthesisLayer(out_channels, out_channels - geo_channels, w_dim=w_dim, resolution=resolution)
        self.torgb = ToRGBLayer(out_channels - geo_channels, img_channels, w_dim=w_dim)

    def forward(self, x, img, ws, shape, noise_mode):
        w_iter = iter(ws.unbind(dim=1))

        x = torch.cat([x, shape], dim=1)
        x = self.conv0(x, next(w_iter), noise_mode=noise_mode)
        x = self.conv1(x, next(w_iter), noise_mode=noise_mode)

        y = self.torgb(x, next(w_iter))
        img = self.resampler(img)
        img = img.add_(y)

        return x, img


class SynthesisBlockSparse(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, img_channels):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.img_channels = img_channels
        self.num_conv = 0
        self.num_torgb = 0
        self.resampler = SmoothUpsampleSparse()
        self.conv0 = SynthesisLayerSparse(in_channels, out_channels, w_dim=w_dim, resampler=self.resampler)
        self.conv1 = SynthesisLayerSparse(out_channels, out_channels, w_dim=w_dim)
        self.torgb = ToRGBLayerSparse(out_channels, img_channels, w_dim=w_dim)

    def forward(self, x_feat, x_coord_0, x_coord_1, img_feat, ws):
        w_iter = iter(ws.unbind(dim=1))

        x_feat = self.conv0(x_feat, x_coord_0, x_coord_1, next(w_iter))
        x_feat = self.conv1(x_feat, x_coord_1, None, next(w_iter))

        y_feat = self.torgb(x_feat, x_coord_1, next(w_iter))
        img_feat = self.resampler(ME.SparseTensor(img_feat, x_coord_0.int()), x_coord_1)
        img_feat = img_feat.add_(y_feat)

        return x_feat, img_feat


class ToRGBLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1):
        super().__init__()
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size, kernel_size]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 3))

    def forward(self, x, w):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv3d(x=x, weight=self.weight, styles=styles, demodulate=False)
        return torch.clamp(x + self.bias[None, :, None, None, None], -256, 256)


class SynthesisLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, resolution, kernel_size=3, resampler=identity, activation='lrelu'):
        super().__init__()
        self.resolution = resolution
        self.resampler = resampler
        self.activation = activation_funcs[activation]['fn']
        self.activation_gain = activation_funcs[activation]['def_gain']
        self.padding = kernel_size // 2
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size, kernel_size]))

        self.register_buffer('noise_const', torch.randn([resolution, resolution, resolution]))
        self.noise_strength = torch.nn.Parameter(torch.zeros([1]))

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode, gain=1):
        styles = self.affine(w)

        noise = None
        if noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        x = modulated_conv3d(x=x, weight=self.weight, styles=styles, padding=self.padding)
        x = self.resampler(x)
        x = x + noise

        return clamp_gain(self.activation(x + self.bias[None, :, None, None, None]), self.activation_gain * gain, 256 * gain)


class ToRGBLayerSparse(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1):
        super().__init__()
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.mod_conv3d = MinkowskiModulatedConvolution(in_channels, out_channels, kernel_size=1, bias=True)
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 3))

    def forward(self, x_feat, x_coord, w):
        styles = self.affine(w) * self.weight_gain
        x_feat = self.mod_conv3d(x_feat, x_coord, styles, False)
        x_feat = torch.clamp(x_feat, -256, 256)
        return x_feat


class SynthesisLayerSparse(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, kernel_size=3, resampler=None, activation='lrelu'):
        super().__init__()
        self.resampler = resampler
        self.activation = activation_funcs[activation]['fn']
        self.activation_gain = activation_funcs[activation]['def_gain']
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.mod_conv3d = MinkowskiModulatedConvolution(in_channels, out_channels, kernel_size=kernel_size, bias=True)
        self.noise_strength = torch.nn.Parameter(torch.zeros([1]))

    def forward(self, x_feat, x_coord_0, x_coord_1, w, gain=1):
        styles = self.affine(w)
        x_feat = self.mod_conv3d(x_feat, x_coord_0, styles)
        if self.resampler is not None:
            x_feat = self.resampler(ME.SparseTensor(x_feat, x_coord_0.int()), x_coord_1)
        noise = torch.randn_like(x_feat) * self.noise_strength
        x_feat = x_feat + noise
        return clamp_gain(self.activation(x_feat), self.activation_gain * gain, 256 * gain)


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


def test_generator():
    import time
    batch_size = 2
    G = Generator(512, 512, 2, 64, 3).cuda()
    E = SDFEncoder(1).cuda()
    print_model_parameter_count(G)
    print_model_parameter_count(E)
    for batch_idx in range(16):
        # sanity test forward pass
        z = torch.randn(batch_size, 512).to(torch.device("cuda:0"))
        x = torch.randn(batch_size, 1, 64, 64, 64).to(torch.device("cuda:0"))
        shape = E(x)
        w = G.mapping(z)
        t0 = time.time()
        fake = G.synthesis(w, shape)
        print('Time for fake:', time.time() - t0, ', shape:', fake.shape)
        # sanity test backwards
        loss = torch.abs(fake - torch.rand_like(fake)).mean()
        t0 = time.time()
        loss.backward()
        print('Time for backwards:', time.time() - t0)
        print('backwards done')
        break


if __name__ == '__main__':
    from util.misc import print_model_parameter_count, print_module_summary

    # model = Generator(512, 512, 2, 16, 3).cuda()
    # print_module_summary(model, (torch.randn((16, 512)).cuda(), ))
    # print_model_parameter_count(model)

    test_generator()
