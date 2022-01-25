import torch
import numpy as np
from model.stylegan2 import activation_funcs, FullyConnectedLayer, clamp_gain, modulated_conv2d, SmoothUpsample, normalize_2nd_moment, identity


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

    def forward(self, z, silhoutte, truncation_psi=1, truncation_cutoff=None, noise_mode='random'):
        ws = self.mapping(z, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        batch_size, ws_shape_1, ws_shape_2 = ws.shape
        ws = ws.unsqueeze(1).expand(-1, 6, -1, -1).reshape(batch_size * 6, ws_shape_1, ws_shape_2)
        img = self.synthesis(ws, silhoutte, noise_mode)
        return img


class SynthesisNetwork(torch.nn.Module):

    def __init__(self, w_dim, img_resolution, img_channels, channel_base=16384, channel_max=512, synthesis_layer='stylegan2'):
        super().__init__()

        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        self.num_ws = 2 * (len(self.block_resolutions) + 1)
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        self.blocks = torch.nn.ModuleList()
        self.first_block = SynthesisPrologue(channels_dict[self.block_resolutions[0]], w_dim=w_dim,
                                             resolution=self.block_resolutions[0], img_channels=img_channels,
                                             synthesis_layer=synthesis_layer)
        for res in self.block_resolutions[1:]:
            in_channels = channels_dict[res // 2]
            out_channels = channels_dict[res]
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res, img_channels=img_channels, synthesis_layer=synthesis_layer)
            self.blocks.append(block)

    def forward(self, ws, silhoutte, noise_mode='random'):
        split_ws = [ws[:, 0:2, :]] + [ws[:, 2 * n + 1: 2 * n + 4, :] for n in range(len(self.block_resolutions))]
        x, img = self.first_block(split_ws[0], silhoutte, noise_mode)
        for i in range(len(self.block_resolutions) - 1):
            x, img = self.blocks[i](x, img, split_ws[i + 1], silhoutte, noise_mode)
        return img.reshape((-1, 6, self.img_channels, self.img_resolution, self.img_resolution))


class SynthesisPrologue(torch.nn.Module):

    def __init__(self, out_channels, w_dim, resolution, img_channels, synthesis_layer):
        super().__init__()
        SynthesisLayer = SynthesisLayer2 if synthesis_layer == 'stylegan2' else SynthesisLayer1
        ToRGBLayer = ToRGBLayer2 if synthesis_layer == 'stylegan2' else ToRGBLayer1
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.num_atlas = 6
        self.out_channels = out_channels
        self.const = torch.nn.Parameter(torch.randn([self.num_atlas, out_channels, resolution, resolution]))
        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution)
        self.spade_1 = SPADE_IN(out_channels, 2)
        self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim)

    def forward(self, ws, silhoutte, noise_mode):
        w_iter = iter(ws.unbind(dim=1))
        x = self.const.unsqueeze(0).repeat([ws.shape[0] // 6, 1, 1, 1, 1]).reshape([ws.shape[0], self.out_channels, self.resolution, self.resolution])
        x = self.conv1(x, next(w_iter), noise_mode=noise_mode)
        x = self.spade_1(x, silhoutte)
        img = self.torgb(x, next(w_iter))
        return x, img


class SynthesisBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, resolution, img_channels, synthesis_layer):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.num_conv = 0
        self.num_torgb = 0
        self.resampler = SmoothUpsample()
        self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, resampler=self.resampler)
        # self.spade_0 = SPADE_IN(out_channels, 2)
        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution)
        # self.spade_1 = SPADE_IN(out_channels, 2)
        self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim)

    def forward(self, x, img, ws, silhoutte, noise_mode):
        w_iter = iter(ws.unbind(dim=1))

        x = self.conv0(x, next(w_iter), silhoutte, noise_mode=noise_mode)
        x = self.spade_0(x, silhoutte)
        x = self.conv1(x, next(w_iter), silhoutte, noise_mode=noise_mode)
        x = self.spade_1(x, silhoutte)

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

    def __init__(self, in_channels, out_channels, w_dim, resolution, kernel_size=3, resampler=identity, activation='lrelu'):
        super().__init__()
        self.resolution = resolution
        self.resampler = resampler
        self.activation = activation_funcs[activation]['fn']
        self.activation_gain = activation_funcs[activation]['def_gain']
        self.padding = kernel_size // 2
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))

        self.register_buffer('noise_const', torch.randn([resolution, resolution]))
        self.noise_strength = torch.nn.Parameter(torch.zeros([1]))

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode, gain=1):
        styles = self.affine(w)

        noise = None
        if noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
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


class AdaIN(torch.nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.norm = torch.nn.InstanceNorm2d(in_channels)

    def forward(self, x, style):
        style = style.unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(x)
        out = gamma * out + beta

        return out


class ConvDemodulated(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, 1, 1]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x):
        batch_size = x.shape[0]
        out_channels, in_channels, kh, kw = self.weight.shape

        w = self.weight.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]

        x = x.reshape(1, -1, *x.shape[2:])
        w = w.reshape(-1, in_channels, kh, kw)
        x = torch.nn.functional.conv2d(x, w, padding=0, groups=batch_size)
        x = x.reshape(batch_size, -1, *x.shape[2:])
        return torch.clamp(x + self.bias[None, :, None, None], -256, 256)


class SPADE_IN(torch.nn.Module):

    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.param_free_norm = torch.nn.BatchNorm2d(norm_nc, affine=False)

        nhidden = 128

        self.mlp_shared = torch.nn.Sequential(
            ConvDemodulated(label_nc, nhidden),
            torch.nn.ReLU()
        )
        self.mlp_gamma = ConvDemodulated(nhidden, norm_nc)
        self.mlp_beta = ConvDemodulated(nhidden, norm_nc)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = torch.nn.functional.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out


def test_generator():
    import time
    batch_size = 3
    G = Generator(512, 512, 2, 128, 3).cuda()
    for batch_idx in range(16):
        # sanity test forward pass
        z = torch.randn(batch_size, 512).to(torch.device("cuda:0"))
        s = torch.randn(batch_size, 6, 1, 128, 128).to(torch.device("cuda:0")).reshape(batch_size * 6, 1, 128, 128)
        w = G.mapping(z)
        w = w.unsqueeze(1).expand(-1, 6, -1, -1).reshape(batch_size * 6, w.shape[1], w.shape[2])
        t0 = time.time()
        fake = G.synthesis(w, s)
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

    model = Generator(512, 512, 2, 64, 3)
    print_model_parameter_count(model)

    test_generator()
