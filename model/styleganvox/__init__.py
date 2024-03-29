import torch
import numpy as np


@torch.jit.script
def clamp_gain(x: torch.Tensor, g: float, c: float):
    return torch.clamp(x * g, -c, c)


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def identity(x):
    return x


def leaky_relu_0_2(x):
    return torch.nn.functional.leaky_relu(x, 0.2)


activation_funcs = {
    "linear": {
        "fn": identity,
        "def_gain": 1
    },
    "lrelu": {
        "fn": leaky_relu_0_2,
        "def_gain": np.sqrt(2)
    }
}


class FullyConnectedLayer(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, activation='linear', lr_multiplier=1, bias_init=0):
        super().__init__()
        self.activation = activation_funcs[activation]['fn']
        self.activation_gain = activation_funcs[activation]['def_gain']
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None and self.bias_gain != 1:
            b = b * self.bias_gain
        x = self.activation(torch.addmm(b.unsqueeze(0), x, w.t())) * self.activation_gain
        return x


class SmoothDownsample(torch.nn.Module):

    def __init__(self):
        super().__init__()
        kernel = [[1, 3, 3, 1],
                  [3, 9, 9, 3],
                  [3, 9, 9, 3],
                  [1, 3, 3, 1]]
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        kernel /= kernel.sum()
        self.kernel = torch.nn.Parameter(kernel, requires_grad=False)
        self.pad = torch.nn.ReplicationPad2d((2, 1, 2, 1))

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = x.view(-1, 1, h, w)
        x = self.pad(x)
        x = torch.nn.functional.conv2d(x, self.kernel).view(b, c, h, w)
        x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='nearest', recompute_scale_factor=False)
        return x


class SmoothUpsample(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        return x


class EqualizedConv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, activation='linear', resample=identity):
        super().__init__()
        self.resample = resample
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.activation = activation_funcs[activation]['fn']
        self.activation_gain = activation_funcs[activation]['def_gain']
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size])
        bias = torch.zeros([out_channels]) if bias else None
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias) if bias is not None else None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias[None, :, None, None] if self.bias is not None else 0
        x = self.resample(x)
        x = torch.nn.functional.conv2d(x, w, padding=self.padding)
        return clamp_gain(self.activation(x + b), self.activation_gain * gain, 256 * gain)


def modulated_conv2d(x, weight, styles, padding=0, demodulate=True):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape

    # Calculate per-sample weights and demodulation coefficients.
    w = weight.unsqueeze(0)  # [NOIkk]
    w = w * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]

    # Execute as one fused op using grouped convolution.
    batch_size = int(batch_size)
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = torch.nn.functional.conv2d(x, w, padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x


def modulated_conv3d(x, weight, styles, padding=0, demodulate=True):
    batch_size = x.shape[0]
    out_channels, in_channels, kd, kh, kw = weight.shape

    # Calculate per-sample weights and demodulation coefficients.
    w = weight.unsqueeze(0)  # [NOIkkk]
    w = w * styles.reshape(batch_size, 1, -1, 1, 1, 1)  # [NOIkkk]

    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4, 5]) + 1e-8).rsqrt()  # [NO]
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1, 1)  # [NOIkk]

    # Execute as one fused op using grouped convolution.
    batch_size = int(batch_size)
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kd, kh, kw)
    x = torch.nn.functional.conv3d(x, w, padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x


class Conv3DResBlock(torch.nn.Module):

    def __init__(self, nf_in, nf_out, activation):
        super().__init__()
        self.nf_in = nf_in
        self.nf_out = nf_out
        self.norm_0 = torch.nn.BatchNorm3d(nf_in)
        self.conv_0 = torch.nn.Conv3d(nf_in, nf_out, kernel_size=3, padding=1)
        self.norm_1 = torch.nn.BatchNorm3d(nf_out)
        self.conv_1 = torch.nn.Conv3d(nf_out, nf_out, kernel_size=3, padding=1)
        self.activation = activation
        if nf_in != nf_out:
            self.nin_shortcut = torch.nn.Conv3d(nf_out, nf_out, kernel_size=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm_0(h)
        h = self.activation(h)
        h = self.conv_0(h)

        h = self.norm_1(h)
        h = self.activation(h)
        h = self.conv_1(h)

        if self.nf_in != self.nf_out:
            x = self.nin_shortcut(x)

        return x + h


class Conv3DBlock(torch.nn.Module):

    def __init__(self, nf_in, nf_out, activation):
        super().__init__()
        self.nf_in = nf_in
        self.nf_out = nf_out
        self.norm_0 = torch.nn.BatchNorm3d(nf_in)
        self.conv_0 = torch.nn.Conv3d(nf_in, nf_out, kernel_size=3, padding=1)
        self.activation = activation

    def forward(self, x):
        h = x
        h = self.norm_0(h)
        h = self.activation(h)
        h = self.conv_0(h)
        return h


class SDFEncoder(torch.nn.Module):
    def __init__(self, in_channels, layer_dims=(32, 64, 64, 128, 128, 256, 256, 256)):
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.enc_conv_in = torch.nn.Conv3d(in_channels, layer_dims[0], 1)
        self.down_0_block_0 = Conv3DBlock(layer_dims[0], layer_dims[1], self.activation)
        self.down_0_block_1 = Conv3DBlock(layer_dims[1], layer_dims[2], self.activation)
        self.down_1_block_0 = Conv3DBlock(layer_dims[2], layer_dims[3], self.activation)
        self.down_2_block_0 = Conv3DBlock(layer_dims[3], layer_dims[4], self.activation)
        self.down_3_block_0 = Conv3DBlock(layer_dims[4], layer_dims[5], self.activation)
        self.down_4_block_0 = Conv3DBlock(layer_dims[5], layer_dims[6], self.activation)
        self.enc_mid_block_0 = Conv3DBlock(layer_dims[6], layer_dims[7], self.activation)
        self.enc_out_conv = torch.nn.Conv3d(layer_dims[7], layer_dims[7], kernel_size=3, padding=1)
        self.enc_out_norm = torch.nn.BatchNorm3d(layer_dims[7])

    def forward(self, x):
        level_feats = []
        pool_ctr = 0
        x = self.enc_conv_in(x)
        x = self.down_0_block_0(x)
        x = self.down_0_block_1(x)
        level_feats.append(x)
        x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='trilinear', recompute_scale_factor=False, align_corners=True)
        pool_ctr += 1

        x = self.down_1_block_0(x)
        level_feats.append(x)
        x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='trilinear', recompute_scale_factor=False, align_corners=True)
        pool_ctr += 1

        x = self.down_2_block_0(x)
        level_feats.append(x)
        x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='trilinear', recompute_scale_factor=False, align_corners=True)
        pool_ctr += 1

        x = self.down_3_block_0(x)
        level_feats.append(x)
        x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='trilinear', recompute_scale_factor=False, align_corners=True)
        pool_ctr += 1

        x = self.down_4_block_0(x)
        # level_feats.append(x)
        # x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='trilinear', recompute_scale_factor=False, align_corners=True)
        # pool_ctr += 1

        x = self.enc_mid_block_0(x)
        x = self.enc_out_norm(x)
        x = self.activation(x)
        x = self.enc_out_conv(x)

        level_feats.append(x)

        return level_feats


if __name__ == '__main__':
    model = SDFEncoder(1).cuda()
    x = torch.randn(8, 1, 64, 64, 64).cuda()
    ll = model(x)
    for y in ll:
        print(y.shape)

