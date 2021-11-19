import torch
import numpy as np

from model import SmoothDownsample, EqualizedConv2d, FullyConnectedLayer


class Discriminator(torch.nn.Module):

    def __init__(self, img_resolution, img_channels, channel_base=16384, channel_max=512):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        module_list = [EqualizedConv2d(img_channels, channels_dict[img_resolution], kernel_size=1, activation='lrelu')]
        for res in self.block_resolutions:
            in_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            module_list.append(DiscriminatorBlock(in_channels, out_channels))
        module_list.append(DiscriminatorEpilogue(channels_dict[4], resolution=4))
        self.net = torch.nn.Sequential(*module_list)

    def forward(self, x):
        return self.net(x)


class DiscriminatorBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = 0
        downsampler = SmoothDownsample()
        self.conv0 = EqualizedConv2d(in_channels, in_channels, kernel_size=3, activation=activation)
        self.conv1 = EqualizedConv2d(in_channels, out_channels, kernel_size=3, activation=activation, resample=downsampler)
        self.skip = EqualizedConv2d(in_channels, out_channels, kernel_size=1, bias=False, resample=downsampler)

    def forward(self, x):
        y = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x)
        x = self.conv1(x, gain=np.sqrt(0.5))
        x = y.add_(x)
        return x


class DiscriminatorEpilogue(torch.nn.Module):

    def __init__(self, in_channels, resolution, mbstd_group_size=4, mbstd_num_channels=1, activation='lrelu'):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution

        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = EqualizedConv2d(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1)

    def forward(self, x):
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)
        return x


class MinibatchStdLayer(torch.nn.Module):

    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x


if __name__ == '__main__':
    from util.misc import print_model_parameter_count, print_module_summary

    model = Discriminator(img_resolution=64, img_channels=3)
    print_module_summary(model, (torch.randn((16, 3, 64, 64)), ))
    print_model_parameter_count(model)
