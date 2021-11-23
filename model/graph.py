import torch
import torch_scatter


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
