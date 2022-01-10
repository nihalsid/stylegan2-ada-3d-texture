import torch
from torch import nn
import nvdiffrast.torch as dr
from torchvision.ops import masks_to_boxes


def transform_pos(pos, projection_matrix, world_to_cam_matrix):
    # (x,y,z) -> (x,y,z,1)
    t_mtx = torch.matmul(projection_matrix, world_to_cam_matrix)
    # noinspection PyArgumentList
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(pos.device)], axis=1)
    return torch.matmul(posw, t_mtx.t())


def transform_pos_mvp(pos, mvp):
    # noinspection PyArgumentList
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(pos.device)], axis=1)
    return torch.bmm(posw.unsqueeze(0).expand(mvp.shape[0], -1, -1), mvp.permute((0, 2, 1))).reshape((-1, 4))


def render(glctx, pos_clip, pos_idx, vtx_col, col_idx, resolution, ranges, background=None):
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution], ranges=ranges)
    color, _ = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
    color = dr.antialias(color, rast_out, pos_clip, pos_idx)
    mask = color[..., -1:] == 0
    if background is None:
        one_tensor = torch.ones((color.shape[0], color.shape[3], 1, 1), device=color.device)
    else:
        one_tensor = background
    one_tensor_permuted = one_tensor.permute((0, 2, 3, 1)).contiguous()
    color = torch.where(mask, one_tensor_permuted, color)
    return color[:, :, :, :-1]


def render_in_bounds(glctx, pos_clip, pos_idx, vtx_col, col_idx, resolution, ranges, color_space, background=None):
    render_resolution = int(resolution * 1.2)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[render_resolution, render_resolution], ranges=ranges)
    color, _ = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
    color = dr.antialias(color, rast_out, pos_clip, pos_idx)
    mask = color[..., -1:] == 0
    if background is None:
        if color_space == 'rgb':
            one_tensor = torch.ones((color.shape[0], color.shape[3], 1, 1), device=color.device)
        else:
            one_tensor = torch.zeros((color.shape[0], color.shape[3], 1, 1), device=color.device)
            one_tensor[:, 0, :, :] = 1
    else:
        one_tensor = background
    one_tensor_permuted = one_tensor.permute((0, 2, 3, 1)).contiguous()
    color = torch.where(mask, one_tensor_permuted, color)  # [:, :, :, :-1]
    color[..., -1:] = mask.float()
    color_crops = []
    boxes = masks_to_boxes(torch.logical_not(mask.squeeze(-1)))
    for img_idx in range(color.shape[0]):
        x1, y1, x2, y2 = [int(val) for val in boxes[img_idx, :].tolist()]
        color_crop = color[img_idx, y1: y2, x1: x2, :].permute((2, 0, 1))
        pad = [[0, 0], [0, 0]]
        if y2 - y1 > x2 - x1:
            total_pad = (y2 - y1) - (x2 - x1)
            pad[0][0] = total_pad // 2
            pad[0][1] = total_pad - pad[0][0]
            pad[1][0], pad[1][1] = 0, 0
            additional_pad = int((y2 - y1) * 0.1)
        else:
            total_pad = (x2 - x1) - (y2 - y1)
            pad[0][0], pad[0][1] = 0, 0
            pad[1][0] = total_pad // 2
            pad[1][1] = total_pad - pad[1][0]
            additional_pad = int((x2 - x1) * 0.1)
        for i in range(4):
            pad[i // 2][i % 2] += additional_pad

        padded = torch.ones((color_crop.shape[0], color_crop.shape[1] + pad[1][0] + pad[1][1], color_crop.shape[2] + pad[0][0] + pad[0][1]), device=color_crop.device)
        padded[:3, :, :] = padded[:3, :, :] * one_tensor[img_idx, :3, :, :]
        padded[:, pad[1][0]: padded.shape[1] - pad[1][1], pad[0][0]: padded.shape[2] - pad[0][1]] = color_crop
        # color_crop = T.Pad((pad[0][0], pad[1][0], pad[0][1], pad[1][1]), 1)(color_crop)
        color_crop = torch.nn.functional.interpolate(padded.unsqueeze(0), size=(resolution, resolution), mode='bilinear', align_corners=False).permute((0, 2, 3, 1))
        color_crops.append(color_crop)
    return torch.cat(color_crops, dim=0)


def intrinsic_to_projection(intrinsic_matrix):
    near, far = 0.1, 50.
    a, b = -(far + near) / (far - near), -2 * far * near / (far - near)
    projection_matrix = torch.tensor([
        intrinsic_matrix[0][0] / intrinsic_matrix[0][2],    0,                                                0, 0,
        0,                                                  -intrinsic_matrix[1][1] / intrinsic_matrix[1][2], 0, 0,
        0,                                                  0,                                                a, b,
        0,                                                  0,                                               -1, 0
    ]).float().reshape((4, 4))
    return projection_matrix


class DifferentiableRenderer(nn.Module):

    def __init__(self, resolution, mode='standard', color_space='rgb', num_channels=3):
        super().__init__()
        self.glctx = dr.RasterizeGLContext()
        self.resolution = resolution
        self.render_func = render
        self.color_space = color_space
        self.num_channels = num_channels
        if mode == 'bounds':
            self.render_func = render_in_bounds

    def render(self, vertex_positions, triface_indices, vertex_colors, ranges=None, background=None, resolution=None):
        if ranges is None:
            ranges = torch.tensor([[0, triface_indices.shape[0]]]).int()
        if resolution is None:
            resolution = self.resolution
        color = self.render_func(self.glctx, vertex_positions, triface_indices, vertex_colors, triface_indices, resolution, ranges, self.color_space, background)
        return color[:, :, :, :self.num_channels]
