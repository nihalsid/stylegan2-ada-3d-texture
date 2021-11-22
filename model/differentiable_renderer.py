import torch
from torch import nn
import nvdiffrast.torch as dr


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


def render(glctx, pos_clip, pos_idx, vtx_col, col_idx, resolution, ranges):
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution], ranges=ranges)
    color, _ = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
    color = dr.antialias(color, rast_out, pos_clip, pos_idx)
    return color


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

    def __init__(self, resolution):
        super().__init__()
        self.glctx = dr.RasterizeGLContext()
        self.resolution = resolution

    def render(self, vertex_positions, triface_indices, vertex_colors, ranges=None):
        if ranges is None:
            ranges = torch.tensor([[0, triface_indices.shape[0]]]).int()
        color = render(self.glctx, vertex_positions, triface_indices, vertex_colors, triface_indices, self.resolution, ranges)
        return color
