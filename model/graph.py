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
