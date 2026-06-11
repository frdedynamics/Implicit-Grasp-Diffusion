"""
Drop-in replacement for torch_scatter using native PyTorch (>=1.12).
"""
import torch

def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    if dim_size is None:
        if out is not None:
            dim_size = out.size(dim)
        elif index.numel() == 0:
            dim_size = 1
        else:
            dim_size = int(index.max().item()) + 1

    if out is None:
        size = list(src.size())
        size[dim] = dim_size
        out = src.new_zeros(size)

    if index.numel() == 0:
        return out

    # index is [B, N], src is [B, C, N]
    # need to expand index to [B, C, N] by repeating across the C dimension
    while index.dim() < src.dim():
        index = index.unsqueeze(-2)   # insert before the last dim
    index = index.expand_as(src)

    index = index.clamp(0, dim_size - 1)
    out.scatter_reduce_(dim, index, src, reduce='mean', include_self=True)
    return out

def scatter_max(src, index, dim=-1, out=None, dim_size=None):
    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    if out is None:
        size = list(src.size())
        size[dim] = dim_size
        out = src.new_full(size, fill_value=float('-inf'))

    while index.dim() < src.dim():
        index = index.unsqueeze(-2)
    index = index.expand_as(src)
    index = index.clamp(0, dim_size - 1)

    out.scatter_reduce_(dim, index, src, reduce='amax', include_self=True)
    out[out == float('-inf')] = 0.0
    argmax = torch.zeros_like(out, dtype=torch.long)
    return out, argmax

def _expand_index(index, src, dim):
    if index.dim() == src.dim():
        return index.expand_as(src)
    for _ in range(src.dim() - index.dim()):
        index = index.unsqueeze(1)
    return index.expand_as(src)