import torch
import torch.nn as nn
import math
from functools import lru_cache


class SinusoidalEmbedding(nn.Module):
    def __init__(self, hidden_dim, min_period, max_period, dtype=torch.float32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.min_period = min_period
        self.max_period = max_period
        self.register_buffer(
            "freqs",
            2
            * math.pi
            / torch.logspace(
                math.log10(min_period),
                math.log10(max_period),
                hidden_dim // 2,
                dtype=dtype,
            ),
        )

    def forward(self, x):
        triarg = self.freqs * x.unsqueeze(-1)
        # interleave sin and cos
        pe = torch.zeros(*x.shape[:2], self.hidden_dim, device=x.device, dtype=x.dtype)
        pe[:, :, 0::2] = torch.sin(triarg)
        pe[:, :, 1::2] = torch.cos(triarg)
        return pe


@lru_cache
def sinusoidal_embedding_1d(n: int | torch.Tensor, dim, temperature=10000, dtype=torch.float32):
    if isinstance(n, int) or (isinstance(n, torch.Tensor) and n.numel() == 1):
        n = torch.arange(n)
    else:
        assert n.ndim == 1, "n must be 1D tensor, if not an integer"
        # leave it alone if it is a 1D tensor
    assert (dim % 2) == 0, "feature dimension must be multiple of 2 for sincos emb"
    omega = torch.arange(dim // 2) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.zeros(n.shape[0], dim)
    pe[:, 0::2] = n.sin()
    pe[:, 1::2] = n.cos()
    # pe = torch.cat((n.sin(), n.cos()), dim = 1)
    return pe.type(dtype)


@lru_cache
def sinusoidal_embedding_2d(
    h: int | torch.Tensor, w: int | torch.Tensor, dim, temperature: int = 10000, dtype=torch.float32
):
    assert type(h) is type(w), "h and w must be of the same type"
    if isinstance(h, int) or (isinstance(h, torch.Tensor) and h.numel() == 1):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    elif isinstance(h, torch.Tensor) and h.ndim == 1:
        y, x = torch.meshgrid(h, w, indexing="ij")
    else:
        raise ValueError("h and w must be either integers or 1D tensors")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.zeros(x.shape[0], dim)
    pe[:, 0::4] = x.sin()
    pe[:, 1::4] = x.cos()
    pe[:, 2::4] = y.sin()
    pe[:, 3::4] = y.cos()
    # pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)
