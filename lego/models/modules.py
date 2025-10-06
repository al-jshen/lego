from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Lambda(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class GEGLU(nn.Module):
    def __init__(self, dim: int, embed_dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.fc = nn.Linear(dim, (embed_dim or dim) * 2, bias=False)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * F.gelu(b, approximate="tanh")


class SwiGLU(nn.Module):
    def __init__(self, dim: int, embed_dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.fc = nn.Linear(dim, (embed_dim or dim) * 2, bias=False)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * F.silu(b, inplace=True)  # silu = swish


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=None,
        hidden_dim=None,
        dropout=0.0,
        activation=None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        output_dim = output_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation or nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )

        self.net.apply(self.init_weights)

    def forward(self, x):
        return self.net(x)


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, spatial_dims, embed_dim):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.gamma = nn.Parameter(
            torch.zeros(1, *(1 for _ in range(spatial_dims)), embed_dim)
        )
        self.beta = nn.Parameter(
            torch.zeros(1, *(1 for _ in range(spatial_dims)), embed_dim)
        )

    def forward(self, x):
        Gx = torch.norm(
            x, p=2, dim=tuple(range(1, 1 + self.spatial_dims)), keepdim=True
        )
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class DropoutDim(nn.Module):
    def __init__(self, p=0.5, axis=(-1,), inplace=False):
        """
        Custom dropout that zeros out slices along specified dimensions.

        Args:
            p (float): Probability of an element being zeroed out.
            dim (int or tuple of ints): Dimension(s) along which to apply dropout.
            inplace (bool): If True, does the operation in-place.
        """
        super().__init__()
        self.p = p
        self.dim = (axis,) if isinstance(axis, int) else axis
        self.inplace = inplace

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        mask_shape = list(x.shape)
        for d in self.dim:
            mask_shape[d] = 1  # Create dropout mask along these dimensions

        mask = torch.ones(mask_shape, device=x.device, dtype=x.dtype)
        mask = F.dropout(mask, p=self.p, training=True, inplace=self.inplace)

        return x * mask


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int = 4096,
        ffn_dim_multiplier: Optional[float] = None,
        ffn_dropout_p: float = 0.1,
        multiple_of: int = 256,
    ):
        super().__init__()
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, multiple_of)

        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.ffn_dropout = nn.Dropout(ffn_dropout_p)

    def forward(self, x):
        x1, x3 = torch.chunk(self.w13(x), 2, dim=-1)
        return self.ffn_dropout(self.w2(F.silu(x1) * x3))
