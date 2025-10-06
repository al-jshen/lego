from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from ..utils import linear_init, on_channel_first, zero_init


class CF(nn.Module):
    """
    Wrapper around nn.Module that allows for channel-first normalization.
    Does this by swapping the channel dimension to the front, applying the
    normalization, and then swapping it back.
    """

    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, *args, **kwargs):
        return on_channel_first(self.norm, *args, **kwargs)


def modulate(x, scale, shift):
    return x * (1.0 + scale) + shift


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.cond_proj = zero_init(nn.Linear(embedding_dim, input_dim * 2))

    def forward(
        self,
        x: Float[torch.Tensor, "b ... c"],
        cond: Float[torch.Tensor, "b d"],
    ) -> Float[torch.Tensor, "b ... c"]:
        c = x.shape[-1]
        num_spatial_dims = len(x.shape) - 2
        assert c == self.input_dim, f"input_dim must match the last dimension of x, got {c} != {self.input_dim}"

        cond = self.cond_proj(cond)[:, *((None,) * num_spatial_dims), :]
        scale, shift = cond.chunk(2, dim=-1)

        x = F.layer_norm(x, [c])

        x = modulate(x, scale, shift)

        return x


class MaybeAdaLayerNorm(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.adaptive = embedding_dim > 0
        if self.adaptive:
            self.norm = AdaLayerNorm(input_dim, embedding_dim)
        else:
            self.norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        x: Float[torch.Tensor, "b ... c"],
        cond: Optional[Float[torch.Tensor, "b d"]] = None,
    ) -> Float[torch.Tensor, "b ... c"]:
        if self.adaptive:
            return self.norm(x, cond)
        else:
            return self.norm(x)


def rms_norm(x, scale, eps):
    dtype = torch.promote_types(x.dtype, torch.float32)
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, input_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(input_dim))

    def forward(self, x: Float[torch.Tensor, "b ... c"]) -> Float[torch.Tensor, "b ... c"]:
        return rms_norm(x, self.scale, self.eps)


class AdaRMSNorm(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.linear = linear_init(nn.Linear(embedding_dim, input_dim, bias=False))

    def forward(
        self, x: Float[torch.Tensor, "b ... c"], cond: Float[torch.Tensor, "b d"]
    ) -> Float[torch.Tensor, "b ... c"]:
        num_spatial_dims = len(x.shape) - 2
        scale = 1.0 + self.linear(cond)[:, *((None,) * num_spatial_dims), :]
        return rms_norm(x, scale, self.eps)


class MaybeAdaRMSNorm(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.adaptive = embedding_dim > 0
        if self.adaptive:
            self.norm = AdaRMSNorm(input_dim, embedding_dim, eps)
        else:
            self.norm = RMSNorm(input_dim, eps)

    def forward(
        self,
        x: Float[torch.Tensor, "b ... c"],
        cond: Optional[Float[torch.Tensor, "b d"]] = None,
    ) -> Float[torch.Tensor, "b ... c"]:
        if self.adaptive:
            return self.norm(x, cond)
        else:
            return self.norm(x)


class RevIN(nn.Module):
    """
    Affine reversible instance norm.
    """

    def __init__(self, channels):
        super().__init__()
        self.dims = channels
        self.norm_scale = nn.Parameter(torch.ones(1, channels))  # extra dim for batch
        self.norm_shift = nn.Parameter(torch.zeros(1, channels))
        self.unnorm_scale = nn.Parameter(torch.ones(1, channels))
        self.unnorm_shift = nn.Parameter(torch.zeros(1, channels))

    def normalize(self, x):
        rest = tuple(range(2, x.ndim))  # everything but batch and channels
        assert x.shape[1] == self.dims  # proper number of channels
        std, mean = torch.std_mean(x, axis=rest, keepdim=True)
        n_rest = len(rest)
        return (
            ((x - mean) / std) * self.expand_back(self.norm_scale, n_rest) + self.expand_back(self.norm_shift, n_rest),
            mean,
            std,
        )

    def forward(self, x):
        return self.normalize(x)

    def unnormalize(self, x, mean, std):
        n_rest = len(x.shape) - 2
        x = (x - self.expand_back(self.norm_shift, n_rest)) / self.expand_back(self.norm_scale, n_rest)
        x = x * std + mean
        x = x * self.expand_back(self.unnorm_scale, n_rest) + self.expand_back(self.unnorm_shift, n_rest)
        return x

    def expand_back(self, x, num_dims):
        return x[..., *[None for _ in range(num_dims)]]
