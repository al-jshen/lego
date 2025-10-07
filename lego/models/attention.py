from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import DropPath, FeedForward
from .norm import RMSNorm


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=1):
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack(
        [freqs_cis.real, freqs_cis.imag], dim=-1
    )  # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache]
    )  # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache


def precompute_freqs_cis_2d(
    grid_size: int, n_elem: int, base: int = 10000, cls_token_num=1
):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (
        base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim)
    )
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (grid_size, head_dim // 2)
    freqs_grid = torch.concat(
        [
            freqs[:, None, :].expand(-1, grid_size, -1),
            freqs[None, :, :].expand(grid_size, -1, -1),
        ],
        dim=-1,
    )  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack(
        [torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1
    )  # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache]
    )  # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(
        *x.shape[:-1], -1, 2
    )  # (bs, seq_len, n_head, head_dim//2, 2)
    if freqs_cis.ndim == 3:
        freqs_cis = freqs_cis.view(
            1, xshaped.size(1), 1, xshaped.size(3), 2
        )  # (1, seq_len, 1, head_dim//2, 2)
    elif freqs_cis.ndim == 4:
        freqs_cis = freqs_cis.view(
            xshaped.size(0), xshaped.size(1), 1, xshaped.size(3), 2
        )
    else:
        raise ValueError(f"freqs_cis should be 3D or 4D, got {freqs_cis.ndim}D")
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        dim=-1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(
        self,
        dim: int = 4096,
        n_head: int = 8,
        n_kv_head: Optional[int] = None,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.kv_cache = None

        self.qnorm = RMSNorm(self.head_dim, eps=1e-5)  # query normalization
        self.knorm = RMSNorm(self.head_dim, eps=1e-5)  # key normalization

        # regularization
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)
        self.causal = causal

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor = None,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        xq = self.qnorm(xq)
        xk = self.knorm(xk)

        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        if mask is not None:
            # match batch size to xq/keys/values
            nrep = bsz // mask.shape[0]
            mask = mask.repeat(nrep, *(1 for _ in range(mask.ndim - 1)))
            # ok, so now mask is [bsz, seqlen, seqlen]
            # but xq and keys are possibly [bsz, ..., seqlen, seqlen]
            # so we need to expand mask to match the shape of xq and keys
            n_expand = xq.ndim - mask.ndim
            for _ in range(n_expand):
                mask = mask.unsqueeze(1)

        output = F.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=mask,
            is_causal=self.causal
            and mask is None,  # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0,
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 4096,
        n_head: int = 8,
        n_kv_head: Optional[int] = None,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.0,
        ffn_dim_multiplier: Optional[float] = None,
        ffn_dropout_p: float = 0.0,
        norm_eps: float = 1e-5,
        multiple_of: int = 256,
        drop_path: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        self.attention = Attention(
            dim=dim,
            n_head=n_head,
            n_kv_head=n_kv_head,
            attn_dropout_p=attn_dropout_p,
            resid_dropout_p=resid_dropout_p,
            causal=causal,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            ffn_dim_multiplier=ffn_dim_multiplier,
            ffn_dropout_p=ffn_dropout_p,
            multiple_of=multiple_of,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        input_pos: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        h = x + self.drop_path(
            self.attention(self.attention_norm(x), freqs_cis, input_pos, mask)
        )
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out
