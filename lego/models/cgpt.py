import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from tqdm.auto import tqdm

from .attention import KVCache, TransformerBlock, precompute_freqs_cis
from .norm import RMSNorm
from ..utils import find_multiple


class ContinuousGPT(nn.Module):
    """
    Continuous-valued causal transformer with a flow model output head.

    Operates on generic sequential data of shape [B, S, ...] where:
      - B is the batch dimension
      - S is the sequence dimension (autoregressive axis)
      - ... is the token shape (flattened internally to token_dim for embedding)

    Any domain-specific preprocessing (patchification, spatial folding into batch,
    etc.) should happen externally before passing data to this model.

    Supports optional conditioning via prefix vectors prepended to the sequence.
    Conditioning vectors must be pre-projected to the model's hidden dimension
    (``dim``). They participate in causal self-attention as context for the data
    tokens but are stripped from the output.

    Examples:
      - Next-patch image generation: [B, N, C*pH*pW]  (N patches per image)
      - Video with spatial patch folding: [B*nph*npw, T, C*psh*psw]
      - Video with whole-frame tokens: [B, T, C*H*W]
      - Class-conditional: cond [B, 1, dim] from nn.Embedding + linear proj
      - Image-conditional: cond [B, N_img, dim] from a vision encoder + linear proj

    Args:
        flow: Flow model (e.g. RectifiedFlow) used as the output head.
        token_dim: Flat dimension of each input token (prod of trailing dims).
        dim: Hidden dimension of the transformer.
        output_dim: Dimension of the output projection. Defaults to dim.
            This is what the flow receives as conditioning.
    """

    def __init__(
        self,
        flow: nn.Module,
        token_dim: int,
        dim: int = 4096,
        output_dim: Optional[int] = None,
        n_layer: int = 32,
        n_head: int = 32,
        n_kv_head: Optional[int] = None,
        ffn_hidden_dim: Optional[int] = None,
        rope_base: float = 10000,
        norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.1,
        ffn_dropout_p: float = 0.1,
        drop_path_rate: float = 0.0,
        cond_dropout_p: float = 0.0,
        block_size: int = 256,
    ):
        super().__init__()
        self.n_head = n_head
        self.dim = dim
        self.output_dim = output_dim or dim
        self.token_dim = token_dim
        self.rope_base = rope_base
        self.initializer_range = initializer_range
        self.n_layer = n_layer
        self.cond_dropout_p = cond_dropout_p
        self.block_size = block_size

        self.tok_embeddings = nn.Linear(token_dim, dim, bias=False)
        self.output_proj = nn.Linear(dim, self.output_dim, bias=False)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layer)]
        self.layers = nn.ModuleList(
            TransformerBlock(
                dim=dim,
                n_head=n_head,
                n_kv_head=n_kv_head,
                attn_dropout_p=attn_dropout_p,
                resid_dropout_p=resid_dropout_p,
                ffn_hidden_dim=ffn_hidden_dim,
                ffn_dropout_p=ffn_dropout_p,
                norm_eps=norm_eps,
                drop_path=dpr[i],
            )
            for i in range(n_layer)
        )

        self.norm = RMSNorm(dim, eps=norm_eps)
        self.flow = flow

        self.freqs_cis = precompute_freqs_cis(
            self.block_size, dim // n_head, self.rope_base, 0
        )

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    # ---- KV Cache Management ----

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        head_dim = self.dim // self.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        for layer in self.layers:
            layer.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_length,
                layer.attention.n_kv_head,
                head_dim,
                dtype,
            )

        causal_mask = torch.tril(
            torch.ones(max_seq_length, max_seq_length, dtype=torch.bool)
        )
        self.causal_mask = causal_mask.unsqueeze(0)

        self.freqs_cis = precompute_freqs_cis(
            self.block_size, self.dim // self.n_head, self.rope_base, 0
        )

    def clear_caches(self):
        for layer in self.layers:
            layer.attention.kv_cache = None

    # ---- Forward ----

    def forward(
        self,
        idx: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            idx: [B, S, ...] input tokens. Trailing dims are flattened to token_dim.
            cond: Optional [B, N_cond, dim] conditioning vectors prepended to the
                sequence. Must already be projected to the model's hidden dimension.
            input_pos: Optional position indices for KV-cached inference. When
                ``cond`` is provided, must cover both conditioning and data
                positions (length ``N_cond + S``).
            mask: Optional attention mask (auto-built from causal_mask when using KV cache).

        Returns:
            [B, S, output_dim] conditioning vectors for the flow model
            (prefix conditioning tokens are stripped from the output).
        """
        b, s = idx.shape[:2]
        x = idx.reshape(b, s, -1)
        x = self.tok_embeddings(x)

        n_cond = 0
        if cond is not None:
            n_cond = cond.shape[1]
            x = torch.cat([cond, x], dim=1)

        self.freqs_cis = self.freqs_cis.to(x.device)
        total_s = n_cond + s
        freqs_cis = (
            self.freqs_cis[:total_s]
            if input_pos is None
            else self.freqs_cis[input_pos]
        )

        if input_pos is not None:
            causal_mask = self.causal_mask.to(x.device).expand(b, -1, -1)
            mask = causal_mask[:, None, input_pos]

        for layer in self.layers:
            x = layer(x, freqs_cis, input_pos, mask)

        x = self.norm(x)
        x = self.output_proj(x)

        if n_cond > 0:
            x = x[:, n_cond:]

        return x

    # ---- Training ----

    def step(self, batch, batch_idx=None, **kwargs):
        """
        Args:
            batch: Either a single tensor [B, S, ...] or a tuple of
                (sequence [B, S, ...], cond [B, N_cond, dim]).

        During training, per-sample conditioning dropout is applied: each
        sample's conditioning is independently zeroed with probability
        ``cond_dropout_p``, teaching the model an unconditional mode for CFG.
        """
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, cond = batch
        else:
            x, cond = batch, None

        if cond is not None and self.training and self.cond_dropout_p > 0:
            keep = (torch.rand(x.shape[0], 1, 1, device=cond.device) >= self.cond_dropout_p).to(cond.dtype)
            cond = cond * keep

        x_in, x_out = x[:, :-1], x[:, 1:]
        conditioner = self(idx=x_in, cond=cond)
        x_out = rearrange(x_out, "b s ... -> (b s) ...")
        conditioner = rearrange(conditioner, "b s ... -> (b s) ...")
        loss = self.flow.step((x_out, conditioner))
        return loss

    # ---- Inference ----

    def _sample_flow(self, conditioner, token_shape, uncond_conditioner=None, cfg_scale=1.0, **kwargs):
        """Sample next token from the flow, conditioned on the last sequence position.

        Args:
            conditioner: [B, S, output_dim] conditional transformer output.
            uncond_conditioner: Optional [B, S, output_dim] unconditional
                transformer output (from running without external cond).
                Passed to the flow as ``null_context`` for CFG.
        """
        last_cond = conditioner[:, -1]  # [B, output_dim]
        last_uncond = uncond_conditioner[:, -1] if uncond_conditioner is not None else None
        b = last_cond.shape[0]
        samples, _ = self.flow.sample(
            (b, *token_shape),
            context=last_cond,
            null_context=last_uncond,
            cfg_w=cfg_scale,
            **kwargs,
        )
        return samples.unsqueeze(1)  # [B, 1, *token_shape]

    @torch.no_grad()
    def generate_nocache(self, idx, max_new_tokens, cond=None, cfg_scale=1.0, **kwargs):
        """Autoregressive generation without KV caching (recomputes full context each step)."""
        self.clear_caches()
        token_shape = idx.shape[2:]
        null_cond = torch.zeros_like(cond) if cond is not None else None
        for _ in tqdm(range(max_new_tokens)):
            cond_out = self(idx, cond=cond)
            if null_cond is not None and cfg_scale != 1.0:
                uncond_out = self(idx, cond=null_cond)
            else:
                uncond_out = None
            samp = self._sample_flow(
                cond_out, uncond_conditioner=uncond_out,
                token_shape=token_shape, cfg_scale=cfg_scale, **kwargs,
            )
            idx = torch.cat((idx, samp), dim=1)
        return idx

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, cond=None, cfg_scale=1.0, **kwargs):
        """Autoregressive generation with KV caching (prefill + incremental decode).

        When ``cfg_scale != 1.0`` and ``cond`` is provided, the batch is
        doubled internally: the first half runs with zeroed conditioning
        (unconditional) and the second half with real conditioning, sharing a
        single KV cache for efficiency.
        """
        b, t = idx.shape[:2]
        token_shape = idx.shape[2:]
        device = idx.device
        n_cond = cond.shape[1] if cond is not None else 0
        use_cfg = cfg_scale != 1.0 and cond is not None

        if use_cfg:
            idx_batched = idx.repeat(2, *([1] * (idx.ndim - 1)))
            cond_batched = torch.cat([torch.zeros_like(cond), cond], dim=0)
            effective_b = 2 * b
        else:
            idx_batched = idx
            cond_batched = cond
            effective_b = b

        self.clear_caches()
        with torch.device(device):
            self.setup_caches(
                max_batch_size=effective_b,
                max_seq_length=n_cond + t + max_new_tokens,
                dtype=self.tok_embeddings.weight.dtype,
            )

        # prefill: conditioning tokens (if any) + prompt
        input_pos = torch.arange(n_cond + t, device=device, dtype=torch.int)
        out = self(idx_batched, cond=cond_batched, input_pos=input_pos)

        if use_cfg:
            uncond_out, cond_out = out.chunk(2, dim=0)
            samp = self._sample_flow(
                cond_out, uncond_conditioner=uncond_out,
                token_shape=token_shape, cfg_scale=cfg_scale, **kwargs,
            )
            samp_batched = samp.repeat(2, *([1] * (samp.ndim - 1)))
        else:
            samp = self._sample_flow(
                out, token_shape=token_shape, cfg_scale=cfg_scale, **kwargs,
            )
            samp_batched = samp

        idx = torch.cat((idx, samp), dim=1)

        # incremental decode: cond is already in the KV cache
        input_pos = torch.tensor([n_cond + t], device=device, dtype=torch.int)
        for _ in tqdm(range(max_new_tokens - 1)):
            out = self(samp_batched, input_pos=input_pos)
            if use_cfg:
                uncond_out, cond_out = out.chunk(2, dim=0)
                samp = self._sample_flow(
                    cond_out, uncond_conditioner=uncond_out,
                    token_shape=token_shape, cfg_scale=cfg_scale, **kwargs,
                )
                samp_batched = samp.repeat(2, *([1] * (samp.ndim - 1)))
            else:
                samp = self._sample_flow(
                    out, token_shape=token_shape, cfg_scale=cfg_scale, **kwargs,
                )
                samp_batched = samp
            idx = torch.cat((idx, samp), dim=1)
            input_pos += 1

        return idx
