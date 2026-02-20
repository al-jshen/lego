from abc import ABC
from typing import Optional

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

from .embedding import TimestepEmbedding
from .norm import AdaLayerNormZero


def autograd_trace(outputs, inputs):
    trJ = 0.0
    dims = inputs.shape[1]
    for i in range(dims):
        trJ += torch.autograd.grad(outputs[:, i].sum(), inputs, create_graph=True)[0][
            :, i
        ]
    return trJ


class TimeSchedule(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_size: int):
        raise NotImplementedError


class BetaSchedule(TimeSchedule):
    def __init__(self, alpha: float = 2.0, beta: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, batch_size: int):
        return torch.distributions.beta.Beta(self.alpha, self.beta).sample(
            (batch_size,)
        )


class ODEFnWrapper(nn.Module):
    def __init__(self, ode_func):
        super().__init__()
        self.ode_func = ode_func

    def forward(self, *args, **kwargs):
        return self.ode_func(*args, **kwargs)


class MLPBackboneBlock(nn.Module):
    """MLP denoising backbone block for a token-level model"""

    def __init__(self, data_dim, cond_dim, hidden_dim=None):
        super().__init__()
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else data_dim * 2

        self.adaln = AdaLayerNormZero(data_dim, cond_dim)
        self.linear1 = nn.Linear(data_dim, self.hidden_dim)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(self.hidden_dim, data_dim)

    def forward(self, x, t, context=None):
        out, gate = self.adaln(x, context)  # input dim, gate is for residual connection
        out = self.linear2(self.activation(self.linear1(out)))
        return out * gate + x


class MLPBackbone(nn.Module):
    """MLP denoising backbone for a token-level model"""

    def __init__(self, data_dim, cond_dim, hidden_dim=None, num_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MLPBackboneBlock(data_dim, cond_dim, hidden_dim)
                for _ in range(num_blocks)
            ]
        )
        self.temb = TimestepEmbedding(cond_dim)

    def forward(self, x, t, context=None):
        ctx = self.temb(t) + context  # mix time and context
        for block in self.blocks:
            x = block(x, t, context=ctx)
        return x


class RectifiedFlow(nn.Module):
    def __init__(
        self,
        time_schedule: nn.Module,
        score_model: nn.Module,
        conditioner: nn.Module = nn.Identity(),
        conditioner_dropout_p: float = 0.0,
    ):
        super().__init__()
        self.time_schedule = time_schedule
        self.score_model = score_model
        self.conditioner = conditioner
        self.conditioner_dropout = nn.Dropout(conditioner_dropout_p, inplace=True)
        self.register_buffer("base_mu", torch.tensor(0.0))
        self.register_buffer("base_sigma", torch.tensor(1))
        self.base_dist = torch.distributions.Normal

    def sample_base(self, sample_shape, temperature: float = 1.0):
        return self.base_dist(self.base_mu, self.base_sigma * temperature).sample(
            sample_shape
        )

    def log_prob_base(self, inputs):
        return (
            self.base_dist(self.base_mu, self.base_sigma)
            .log_prob(inputs)
            .sum(axis=list(range(1, inputs.ndim)))
        )  # sum over all data dims (not batch)

    def forward_score_model(self, x, t, context=None):
        return self.score_model(x, t, context=context)

        # x prediction, v loss
        # t_expanded = t.view(-1, *([1] * (x.ndim - 1)))
        # x_pred = self.score_model(x, t, context=context)  # x + v = x_pred
        # v_pred = (x_pred - x) / torch.clip(1 - t_expanded, min=0.05)
        # return v_pred

    def loss(self, target_samples, context=None):
        context = self.conditioner(context)
        context = self.conditioner_dropout(context)
        base_samples = self.sample_base(target_samples.shape)  # b c ...
        times = self.time_schedule(target_samples.shape[0]).to(base_samples)
        expand_dims = (1,) * (target_samples.ndim - 1)
        times_expanded = times.view(-1, *expand_dims)
        inputs = times_expanded * target_samples + (1.0 - times_expanded) * base_samples
        velocity = self.forward_score_model(inputs, times, context=context)
        target = target_samples - base_samples
        loss = torch.mean(
            (target - velocity) ** 2, dim=tuple(range(1, base_samples.ndim))
        )
        return loss

    def step(self, batch, batch_idx=None, **kwargs):
        if len(batch) == 2:
            x, context = batch
            return self.loss(x, context).mean()
        # otherwise, just one element (unconditional)
        return self.loss(batch).mean()

    def make_ode_fn(self, context, cfg_w, null_context=None):
        if context is not None:
            context = self.conditioner(context)
            if null_context is not None:
                null_context = self.conditioner(null_context)
            else:
                null_context = torch.zeros_like(context)
        else:
            null_context = None

        def ode_func(t, yt):
            t = torch.full((yt.shape[0],), t.item(), device=yt.device, dtype=yt.dtype)
            if cfg_w == 0.0:  # unconditional model
                assert null_context is not None
                v = self.forward_score_model(yt, t, context=null_context)
            elif cfg_w == 1.0:  # conditional model
                v = self.forward_score_model(yt, t, context=context)
            else:  # combination of conditional and unconditional
                assert null_context is not None
                batched_context = torch.cat([null_context, context], dim=0)
                # repeat yt and t twice along dim 0
                yt = torch.cat([yt, yt], dim=0)
                t = torch.cat([t, t], dim=0)
                # batched forward
                v = self.forward_score_model(yt, t, context=batched_context)
                # split and combine
                v_uncond, v_cond = torch.split(
                    v, [context.shape[0], context.shape[0]], dim=0
                )
                v = v_uncond + cfg_w * (v_cond - v_uncond)
            return v

        return ode_func

    def sample(
        self,
        sample_shape,
        context: Optional[torch.Tensor] = None,
        null_context: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        atol=1e-5,
        rtol=1e-5,
        trace: str = "exact",
        adjoint: bool = False,
        cfg_w: float = 1.0,
        with_logprob: bool = False,
        **kwargs,
    ):
        """Sample from the flow. First draw from the base distribution, and then
        send the samples through the score model to get the velocity. Integrate
        the velocity to get the resulting samples.

        Args:
            null_context: Explicit unconditional context for CFG. If provided,
                used instead of the default zeros. This allows the caller to
                supply the transformer's output when run without external
                conditioning, so CFG interpolates between the true conditional
                and unconditional velocity fields.
        """

        inputs = self.sample_base(sample_shape, temperature=temperature)

        _ode_fn = self.make_ode_fn(context, cfg_w, null_context=null_context)

        integrator = odeint_adjoint if adjoint else odeint

        if with_logprob:
            noise = torch.randn_like(inputs)

            def ode_wrapper_hutchinson(t, y):
                y, _ = y
                fn_out, dfdy = torch.autograd.functional.vjp(
                    lambda x: _ode_fn(t=t, yt=x), y, noise
                )  # hardcode noise in here
                logp = (dfdy * noise).sum(axis=-1)
                return fn_out, logp

            def ode_wrapper_exact(t, y):
                y, _ = y
                with torch.enable_grad():
                    y = y.requires_grad_(True)
                    fn_out = _ode_fn(t=t, yt=y)
                    dfdy = autograd_trace(fn_out, y)
                return fn_out, dfdy

            ode_fn = dict(
                exact=ode_wrapper_exact,
                hutchinson=ode_wrapper_hutchinson,
            )[trace]

            if adjoint:
                ode_fn = ODEFnWrapper(ode_fn)

            init_delta_logp = torch.zeros(
                inputs.shape[0], device=inputs.device, dtype=inputs.dtype
            )

            ret, delta_logp = integrator(
                ode_fn,
                (inputs, init_delta_logp),
                torch.tensor([0.0, 1.0], device=inputs.device, dtype=inputs.dtype),
                atol=atol,
                rtol=rtol,
                **kwargs,
            )

            delta_logp = delta_logp[-1]
            log_prob = self.log_prob_base(inputs)
            lp = log_prob - delta_logp
        else:
            ret = integrator(
                _ode_fn,
                inputs,
                torch.tensor([0.0, 1.0], device=inputs.device, dtype=inputs.dtype),
                atol=atol,
                rtol=rtol,
                **kwargs,
            )

            lp = None

        ret = ret[-1]
        if context is None:
            ret = ret.squeeze(1)

        return ret, lp

    def log_prob(
        self,
        inputs,
        context=None,
        null_context: Optional[torch.Tensor] = None,
        atol=1e-10,
        rtol=1e-5,
        trace="exact",
        adjoint: bool = False,
        cfg_w: float = 1.0,
        **kwargs,
    ):
        """Compute the log probability of the inputs under the flow. Do this by
        taking the input samples and reverse integrating them back to the base distribution.
        Then compute the log probability of the base samples under the base distribution.
        """

        noise = torch.randn_like(inputs)

        _ode_fn = self.make_ode_fn(context, cfg_w)

        integrator = odeint_adjoint if adjoint else odeint

        def ode_wrapper_hutchinson(t, y):
            y, _ = y
            fn_out, dfdy = torch.autograd.functional.vjp(
                lambda x: _ode_fn(t=t, yt=x), y, noise
            )  # hardcode noise in here
            logp = (dfdy * noise).sum(axis=-1)
            return fn_out, logp

        def ode_wrapper_exact(t, y):
            y, _ = y
            with torch.enable_grad():
                y = y.requires_grad_(True)
                fn_out = _ode_fn(t=t, yt=y)
                dfdy = autograd_trace(fn_out, y)
            return fn_out, dfdy

        ode_fn = dict(
            exact=ode_wrapper_exact,
            hutchinson=ode_wrapper_hutchinson,
        )[trace]

        if adjoint:
            ode_fn = ODEFnWrapper(ode_fn)

        init_delta_logp = torch.zeros(
            inputs.shape[0], device=inputs.device, dtype=inputs.dtype
        )

        out, delta_logp = integrator(
            ode_fn,
            (inputs, init_delta_logp),
            torch.tensor([1.0, 0.0], device=inputs.device, dtype=inputs.dtype),
            atol=atol,
            rtol=rtol,
            **kwargs,
        )
        out, delta_logp = out[-1], delta_logp[-1]

        log_prob = self.log_prob_base(out)
        return log_prob + delta_logp
