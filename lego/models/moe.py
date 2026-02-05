from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import FeedForward


class Gate(nn.Module):
    """
    DeepSeek-V3: Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.
    """

    def __init__(
        self,
        dim: int = 64,
        n_activated_experts: int = 2,
        n_routed_experts: int = 8,
        n_expert_groups: int = 1,
        n_limited_groups: int = 1,
        score_func: str = "softmax",
        route_scale: float = 1.0,
    ):
        """
        Initializes the Gate module.

        Args:
            dim (int): Dimensionality of input features.
            n_activated_experts (int): Number of top experts activated for each input.
            n_routed_experts (int): Total number of experts available for routing.
            n_expert_groups (int): Number of expert groups for routing.
            n_limited_groups (int): Number of limited groups for MoE routing.
            score_func (str): Scoring function ('softmax' or 'sigmoid').
            route_scale (float): Scaling factor for routing weights.
        """
        super().__init__()
        self.dim = dim
        self.topk = n_activated_experts
        self.n_routed_experts = n_routed_experts
        self.n_groups = n_expert_groups
        self.topk_groups = n_limited_groups
        self.score_func = score_func
        self.route_scale = route_scale
        self.weight = nn.Parameter(torch.empty(n_routed_experts, dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = F.linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        # if self.bias is not None:
        #     scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            group_scores = scores.amax(dim=-1)
            # if self.bias is None:
            #     group_scores = scores.amax(dim=-1)
            # else:
            #     group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(
                1, indices, False
            )
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices


class MoE(nn.Module):
    """
    DeepSeek-V3 Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """

    def __init__(
        self,
        dim: int = 64,
        n_routed_experts: int = 8,
        n_activated_experts: int = 2,
        moe_inter_dim: int = 64,
        n_shared_experts: int = 2,
        n_expert_groups: int = 1,
        n_limited_groups: int = 1,
        score_func: str = "softmax",
        route_scale: float = 1.0,
    ):
        """
        Initializes the MoE module.
        """
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.moe_inter_dim = moe_inter_dim
        self.n_shared_experts = n_shared_experts
        self.n_expert_groups = n_expert_groups
        self.n_limited_groups = n_limited_groups
        self.score_func = score_func
        self.route_scale = route_scale
        self.gate = Gate(
            dim=self.dim,
            n_activated_experts=self.n_activated_experts,
            n_routed_experts=self.n_routed_experts,
            n_expert_groups=n_expert_groups,
            n_limited_groups=n_limited_groups,
            score_func=score_func,
            route_scale=route_scale,
        )
        self.experts = nn.ModuleList(
            [
                FeedForward(self.dim, hidden_dim=moe_inter_dim)
                for i in range(self.n_routed_experts)
            ]
        )
        self.shared_experts = FeedForward(
            self.dim, hidden_dim=self.n_shared_experts * moe_inter_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        ).tolist()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        return (y + z).view(shape)
