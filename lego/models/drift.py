from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_V(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperature: float,
    mask_self: bool = True,
) -> torch.Tensor:
    """
    Compute the drifting field V (Algorithm 2 from paper, Page 12).

    This is the EXACT implementation from the paper's pseudocode.

    Args:
        x: Generated samples in feature space, shape (N, D)
        y_pos: Positive (real data) samples, shape (N_pos, D)
        y_neg: Negative (generated) samples, shape (N_neg, D)
        temperature: Temperature for softmax (smaller = sharper)
        mask_self: Whether to mask self-distances (when y_neg == x)

    Returns:
        V: Drifting field, shape (N, D)
    """
    N = x.shape[0]
    N_pos = y_pos.shape[0]
    N_neg = y_neg.shape[0]
    device = x.device

    # 1. Compute pairwise L2 distances
    dist_pos = torch.cdist(x, y_pos, p=2)  # (N, N_pos)
    dist_neg = torch.cdist(x, y_neg, p=2)  # (N, N_neg)

    # 2. Mask self-distances (when y_neg contains x)
    if mask_self and N == N_neg:
        mask = torch.eye(N, device=device) * 1e6
        dist_neg = dist_neg + mask

    # 3. Compute logits
    logit_pos = -dist_pos / temperature  # (N, N_pos)
    logit_neg = -dist_neg / temperature  # (N, N_neg)

    # 4. Concat for normalization
    logit = torch.cat([logit_pos, logit_neg], dim=1)  # (N, N_pos + N_neg)

    # 5. Normalize along BOTH dimensions (key insight from paper)
    A_row = torch.softmax(logit, dim=1)  # softmax over y (columns)
    A_col = torch.softmax(logit, dim=0)  # softmax over x (rows)
    A = torch.sqrt(A_row * A_col)  # geometric mean

    # 6. Split back to pos and neg
    A_pos = A[:, :N_pos]  # (N, N_pos)
    A_neg = A[:, N_pos:]  # (N, N_neg)

    # 7. Compute weights (cross-weighting from paper)
    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)  # (N, N_pos)
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)  # (N, N_neg)

    # 8. Compute drift
    drift_pos = torch.mm(W_pos, y_pos)  # (N, D)
    drift_neg = torch.mm(W_neg, y_neg)  # (N, D)

    V = drift_pos - drift_neg

    return V


def compute_drifting_loss(
    x_gen: torch.Tensor,
    x_pos: torch.Tensor,
    feature_encoder: Optional[nn.Module] = None,
    temperatures: list = [0.02, 0.05, 0.2],
    use_pixel_space: bool = True,
):
    device = x_gen.device

    if use_pixel_space or feature_encoder is None:
        # Pixel space: single scale
        feat_gen_list = [x_gen.flatten(start_dim=1)]
        feat_pos_list = [x_pos.flatten(start_dim=1)]
    else:
        # Multi-scale feature maps from pretrained encoder
        feat_gen_maps = feature_encoder(x_gen)  # List of (B, C, H, W)
        with torch.no_grad():
            feat_pos_maps = feature_encoder(x_pos)

        # Global average pool each scale to get vectors
        feat_gen_list = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feat_gen_maps]
        feat_pos_list = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feat_pos_maps]

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_drift_norm = 0.0

    # Compute loss at each scale
    for scale_idx, (feat_gen, feat_pos) in enumerate(zip(feat_gen_list, feat_pos_list)):
        # Simple L2 normalization (projects to unit sphere)
        feat_gen_norm = F.normalize(feat_gen, p=2, dim=1)
        feat_pos_norm = F.normalize(feat_pos, p=2, dim=1)

        feat_neg_norm = feat_gen_norm

        # Compute V with multiple temperatures
        V_total = torch.zeros_like(feat_gen_norm)
        for tau in temperatures:
            V_tau = compute_V(
                feat_gen_norm,
                feat_pos_norm,
                feat_neg_norm,
                tau,
                mask_self=True,  # y_neg = x, so mask self
            )
            # Normalize each V before summing
            v_norm = torch.sqrt(torch.mean(V_tau**2) + 1e-8)
            V_tau = V_tau / (v_norm + 1e-8)
            V_total = V_total + V_tau

        # Loss: MSE(phi(x), stopgrad(phi(x) + V))
        target = (feat_gen_norm + V_total).detach()
        loss_scale = F.mse_loss(feat_gen_norm, target)

        total_loss = total_loss + loss_scale
        total_drift_norm += (V_total**2).mean().item() ** 0.5

    loss = total_loss
    info = {
        "loss": loss.item(),
        "drift_norm": total_drift_norm,
    }

    return loss, info
