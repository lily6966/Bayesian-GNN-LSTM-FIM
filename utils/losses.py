"""
losses.py — Loss functions for GAT_RNN_V2 training.

Provides:
  stable_logcosh(x)        → numerically stable log-cosh (no float32 overflow)
  loss_fn(pred, true, args) → hybrid logcosh + optional BCE presence/absence
"""

import torch
import torch.nn.functional as F


def stable_logcosh(x):
    """
    Numerically stable log(cosh(x)).

    The naive implementation torch.log(torch.cosh(x)) overflows float32
    when |x| > 89 (cosh(89) ≈ 2.8e38 which saturates float32).

    Stable form (Softplus-based):
      log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
                   = |x| + softplus(-2|x|) - log(2)

    Gradient equals tanh(x) (same as naive form, no numerical issues).
    Max input magnitude for FIM: |z| ≈ 327 (rare species, log1p scale).
    """
    ae = torch.abs(x)
    return ae + F.softplus(-2 * ae) - torch.log(torch.tensor(2.0, device=x.device))


huber_fn = torch.nn.SmoothL1Loss()


def loss_fn(pred, Y, args, mode="logcosh"):
    """
    Hybrid loss = (1 - bce_weight) * logcosh_abundance + bce_weight * BCE_presence

    logcosh_abundance
        Operates on count-standardised scale so all species contribute equally.
        Uses stable_logcosh to prevent float32 overflow for rare species.

    BCE_presence (activated when args.bce_weight > 0)
        Binary cross-entropy treating (Y > 0) as the presence label and the
        raw log1p-scale prediction as the logit: P(present) = sigmoid(pred).
        Fires on ALL observations including true zeros.

    Supports:
      - 2-D  [B, n_sp]               (flat annual predictions)
      - 3-D  [B, length, n_sp]        (multi-year sequence)
      - 4-D  [B, length, 12, n_sp]    (nested monthly, with month_weights)
    """
    month_weights = getattr(args, 'month_weights', None)
    bce_weight    = float(getattr(args, 'bce_weight', 0.0))

    def to_std(t):
        cnt = torch.expm1(t) if not getattr(args, 'no_log_transform', False) else t
        cnt = torch.clamp(cnt, min=0.0)
        return (cnt - args.count_means) / args.count_stds

    def bce_presence(p, y):
        p_flat  = p.reshape(-1)
        y_flat  = y.reshape(-1)
        present = (y_flat > 0).float()
        valid   = ~(torch.isnan(p_flat) | torch.isnan(y_flat))
        if valid.sum() < 1:
            return torch.tensor(0.0, device=p.device)
        return F.binary_cross_entropy_with_logits(p_flat[valid], present[valid])

    # Monthly path: [*, 12, n_sp] with optional month_weights
    if (month_weights is not None
            and Y.dim() >= 3
            and Y.shape[-2] == 12):

        Y_std    = to_std(Y)
        pred_std = to_std(pred)

        err = Y_std - pred_std
        if mode == "huber":
            elem = F.huber_loss(pred_std, Y_std, reduction='none')
        else:
            elem = stable_logcosh(err)

        elem = elem * month_weights

        not_na         = ~torch.isnan(Y_std)
        abundance_loss = elem[not_na].mean()

        if bce_weight > 0:
            return ((1.0 - bce_weight) * abundance_loss
                    + bce_weight * bce_presence(pred, Y))
        return abundance_loss

    # Flat path: [B, n_sp] or [B, length, n_sp]
    Y_flat    = torch.reshape(Y,    (-1, Y.shape[-1]))
    pred_flat = torch.reshape(pred, (-1, pred.shape[-1]))

    Y_std    = to_std(Y_flat)
    pred_std = to_std(pred_flat)

    not_na   = ~torch.isnan(Y_std)
    pred_std = pred_std[not_na]
    Y_std    = Y_std[not_na]

    if mode == "huber":
        abundance_loss = huber_fn(pred_std, Y_std)
    else:
        err            = Y_std - pred_std
        abundance_loss = stable_logcosh(err).mean()

    if bce_weight > 0:
        return ((1.0 - bce_weight) * abundance_loss
                + bce_weight * bce_presence(pred_flat, Y_flat))
    return abundance_loss
