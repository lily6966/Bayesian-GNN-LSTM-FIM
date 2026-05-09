"""
train.py — Training pipeline for GAT_RNN_V2.

Adapts gnn-rnn/train.py to use:
  - GAT_RNN_V2 model
  - HybridTransformerEncoder
  - 3-dim edge features [dist_w, hab_sim, env_sim]
  - Default max_epoch=30
  - Cross-donor Bayesian transfer priors
  - refresh_transfer_priors() each epoch
"""

import csv
import math
import os
import sys
import datetime
import random
from copy import copy, deepcopy
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

import dgl
import scipy.sparse as sp
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GAT_RNN_V2
from utils import get_X_Y, get_X_Y_2D, build_path, get_git_revision_hash

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

import warnings

def _resolve_device(requested: str = "auto") -> torch.device:
    """Resolve device string to a torch.device.
    'auto' picks cuda → cpu in priority order.
    Note: MPS (Apple Metal) is not supported by DGL and cannot be used.
    """
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return torch.device("cuda:0")
    return torch.device("cpu")

device = _resolve_device()  # updated to args.device inside train()
print("Device", device)


# ── Trunk layer names (species-agnostic, transferable across tasks) ────────────
_TRUNK_PREFIXES = (
    'gat1.', 'gat2.',
    'outer_lstm.',
    'shared_proj.',
)


def load_trunk_weights(model, ckpt_path):
    """Load only trunk (species-agnostic) layers from a checkpoint."""
    sd = torch.load(ckpt_path, map_location=device)
    model_sd = model.state_dict()
    to_load  = {k: v for k, v in sd.items()
                if k in model_sd and any(k.startswith(p) for p in _TRUNK_PREFIXES)}
    model_sd.update(to_load)
    model.load_state_dict(model_sd, strict=False)
    print(f"  [MAML] Loaded {len(to_load)} trunk tensors from {ckpt_path}")
    skipped = [k for k in sd if k not in to_load]
    if skipped:
        print(f"  [MAML] Skipped (species-specific or shape mismatch): {skipped[:6]}")


def freeze_trunk(model):
    """Freeze trunk parameters — only FiLM/species heads are updated."""
    frozen = []
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in _TRUNK_PREFIXES):
            param.requires_grad_(False)
            frozen.append(name)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [MAML] Froze {len(frozen)} trunk params. "
          f"Trainable params remaining: {n_trainable:,}")


# ── Evaluation ────────────────────────────────────────────────────────────────

METRICS = {'rmse', 'r2', 'corr'}

best_test = {'rmse': 1e9, 'r2': -1e9, 'corr': -1e9}
best_val  = {'rmse': 1e9, 'r2': -1e9, 'corr': -1e9}


def eval(pred, Y, args):
    """
    Evaluate predictions against true labels on COUNT scale.
    Converts log1p predictions → counts via expm1, then computes metrics.
    """
    if getattr(args, 'no_log_transform', False):
        Y_count    = Y
        pred_count = torch.clamp(pred, min=0.0)
    else:
        Y_count    = torch.expm1(Y)
        pred_count = torch.clamp(torch.expm1(pred), min=0.0)

    pred_np = pred_count.flatten().detach().cpu().numpy()
    Y_np    = Y_count.flatten().detach().cpu().numpy()

    valid = ~np.isnan(Y_np) & ~np.isnan(pred_np)
    pred_np = pred_np[valid]
    Y_np    = Y_np[valid]

    if Y_np.shape[0] < 2:
        print("Not enough valid labels in this batch")
        return {'rmse': 0, 'r2': 0, 'corr': 0, 'mae': 0, 'mse': 0, 'mape': 0}

    metrics = {}
    metrics['rmse'] = np.sqrt(np.mean((pred_np - Y_np) ** 2))
    metrics['r2']   = r2_score(Y_np, pred_np)
    if np.all(Y_np == Y_np[0]) or np.all(pred_np == pred_np[0]):
        metrics['corr'] = 0
    else:
        metrics['corr'] = np.corrcoef(Y_np, pred_np)[0, 1]
    metrics['mae']  = MAE(Y_np, pred_np)
    metrics['mse']  = np.mean((pred_np - Y_np) ** 2)
    metrics['mape'] = np.mean(np.abs((Y_np - pred_np) / (Y_np + 1e-5)))

    return metrics


def update_metrics(rmse, r2, corr, mode):
    if mode == "Val":
        best_val['rmse'] = rmse
        best_val['r2']   = r2
        best_val['corr'] = corr
    elif mode == "Test":
        best_test['rmse'] = rmse
        best_test['r2']   = r2
        best_test['corr'] = corr


# ── Stable log-cosh loss ──────────────────────────────────────────────────────

def stable_logcosh(x):
    """Numerically stable log(cosh(x)) — avoids float32 overflow."""
    ae = torch.abs(x)
    return ae + F.softplus(-2 * ae) - torch.log(torch.tensor(2.0, device=x.device))


def _safe_expm1_np(a, max_log1p=30.0):
    """np.expm1 with log1p clipping to avoid float32 overflow for rare
    community species whose raw counts would otherwise be astronomical."""
    return np.expm1(np.clip(a, a_min=None, a_max=max_log1p))


def _to_count_np(a, transform='log1p'):
    """Back-transform model output array to raw count scale based on the
    training transform. Used when writing prediction CSVs.

      log1p   :  count = expm1(y)         (with overflow guard)
      root4   :  count = max(y, 0) ** 4
    """
    if transform == 'root4':
        return np.power(np.clip(a, a_min=0.0, a_max=None), 4)
    return _safe_expm1_np(a)


huber_fn = nn.SmoothL1Loss()


def _set_window_month_weights(args, T, n_windows):
    """When per-window calendar weights are loaded, set args.month_weights to
    a [n_windows, 12, n_sp] tensor for the n_windows year_ends implied by T.
    Falls back to the nearest year_end if a year is missing from the dict.
    No-op if per-window weights are not loaded."""
    w_dict = getattr(args, 'window_month_weights_by_yearend', None)
    if not w_dict:
        return
    known = sorted(w_dict.keys())
    def _nearest(ye):
        return min(known, key=lambda k: abs(k - ye))
    stacked = []
    for w in range(1, n_windows + 1):
        year_end = T - (n_windows - w)
        key = year_end if year_end in w_dict else _nearest(year_end)
        stacked.append(w_dict[key])
    args.month_weights = torch.stack(stacked, dim=0)   # [n_windows, 12, n_sp]


def hurdle_loss(pred, Y, args, presence_logit, month_weights=None):
    """Hurdle (zero-inflated two-stage) loss:
       presence head: BCE on (y > 0) — all rows
       abundance head: logcosh on (pred, log1p y) — y > 0 rows only

    Standardization is done in **log1p space, per species** (using
    args.means/args.stds), so all species have ≈N(0,1) residuals and the
    abundance loss is balanced across species regardless of their raw count
    magnitude (Pinfish ~ thousands of fish, Sheepshead ~ a few).
    """
    if month_weights is None:
        month_weights = getattr(args, 'month_weights', None)
    pres_w = float(getattr(args, 'hurdle_pres_weight', 0.4))

    not_na   = ~torch.isnan(Y)
    presence = (Y > 0).float()

    # Presence loss (all observed rows)
    pres_flat = presence_logit[not_na]
    y_flat    = presence[not_na]
    if pres_flat.numel() < 1:
        pres_loss = torch.tensor(0.0, device=pred.device)
    else:
        pres_loss = F.binary_cross_entropy_with_logits(pres_flat, y_flat)

    # Abundance loss on observed presences only — log1p-standardized residuals
    pres_mask = (Y > 0) & not_na
    if pres_mask.sum() < 1:
        abund_loss = torch.tensor(0.0, device=pred.device)
    else:
        Y_std    = (Y    - args.means) / args.stds
        pred_std = (pred - args.means) / args.stds
        elem     = stable_logcosh(Y_std - pred_std)
        if month_weights is not None and Y.dim() >= 3 and Y.shape[-2] == 12:
            elem = elem * month_weights
        abund_loss = elem[pres_mask].mean()

    return (1.0 - pres_w) * abund_loss + pres_w * pres_loss


def loss_fn(pred, Y, args, mode="logcosh", month_weights=None):
    """
    Hybrid loss = (1 - bce_weight) * logcosh_abundance + bce_weight * BCE_presence

    month_weights override lets callers pass a per-window slice
    (e.g. stacked[:-1] or stacked[-1]) instead of args.month_weights.
    """
    if month_weights is None:
        month_weights = getattr(args, 'month_weights', None)
    bce_weight    = float(getattr(args, 'bce_weight', 0.0))

    def to_std(t):
        # Log1p-scale per-species standardization — aligns abundance magnitudes
        # across species so the loss is balanced regardless of raw count scale
        # (e.g. Pinfish thousands vs Sheepshead a few).
        return (t - args.means) / args.stds

    def bce_presence(p, y):
        p_flat  = p.reshape(-1)
        y_flat  = y.reshape(-1)
        present = (y_flat > 0).float()
        valid   = ~(torch.isnan(p_flat) | torch.isnan(y_flat))
        if valid.sum() < 1:
            return torch.tensor(0.0, device=p.device)
        return F.binary_cross_entropy_with_logits(p_flat[valid], present[valid])

    # Monthly path: [*, 12, n_sp] with optional month_weights
    if (month_weights is not None and Y.dim() >= 3 and Y.shape[-2] == 12):
        Y_std    = to_std(Y)
        pred_std = to_std(pred)
        err      = Y_std - pred_std
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
    Y_std     = to_std(Y_flat)
    pred_std  = to_std(pred_flat)
    not_na    = ~torch.isnan(Y_std)
    pred_std  = pred_std[not_na]
    Y_std     = Y_std[not_na]
    if mode == "huber":
        abundance_loss = huber_fn(pred_std, Y_std)
    else:
        err            = Y_std - pred_std
        abundance_loss = stable_logcosh(err).mean()
    if bce_weight > 0:
        return ((1.0 - bce_weight) * abundance_loss
                + bce_weight * bce_presence(pred_flat, Y_flat))
    return abundance_loss


# ── Post-hoc per-species OLS calibration ─────────────────────────────────────

def calibrate_species(val_df, test_df, output_names, results_dir):
    """
    Fit per-species linear calibration on val set, apply to test.
    Saves calibrated_test_results.csv.
    Returns sp_metrics dict, mean_corr float, cal_params dict.
    """
    from scipy import stats as scipy_stats

    cal_params = {}
    test_cal   = test_df.copy()

    for sp in output_names:
        pc = f"predicted_{sp}"
        tc = f"true_{sp}"
        if pc not in val_df.columns or tc not in val_df.columns:
            continue
        vp = val_df[pc].values.astype(float)
        vt = val_df[tc].values.astype(float)
        mask = ~(np.isnan(vp) | np.isnan(vt))
        vp, vt = vp[mask], vt[mask]
        if len(vp) < 5 or np.std(vp) < 1e-9:
            bias_only = float(np.mean(vt) - np.mean(vp))
            cal_params[sp] = (1.0, bias_only)
            continue
        corr_val = float(np.corrcoef(vp, vt)[0, 1])
        if corr_val <= 0:
            bias_only = float(np.mean(vt) - np.mean(vp))
            cal_params[sp] = (1.0, bias_only)
            continue
        slope, intercept, *_ = scipy_stats.linregress(vp, vt)
        slope = float(np.clip(slope, 0.5, 3.0))
        cal_params[sp] = (slope, intercept)
        if pc in test_cal.columns:
            test_cal[pc] = slope * test_cal[pc].values.astype(float) + intercept

    sp_metrics = {}
    all_corrs  = []
    for sp in output_names:
        pc = f"predicted_{sp}"
        tc = f"true_{sp}"
        if pc not in test_cal.columns:
            continue
        pred = test_cal[pc].values.astype(float)
        true = test_cal[tc].values.astype(float)
        mask = ~(np.isnan(pred) | np.isnan(true))
        pred, true = pred[mask], true[mask]
        if len(pred) < 5:
            continue
        corr = float(np.corrcoef(pred, true)[0, 1]) if np.std(pred) > 1e-9 else 0.0
        ss_res = np.sum((pred - true) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        mae  = float(np.mean(np.abs(pred - true)))
        bias = float(np.mean(pred - true))
        binary_true = (true > 0).astype(int)
        n_pos = binary_true.sum()
        n_neg = len(binary_true) - n_pos
        if n_pos >= 2 and n_neg >= 2:
            try:
                roc_auc = float(roc_auc_score(binary_true, pred))
                pr_auc  = float(average_precision_score(binary_true, pred))
            except Exception:
                roc_auc = pr_auc = float('nan')
        else:
            roc_auc = pr_auc = float('nan')
        sp_metrics[sp] = {'corr': corr, 'r2': r2, 'rmse': rmse, 'mae': mae,
                          'bias': bias, 'roc_auc': roc_auc, 'pr_auc': pr_auc}
        all_corrs.append(corr)

    cal_path = os.path.join(results_dir, "calibrated_test_results.csv")
    test_cal.to_csv(cal_path, index=False)

    mean_corr = float(np.mean(all_corrs)) if all_corrs else 0.0
    valid_roc = [m['roc_auc'] for m in sp_metrics.values() if not np.isnan(m['roc_auc'])]
    valid_pr  = [m['pr_auc']  for m in sp_metrics.values() if not np.isnan(m['pr_auc'])]
    mean_roc  = float(np.mean(valid_roc)) if valid_roc else float('nan')
    mean_pr   = float(np.mean(valid_pr))  if valid_pr  else float('nan')

    print(f"\n── Post-hoc calibration applied ({len(cal_params)} species) ──")
    print(f"   Mean species corr  (test, calibrated): {mean_corr:.4f}")
    print(f"   Mean ROC-AUC       (presence/absence): {mean_roc:.4f}")
    print(f"   Mean PR-AUC        (presence/absence): {mean_pr:.4f}")
    print(f"   Saved → {cal_path}\n")

    return sp_metrics, mean_corr, cal_params


# ── Edge feature attachment ───────────────────────────────────────────────────

def _attach_dist_weights(blocks, args):
    """
    Attach 3-dim edge features [dist_w, hab_sim, env_sim] to DGL blocks.

    Supports both:
      Scalar (old format): float → edge feat dim = 1
      Vector (new format): np.ndarray shape (D,) → edge feat dim = D

    For GAT_RNN_V2, D=3 carries [dist_w, hab_sim, env_sim].
    """
    if not getattr(args, 'use_gat', True):
        return
    dist_w = getattr(args, 'dist_weights', None)
    if dist_w is None:
        return

    _sample = next(iter(dist_w.values()))
    if np.ndim(_sample) == 0:
        edge_feat_dim = 1
        default_feat  = np.ones(1, dtype=np.float32)
    else:
        edge_feat_dim = int(np.asarray(_sample).shape[0])
        default_feat  = np.ones(edge_feat_dim, dtype=np.float32)

    for blk in blocks:
        edges_src, edges_dst = blk.edges()
        src_ids = blk.srcdata[dgl.NID].cpu().numpy()
        dst_ids = blk.dstdata[dgl.NID].cpu().numpy()
        feats = []
        for s, d in zip(edges_src.cpu().numpy(), edges_dst.cpu().numpy()):
            global_s = int(src_ids[s])
            global_d = int(dst_ids[d])
            val = dist_w.get((global_s, global_d), default_feat)
            if np.ndim(val) == 0:
                val = np.array([float(val)], dtype=np.float32)
            feats.append(np.asarray(val, dtype=np.float32))
        blk.edata['dist_w'] = torch.tensor(
            np.stack(feats, axis=0), dtype=torch.float32,
            device=blk.device
        )


def load_subtensor(year_XY, year, in_nodes, out_nodes, device):
    X, Y, counties = year_XY[year]
    X        = X.to(device)
    Y        = Y.to(device)
    counties = counties.to(device)
    in_nodes  = in_nodes.to(device)
    out_nodes = out_nodes.to(device)
    return (X[in_nodes].float(), Y[out_nodes].float(),
            counties[out_nodes].int())


# ── 2D Training Epoch ─────────────────────────────────────────────────────────

def train_epoch_2D(args, model, device, nodeloader, year_XY, county_avg,
                   year_avg_Y, cur_step, optimizer, epoch, writer=None):
    """Training epoch for GAT_RNN_V2. X has shape [n, n_years+1, 12, n_feat]."""
    print("\n----- Epoch {} (2D nested) -----".format(epoch))
    model.train()
    tot_loss = 0.
    all_pred, all_Y = [], []
    result_dfs = []
    n_batch = 0

    all_years = list(year_XY.keys())
    np.random.shuffle(all_years)
    for year in all_years:
        if year == args.test_year or year == args.test_year - 1:
            continue
        for batch_idx, (in_nodes, out_nodes, blocks) in enumerate(nodeloader):
            X, Y, counties = year_XY[year]
            X        = X[in_nodes].float().to(device)
            Y        = Y[out_nodes].float().to(device)
            counties = counties[out_nodes].int().to(device)

            if torch.isnan(Y[:, -1]).all():
                continue

            Y_std = (Y - args.means) / args.stds

            blocks = [b.int().to(device) for b in blocks]
            _attach_dist_weights(blocks, args)
            pred_std = model(blocks, X, Y_std[:, :-1])
            pred     = pred_std * args.stds + args.means

            loss = (loss_fn(pred[:, :args.length - 1], Y[:, :args.length - 1], args) * args.c1 +
                    loss_fn(pred[:, -1],                Y[:, -1],                args) * args.c2)

            # Bayesian KL regularisation
            beta_kl = getattr(args, 'beta_kl', 0.0)
            if beta_kl > 0 and hasattr(model, 'bayes_kl_loss'):
                rarity_w = getattr(args, 'species_rarity_weights', None)
                kl = model.bayes_kl_loss(rarity_w) / max(X.shape[0], 1)
                loss = loss + beta_kl * kl

            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            n_batch += 1

            all_pred.append(pred[:, -1].reshape(-1, pred.shape[-1]))
            all_Y.append(Y[:, -1].reshape(-1, Y.shape[-1]))
            tot_loss += loss.item()

            if n_batch % args.check_freq == 0:
                m = eval(pred[:, -1].reshape(-1, pred.shape[-1]),
                         Y[:, -1].reshape(-1, Y.shape[-1]), args)
                print("batch {} | loss {:.4f} rmse {:.3f} r2 {:.3f} corr {:.3f}".format(
                    n_batch - 1, loss.item(), m['rmse'], m['r2'], m['corr']))

            if writer is not None:
                writer.add_scalar("Train/loss", loss.item(), cur_step)
            cur_step += 1

            fips_np = counties.detach().cpu().numpy().astype(int).tolist()
            for seq_pos in range(args.length):
                actual_year = year - (args.length - 1) + seq_pos
                for mo in range(12):
                    row = {"fips":  fips_np,
                           "year":  [actual_year] * len(fips_np),
                           "month": [mo + 1] * len(fips_np)}
                    for i, name in enumerate(args.output_names):
                        _pred_np = pred[:, seq_pos, mo, i].detach().cpu().numpy()
                        _true_np = Y[:, seq_pos, mo, i].detach().cpu().numpy()
                        if getattr(args, 'no_log_transform', False):
                            row["predicted_" + name] = np.clip(_pred_np, 0, None).tolist()
                            row["true_"      + name] = _true_np.tolist()
                        else:
                            row["predicted_" + name] = np.clip(_to_count_np(_pred_np, getattr(args,'transform','log1p')), 0, None).tolist()
                            row["true_"      + name] = _to_count_np(_true_np, getattr(args,'transform','log1p')).tolist()
                    result_dfs.append(pd.DataFrame(row))

    results  = pd.concat(result_dfs)
    all_pred = torch.cat(all_pred, dim=0)
    all_Y    = torch.cat(all_Y,    dim=0)
    metrics  = eval(all_pred, all_Y, args)
    print("\n###### Overall 2D training metrics")
    print("loss: {:.4f} rmse: {:.3f} r2: {:.3f} corr: {:.3f}".format(
        tot_loss / max(n_batch, 1), metrics['rmse'], metrics['r2'], metrics['corr']))
    return cur_step, metrics, results


# ── 2D Validation/Test Epoch ──────────────────────────────────────────────────

def val_epoch_2D(args, model, device, nodeloader, year_XY, county_avg,
                 year_avg_Y, epoch, mode="Val", writer=None):
    """Val/Test epoch for GAT_RNN_V2."""
    print("***** Epoch {} {} (2D nested) *****".format(epoch, mode))
    model.eval()
    year = args.test_year - 1 if mode == "Val" else args.test_year
    tot_loss = 0.
    all_pred, all_Y = [], []
    result_dfs = []

    with torch.no_grad():
        for batch_idx, (in_nodes, out_nodes, blocks) in enumerate(nodeloader):
            X, Y, counties = year_XY[year]
            X        = X[in_nodes].float().to(device)
            Y        = Y[out_nodes].float().to(device)
            counties = counties[out_nodes].int().to(device)
            Y_std    = (Y - args.means) / args.stds

            blocks   = [b.int().to(device) for b in blocks]
            _attach_dist_weights(blocks, args)
            pred_std = model(blocks, X, Y_std[:, :-1])
            pred     = pred_std * args.stds + args.means

            loss = loss_fn(pred[:, -1], Y[:, -1], args)
            tot_loss += loss.item()

            all_pred.append(pred[:, -1].reshape(-1, pred.shape[-1]))
            all_Y.append(Y[:, -1].reshape(-1, Y.shape[-1]))

            fips_np = counties.detach().cpu().numpy().astype(int).tolist()
            for seq_pos in range(args.length):
                actual_year = year - (args.length - 1) + seq_pos
                for mo in range(12):
                    row = {"fips":  fips_np,
                           "year":  [actual_year] * len(fips_np),
                           "month": [mo + 1] * len(fips_np)}
                    for i, name in enumerate(args.output_names):
                        _pred_np = pred[:, seq_pos, mo, i].detach().cpu().numpy()
                        _true_np = Y[:, seq_pos, mo, i].detach().cpu().numpy()
                        if getattr(args, 'no_log_transform', False):
                            row["predicted_" + name] = np.clip(_pred_np, 0, None).tolist()
                            row["true_"      + name] = _true_np.tolist()
                        else:
                            row["predicted_" + name] = np.clip(_to_count_np(_pred_np, getattr(args,'transform','log1p')), 0, None).tolist()
                            row["true_"      + name] = _to_count_np(_true_np, getattr(args,'transform','log1p')).tolist()
                    result_dfs.append(pd.DataFrame(row))

    results  = pd.concat(result_dfs)
    all_pred = torch.cat(all_pred, dim=0)
    all_Y    = torch.cat(all_Y,    dim=0)
    metrics  = eval(all_pred, all_Y, args)
    n_batch  = batch_idx + 1
    print("loss: {:.4f} rmse: {:.3f} r2: {:.3f} corr: {:.3f}".format(
        tot_loss / n_batch, metrics['rmse'], metrics['r2'], metrics['corr']))
    print("********************")

    if writer is not None:
        writer.add_scalar("{}/rmse".format(mode), metrics['rmse'], epoch)
        writer.add_scalar("{}/r2".format(mode),   metrics['r2'],   epoch)
        writer.add_scalar("{}/corr".format(mode), metrics['corr'], epoch)
    return metrics, results


# ── Main train() function ─────────────────────────────────────────────────────

def train(args):
    """
    Full training loop for GAT_RNN_V2.
    """
    global device
    device = _resolve_device(getattr(args, 'device', 'auto'))
    print("Device", device)

    print('reading npy...')
    torch.set_num_threads(12)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dgl.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    raw_data = np.load(args.data_dir)
    data     = raw_data['data']

    # GAT_RNN_V2 always uses nested 2D data
    args.nested = True
    print("  [train] GAT_RNN_V2 → using --nested (monthly 2D data path)")

    X_dict, Y_dict, avail_dict, adj, order_map, min_year, max_year, county_set, county_avg, year_avg_Y = \
        get_X_Y_2D(data, args, device)

    print("Average Y calculated originally")
    for year in year_avg_Y:
        year_avg_Y[year] = torch.tensor(year_avg_Y[year], device=device)

    sp_adj = sp.coo_matrix(adj)
    g = dgl.from_scipy(sp_adj).to(device)
    g = dgl.add_self_loop(g)

    N = adj.shape[0]
    sampler    = dgl.dataloading.MultiLayerNeighborSampler([10, 10])
    nodeloader = dgl.dataloading.DataLoader(
        g,
        torch.arange(N).to(device),
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        device=device,
    )

    year_XY = {}
    l = args.length

    for year in range(min_year + l - 1, max_year + 1):
        X_seqs   = []
        Y_seqs   = []
        counties = []
        for county in county_set:
            X_seq = []
            Y_seq = []
            for i in range(year - l + 1, year + 1):
                X_seq.append(X_dict[county][i])
                Y_seq.append(Y_dict[county][i])
            X_seqs.append(X_seq)
            Y_seqs.append(Y_seq)
            counties.append(county)
        X_seqs   = torch.tensor(np.array(X_seqs,   dtype=np.float32))
        Y_seqs   = torch.tensor(np.array(Y_seqs,   dtype=np.float32))
        counties = torch.tensor(counties)
        year_XY[year] = (X_seqs, Y_seqs, counties)

    param_setting = (
        "gat-rnn-v2_bs-{}_lr-{}_maxepoch-{}_sche-{}_T0-{}_etamin-{}"
        "_step-{}_gamma-{}_dropout-{}_sleep-{}_testyear-{}"
        "_encoder-{}_len-{}_weightdecay-{}_seed-{}"
    ).format(
        args.batch_size, args.learning_rate, args.max_epoch,
        args.scheduler, args.T0, args.eta_min, args.lrsteps,
        args.gamma, args.dropout, args.sleep, args.test_year,
        getattr(args, 'encoder_type', 'hybrid'), args.length,
        args.weight_decay, args.seed
    )

    summary_dir  = 'summary/{}/{}/{}'.format(args.dataset, args.test_year, param_setting)
    model_dir    = 'model/{}/{}/{}'.format(args.dataset, args.test_year, param_setting)
    results_dir  = 'results/{}/{}/{}'.format(args.dataset, args.test_year, param_setting)
    build_path(summary_dir)
    build_path(model_dir)
    build_path(results_dir)
    writer = SummaryWriter(log_dir=summary_dir)

    results_summary_file = 'results/{}/{}/results_summary.csv'.format(
        args.dataset, args.test_year)
    if not os.path.isfile(results_summary_file):
        with open(results_summary_file, mode='w') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([
                'dataset', 'model', 'git_commit', 'command',
                'val_year', 'val_rmse', 'val_r2', 'val_corr',
                'test_year', 'test_rmse', 'test_r2', 'test_corr', 'path_to_model'
            ])
    git_commit     = get_git_revision_hash()
    command_string = " ".join(sys.argv)

    # Build model
    print('building network...')
    first_year = min(year_XY.keys())
    in_dim  = year_XY[first_year][0].shape[-1]
    out_dim = len(args.output_names)

    # Load distance weights BEFORE building model so edge_feat_dim is correct
    dist_pkl = getattr(args, 'dist_weights_path', '')
    if dist_pkl and os.path.exists(dist_pkl):
        import pickle as _pkl
        with open(dist_pkl, 'rb') as _f:
            args.dist_weights = _pkl.load(_f)
        _sample = next(iter(args.dist_weights.values()))
        _arr    = np.asarray(_sample)
        args.edge_feat_dim = int(_arr.shape[0]) if _arr.ndim >= 1 else 1
        print(f"  Loaded {len(args.dist_weights)} distance-weighted edges from {dist_pkl}"
              f"  (edge_feat_dim={args.edge_feat_dim})")
    else:
        args.dist_weights  = None
        args.edge_feat_dim = getattr(args, 'edge_feat_dim', 3)
        print(f"  No dist_weights_path provided — using uniform attention fallback "
              f"(edge_feat_dim={args.edge_feat_dim})")

    model = GAT_RNN_V2(args, in_dim, out_dim).to(device)
    print(f"Using GAT_RNN_V2: in_dim={in_dim}, out_dim={out_dim}, "
          f"edge_feat_dim={args.edge_feat_dim}")

    # ── Bayesian species rarity weights ───────────────────────────────────────
    if getattr(args, 'use_bayes', True):
        all_Y_train = []
        train_years = [y for y in year_XY if y < args.test_year - 1]
        for yr in train_years:
            _, Y_yr, _ = year_XY[yr]
            all_Y_train.append(Y_yr.reshape(-1, Y_yr.shape[-1]))
        if all_Y_train:
            Y_all = torch.cat(all_Y_train, dim=0)
            nonzero_frac = (Y_all > 0).float().mean(0).clamp(min=1e-3)
            rarity_w = (1.0 / nonzero_frac)
            rarity_w = rarity_w / rarity_w.mean()
            args.species_rarity_weights = rarity_w
            print(f"  [Bayes] Species rarity weights (mean=1): "
                  f"min={rarity_w.min():.2f}  max={rarity_w.max():.2f}")
        else:
            args.species_rarity_weights = None

    # ── Bayesian transfer learning: informative prior from donor species ──────
    transfer_sp = getattr(args, 'transfer_prior_species', '')
    if transfer_sp and getattr(args, 'use_bayes', True) and hasattr(model, 'set_transfer_priors'):
        sp_names = args.output_names
        donor_names = [s.strip() for s in transfer_sp.split(',') if s.strip()]
        donor_indices = []
        for dn in donor_names:
            idx = next((i for i, n in enumerate(sp_names) if dn in n), None)
            if idx is None:
                print(f"  [BayesTransfer] WARNING: '{dn}' not found in species list, skipping.")
            else:
                donor_indices.append(idx)

        if donor_indices:
            rare_thresh  = getattr(args, 'transfer_rare_threshold', 0.15)
            all_Y_train  = []
            for yr in [y for y in year_XY if y < args.test_year - 1]:
                _, Y_yr, _ = year_XY[yr]
                all_Y_train.append(Y_yr.reshape(-1, Y_yr.shape[-1]))
            if all_Y_train:
                Y_all        = torch.cat(all_Y_train, dim=0)
                nonzero_frac = (Y_all > 0).float().mean(0).clamp(min=1e-3)
            else:
                nonzero_frac = torch.ones(len(sp_names)) * 0.5
            rare_mask = (nonzero_frac < rare_thresh)
            for di in donor_indices:
                rare_mask[di] = False
            donor_arg = donor_indices[0] if len(donor_indices) == 1 else donor_indices
            model.set_transfer_priors(
                donor_idx            = donor_arg,
                rare_mask            = rare_mask,
                transfer_prior_std   = getattr(args, 'transfer_prior_std', 0.3),
                warmstart_alpha      = getattr(args, 'transfer_warmstart_alpha', 0.5),
            )

    # ── Ecological cross-donor transfer priors ────────────────────────────────
    if getattr(args, 'use_cross_donors', False) and \
            getattr(args, 'use_bayes', True) and \
            hasattr(model, 'set_transfer_priors_mapped'):
        _sp_names = getattr(args, 'output_names', [])

        def _idx(fragment):
            return [i for i, n in enumerate(_sp_names) if fragment in n]

        CROSS_DONOR_RULES = [
            ('Centropomus undecimalis_A',   ['Cynoscion nebulosus_R', 'Cynoscion nebulosus_A']),
            ('Centropomus undecimalis_SA',  ['Cynoscion nebulosus_R', 'Centropomus undecimalis_A']),
            ('Sciaenops ocellatus_R',       ['Cynoscion nebulosus_R', 'Cynoscion nebulosus_A']),
            ('Sciaenops ocellatus_SA',      ['Cynoscion nebulosus_R', 'Cynoscion nebulosus_A']),
            ('Cynoscion nebulosus_A',       ['Cynoscion nebulosus_R', 'Lagodon rhomboides_A']),
            ('Mycteroperca microlepis_SA',  ['Lutjanus griseus_SA',   'Lutjanus griseus_R']),
            ('Lutjanus griseus_R',          ['Archosargus probatocephalus_A', 'Lutjanus griseus_SA']),
            ('Lutjanus griseus_SA',         ['Archosargus probatocephalus_A']),
            ('Archosargus probatocephalus_R', ['Archosargus probatocephalus_A', 'Lagodon rhomboides_A']),
        ]

        cross_map = {}
        for rec_frag, donor_frags in CROSS_DONOR_RULES:
            rec_indices   = _idx(rec_frag)
            donor_indices_all = []
            for df_frag in donor_frags:
                donor_indices_all.extend(_idx(df_frag))
            donor_indices_all = list(dict.fromkeys(
                d for ri in rec_indices for d in donor_indices_all if d not in rec_indices
            ))
            for ri in rec_indices:
                if donor_indices_all:
                    cross_map[ri] = donor_indices_all

        if cross_map:
            print(f"  [CrossDonor] Ecological cross-donor map ({len(cross_map)} recipients):")
            for ri, di in cross_map.items():
                r_name = _sp_names[ri] if ri < len(_sp_names) else ri
                d_names = [_sp_names[d] if d < len(_sp_names) else d for d in di]
                print(f"    {r_name}  <-  {d_names}")
            model.set_transfer_priors_mapped(
                donor_map          = cross_map,
                transfer_prior_std = getattr(args, 'transfer_prior_std', 0.3),
                warmstart_alpha    = getattr(args, 'transfer_warmstart_alpha', 0.5),
            )

    # ── MAML / Reptile trunk initialisation ───────────────────────────────────
    trunk_ckpt = getattr(args, 'trunk_ckpt', '')
    if trunk_ckpt and os.path.exists(trunk_ckpt):
        load_trunk_weights(model, trunk_ckpt)
        if getattr(args, 'freeze_trunk', False):
            freeze_trunk(model)
    elif trunk_ckpt:
        print(f"  [MAML] WARNING: trunk_ckpt not found: {trunk_ckpt}")

    # Optimiser and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, eta_min=args.eta_min, T_0=args.T0, T_mult=args.T_mult)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, args.lrsteps, args.gamma)
    elif args.scheduler == "const":
        scheduler = None
    else:
        raise ValueError(f"scheduler '{args.scheduler}' not supported")

    best_val_rmse = 1e9
    cur_step      = 0

    train_rmse_list, val_rmse_list, test_rmse_list = [], [], []
    train_r2_list, val_r2_list, test_r2_list       = [], [], []

    best_model_path = os.path.join(model_dir, 'model-0')  # initialise

    for epoch in range(args.max_epoch):
        epoch_start = time()

        cur_step, train_metrics, train_results = train_epoch_2D(
            args, model, device, nodeloader, year_XY, county_avg,
            year_avg_Y, cur_step, optimizer, epoch, writer)
        val_metrics, val_results = val_epoch_2D(
            args, model, device, nodeloader, year_XY, county_avg,
            year_avg_Y, epoch, "Val", writer)
        test_metrics, test_results = val_epoch_2D(
            args, model, device, nodeloader, year_XY, county_avg,
            year_avg_Y, epoch, "Test", writer)

        epoch_time = time() - epoch_start
        print("Epoch finished in", epoch_time)

        val_rmse = val_metrics['rmse']
        train_rmse_list.append(train_metrics['rmse'])
        val_rmse_list.append(val_metrics['rmse'])
        test_rmse_list.append(test_metrics['rmse'])
        train_r2_list.append(train_metrics['r2'])
        val_r2_list.append(val_metrics['r2'])
        test_r2_list.append(test_metrics['r2'])

        if val_rmse < best_val_rmse:
            update_metrics(val_metrics['rmse'], val_metrics['r2'], val_metrics['corr'], "Val")
            update_metrics(test_metrics['rmse'], test_metrics['r2'], test_metrics['corr'], "Test")
            best_val_rmse = val_rmse

            best_model_path = os.path.join(model_dir, f'model-{epoch}')
            torch.save(model.state_dict(), best_model_path)
            print('save model to', best_model_path)

            train_results.to_csv(os.path.join(results_dir, "train_results.csv"),  index=False)
            val_results.to_csv(  os.path.join(results_dir, "val_results.csv"),    index=False)
            test_results.to_csv( os.path.join(results_dir, "test_results.csv"),   index=False)

            # Save attention weights (per-species × per-token)
            if (hasattr(model, 'last_attn_weights') and model.last_attn_weights is not None):
                aw = model.last_attn_weights.cpu().numpy()  # [n_src, n_species, n_tokens]
                np.save(os.path.join(results_dir, "last_attn_weights.npy"), aw)
                # Mean over stations: [n_species, n_tokens]
                aw_mean = aw.mean(axis=0)
                n_species, n_tokens = aw_mean.shape
                sp_names = args.output_names
                rows = []
                for si in range(n_species):
                    for ti in range(n_tokens):
                        rows.append({
                            'species':     sp_names[si] if si < len(sp_names) else si,
                            'token_idx':   ti,
                            'mean_weight': float(aw_mean[si, ti]),
                        })
                attn_df = pd.DataFrame(rows)
                attn_df.to_csv(os.path.join(results_dir, "attention_weights.csv"), index=False)
                print(f"  Saved attention weights: [{n_species} species × {n_tokens} tokens]")

            with open(os.path.join(results_dir, "summary_current.txt"), 'w') as f:
                f.write("Algorithm: GAT_RNN_V2\n")
                f.write("Dataset: " + args.dataset + "\n")
                f.write("Git commit: " + git_commit + "\n")
                f.write("Command: " + command_string + "\n")
                f.write("Final model path: " + best_model_path + "\n\n")
                f.write("BEST Val ({}) | rmse: {}, r2: {}, corr: {}\n".format(
                    args.test_year - 1, best_val['rmse'], best_val['r2'], best_val['corr']))
                f.write("BEST Test ({}) | rmse: {}, r2: {}, corr: {}\n".format(
                    args.test_year, best_test['rmse'], best_test['r2'], best_test['corr']))

        # Epoch plots (every 5 epochs or last epoch)
        if _HAS_MPL and (epoch % 5 == 0 or epoch == args.max_epoch - 1):
            epoch_list = range(len(train_rmse_list))
            plt.figure()
            plt.plot(epoch_list, train_rmse_list, color='blue',
                     label=f'Train RMSE ({min_year}-{args.test_year-2})')
            plt.plot(epoch_list, val_rmse_list,   color='orange',
                     label=f'Val RMSE ({args.test_year-1})')
            plt.plot(epoch_list, test_rmse_list,  color='red',
                     label=f'Test RMSE ({args.test_year})')
            plt.legend(); plt.xlabel('Epoch'); plt.ylabel('RMSE')
            plt.savefig(os.path.join(results_dir, "metrics_rmse.png"))
            plt.close()

            plt.figure()
            plt.plot(epoch_list, train_r2_list, color='blue',  label='Train R²')
            plt.plot(epoch_list, val_r2_list,   color='orange', label='Val R²')
            plt.plot(epoch_list, test_r2_list,  color='red',   label='Test R²')
            plt.legend(); plt.xlabel('Epoch'); plt.ylabel('R²'); plt.ylim(0, 1)
            plt.savefig(os.path.join(results_dir, "metrics_r2.png"))
            plt.close()

        print("BEST Val | rmse: {}, r2: {}, corr: {}".format(
            best_val['rmse'], best_val['r2'], best_val['corr']))
        print("BEST Test | rmse: {}, r2: {}, corr: {}".format(
            best_test['rmse'], best_test['r2'], best_test['corr']))
        print("Model path:", best_model_path)

        if scheduler is not None and epoch < args.sleep - 1:
            scheduler.step()

        # Refresh cross-donor prior means each epoch
        if hasattr(model, 'refresh_transfer_priors'):
            model.refresh_transfer_priors()

    # Final record
    print("Command used:", command_string)
    print("Model dir:", model_dir)
    with open(results_summary_file, mode='a+') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([
            args.dataset, 'GAT_RNN_V2', git_commit, command_string,
            str(args.test_year - 1), best_val['rmse'], best_val['r2'], best_val['corr'],
            str(args.test_year), best_test['rmse'], best_test['r2'], best_test['corr'],
            best_model_path
        ])

    with open(os.path.join(results_dir, "summary_FINAL.txt"), 'w') as f:
        f.write("Algorithm: GAT_RNN_V2\n")
        f.write("Dataset: " + args.dataset + "\n")
        f.write("Git commit: " + git_commit + "\n")
        f.write("Command: " + command_string + "\n")
        f.write("Final model path: " + best_model_path + "\n\n")
        f.write("BEST Val ({}) | rmse: {}, r2: {}, corr: {}\n".format(
            args.test_year - 1, best_val['rmse'], best_val['r2'], best_val['corr']))
        f.write("BEST Test ({}) | rmse: {}, r2: {}, corr: {}\n".format(
            args.test_year, best_test['rmse'], best_test['r2'], best_test['corr']))

    # Post-hoc calibration
    try:
        val_df_path  = os.path.join(results_dir, "val_results.csv")
        test_df_path = os.path.join(results_dir, "test_results.csv")
        if os.path.exists(val_df_path) and os.path.exists(test_df_path):
            val_df  = pd.read_csv(val_df_path)
            test_df = pd.read_csv(test_df_path)
            sp_metrics, mean_cal_corr, _ = calibrate_species(
                val_df, test_df, args.output_names, results_dir)
            roc_vals = [m['roc_auc'] for m in sp_metrics.values()
                        if not np.isnan(m.get('roc_auc', float('nan')))]
            pr_vals  = [m['pr_auc']  for m in sp_metrics.values()
                        if not np.isnan(m.get('pr_auc',  float('nan')))]
            mean_roc = float(np.mean(roc_vals)) if roc_vals else float('nan')
            mean_pr  = float(np.mean(pr_vals))  if pr_vals  else float('nan')
        else:
            mean_cal_corr = best_test['corr']
            mean_roc = mean_pr = float('nan')
    except Exception as e:
        print(f"  [calibration] failed: {e}")
        mean_cal_corr = best_test['corr']
        mean_roc = mean_pr = float('nan')

    return {
        'rmse':      best_test['rmse'],
        'r2':        best_test['r2'],
        'corr':      best_test['corr'],
        'cal_corr':  mean_cal_corr,
        'roc_auc':   mean_roc,
        'pr_auc':    mean_pr,
    }


def train_rolling(args):
    """
    Walk-forward rolling window cross-validation for GAT_RNN_V2.

    For each test_year in [rolling_start, rolling_end]:
      - Train on the 5 years immediately before val_year
          i.e. train_start = test_year - train_window - 1
      - Val  on test_year - 1
      - Test on test_year

    Example with train_window=5, length=5:
      test_year=2007: train 2001-2005, val 2006, test 2007
      test_year=2008: train 2002-2006, val 2007, test 2008
      ...
      test_year=2024: train 2018-2022, val 2023, test 2024

    Each prediction uses only data from the prior window (no future leakage).

    Results saved to:
      results/{dataset}/{test_year}/{param_setting}/  (per-year, same as single run)
      results/{dataset}/rolling/rolling_results.csv   (aggregated)
      results/{dataset}/rolling/rolling_metrics.png
    """
    import copy

    rolling_start = getattr(args, 'rolling_start', 2007)
    rolling_end   = getattr(args, 'rolling_end',   2024)
    train_window  = getattr(args, 'train_window',  5)

    rolling_dir = 'results/{}/rolling'.format(args.dataset)
    build_path(rolling_dir)

    all_rows = []

    for test_year in range(rolling_start, rolling_end + 1):
        # Each prediction window: prior train_window years for training,
        # then val on test_year-1, test on test_year
        train_start = test_year - train_window - 1
        val_year    = test_year - 1

        print("\n" + "=" * 60)
        print(f"Rolling window  test_year={test_year}")
        print(f"  Train : {train_start} – {test_year - 2}  ({train_window} yrs)")
        print(f"  Val   : {val_year}")
        print(f"  Test  : {test_year}")
        print("=" * 60)

        # Clone args and set window-specific fields
        window_args = copy.deepcopy(args)
        window_args.test_year        = test_year
        window_args.train_start_year = train_start

        # Reset global best trackers for this window
        global best_val, best_test
        best_val  = {'rmse': 1e9, 'r2': -1e9, 'corr': -1e9}
        best_test = {'rmse': 1e9, 'r2': -1e9, 'corr': -1e9}

        test_metrics = train(window_args)

        all_rows.append({
            'test_year'   : test_year,
            'train_start' : train_start,
            'val_year'    : val_year,
            'test_rmse'   : test_metrics['rmse'],
            'test_r2'     : test_metrics['r2'],
            'test_corr'   : test_metrics['corr'],
            'cal_corr'    : test_metrics.get('cal_corr',  float('nan')),
            'test_roc_auc': test_metrics.get('roc_auc',  float('nan')),
            'test_pr_auc' : test_metrics.get('pr_auc',   float('nan')),
        })
        print(f"Window done — test R²={test_metrics['r2']:.4f}  "
              f"corr={test_metrics['corr']:.4f}  "
              f"ROC-AUC={test_metrics.get('roc_auc', float('nan')):.4f}")

    # ── Save aggregated results ────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(rolling_dir, 'rolling_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nRolling CV results saved to {csv_path}")
    print(df.to_string(index=False))

    # ── Plot rolling metrics ───────────────────────────────────────────────────
    if _HAS_MPL:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        years = df['test_year'].values

        plot_metrics = [
            ('test_rmse',    'RMSE (count scale)',  'crimson'),
            ('test_r2',      'R²',                  'steelblue'),
            ('test_corr',    'Pearson Correlation',  'seagreen'),
            ('test_roc_auc', 'ROC-AUC (presence)',   'darkorange'),
        ]
        for ax, (metric, label, color) in zip(axes, plot_metrics):
            vals = df[metric].values if metric in df.columns else np.full(len(years), np.nan)
            valid = ~np.isnan(vals.astype(float))
            ax.plot(years[valid], vals[valid], marker='o', color=color, linewidth=2)
            if valid.any():
                ax.axhline(float(np.nanmean(vals)), color=color, linestyle='--', alpha=0.5,
                           label=f'Mean = {float(np.nanmean(vals)):.3f}')
            ax.set_title(label)
            ax.set_xlabel('Test Year')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(years[::2])
            ax.set_xticklabels(years[::2], rotation=45)

        fig.suptitle(
            f'GAT-RNN-V2 Walk-Forward Rolling CV  '
            f'(train_window={train_window} yrs, test {rolling_start}–{rolling_end})',
            fontsize=13
        )
        plt.tight_layout()
        fig_path = os.path.join(rolling_dir, 'rolling_metrics.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Rolling metrics plot saved to {fig_path}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Windowed (sliding-window) data pipeline and training loop
# ══════════════════════════════════════════════════════════════════════════════
#
# Architecture overview
# ---------------------
# Instead of feeding the model a fixed sequence of N calendar years, we build
# 10 overlapping windows of `win_size` years each for every prediction year T:
#
#   window w  (w = 1 … n_windows):
#       year_end   = T - (n_windows - w)      # latest year in this window
#       year_start = year_end - win_size + 1   # earliest year in this window
#
# For each window the observations across its `win_size` years are *averaged*
# per station per month, yielding a single [12, n_features] / [12, n_species]
# summary slice.  Missing station-years are zero-filled before averaging.
#
# The model then receives a sequence of 10 such slices (one per window) instead
# of the old sequence of N calendar years, so the tensor shapes fed to
# model.forward() are:
#
#   X : [batch, n_windows, 12, n_features]
#   Y : [batch, n_windows, 12, n_species]
#
# The inner LSTM still summarises the 12 months of each window → h_window, and
# the outer LSTM processes h_w1 … h_w10 to predict year T.  Loss is computed
# only on the final window (window index n_windows-1 == the prediction for T).
# ══════════════════════════════════════════════════════════════════════════════


def build_merged_windows(X_dict, Y_dict, avail_dict, county_set,
                         T, win_size=3, n_windows=10):
    """Build n_windows sliding-window averages for prediction year T.

    For window w (1-indexed, 1 … n_windows):
        year_end   = T - (n_windows - w)
        year_start = year_end - win_size + 1
        years      = [year_start, …, year_end]

    For each station and month, observations from the years that exist in
    X_dict are averaged; years with no data contribute a zero slice.

    Parameters
    ----------
    X_dict     : dict  station_id → {year: np.ndarray [12, n_features]}
    Y_dict     : dict  station_id → {year: np.ndarray [12, n_species]}
    avail_dict : dict  year → list[station_idx]  (not used directly but kept
                       for API symmetry with the existing train pipeline)
    county_set : list  ordered station ids (defines row order in output arrays)
    T          : int   prediction year
    win_size   : int   number of years per window (default 3)
    n_windows  : int   total windows (default 10)

    Returns
    -------
    window_XY : dict
        window_idx (1 … n_windows) →
            (X_merged, Y_merged, counties)
            X_merged : np.ndarray  [n_counties, 12, n_features]
            Y_merged : np.ndarray  [n_counties, 12, n_species]
            counties : list[int]   same order as county_set
    """
    # Infer feature / species dimensions from the first available entry
    _sample_station = county_set[0]
    _sample_year    = next(iter(X_dict[_sample_station]))
    n_feat    = X_dict[_sample_station][_sample_year].shape[1]
    n_species = Y_dict[_sample_station][_sample_year].shape[1]
    n_counties = len(county_set)

    window_XY = {}

    for w in range(1, n_windows + 1):
        year_end   = T - (n_windows - w)
        year_start = year_end - win_size + 1
        window_years = list(range(year_start, year_end + 1))

        # Accumulators: sum and count arrays for averaging
        X_sum   = np.zeros((n_counties, 12, n_feat),    dtype=np.float64)
        Y_sum   = np.zeros((n_counties, 12, n_species),  dtype=np.float64)
        counts  = np.zeros((n_counties, 1,  1),          dtype=np.float64)  # broadcast

        for ci, station in enumerate(county_set):
            s_xdict = X_dict.get(station, {})
            s_ydict = Y_dict.get(station, {})
            n_valid = 0
            for yr in window_years:
                if yr in s_xdict:
                    X_sum[ci] += s_xdict[yr]   # [12, n_feat]
                    Y_sum[ci] += s_ydict[yr]   # [12, n_species]
                    n_valid   += 1
            counts[ci, 0, 0] = n_valid

        # Average over the years that had data; leave zero where no data exists
        valid_mask = counts > 0
        X_merged = np.where(valid_mask, X_sum / np.where(counts > 0, counts, 1), 0.0)
        Y_merged = np.where(valid_mask, Y_sum / np.where(counts > 0, counts, 1), 0.0)

        window_XY[w] = (
            X_merged.astype(np.float32),   # [n_counties, 12, n_feat]
            Y_merged.astype(np.float32),   # [n_counties, 12, n_species]
            list(county_set),
        )

    return window_XY


def build_year_XY_windowed(X_dict, Y_dict, avail_dict, county_set,
                            test_year, win_size=3, n_windows=10):
    """Build windowed XY data for every prediction year up to test_year.

    The earliest prediction year T_min is the first year for which all
    n_windows windows are fully inside the available data range, i.e.:

        year_start of window 1 = T - n_windows - win_size + 2

    For example with n_windows=10, win_size=3:
        T_min satisfies  T_min - 11 >= min_data_year
        ⟹  T_min = min_data_year + 11

    Parameters
    ----------
    X_dict     : dict  station_id → {year: np.ndarray [12, n_features]}
    Y_dict     : dict  station_id → {year: np.ndarray [12, n_species]}
    avail_dict : dict  year → list[station_idx]
    county_set : list  ordered station ids
    test_year  : int   last prediction year (inclusive)
    win_size   : int   years per window (default 3)
    n_windows  : int   number of windows (default 10)

    Returns
    -------
    all_windowed : dict
        T → window_XY   (see build_merged_windows for the inner structure)
    """
    # Determine data range from X_dict
    all_years = set()
    for s in county_set:
        all_years.update(X_dict[s].keys())
    if not all_years:
        raise ValueError("X_dict contains no year data for any station in county_set")

    data_start  = min(all_years)
    # Window 1 for prediction year T starts at: T - n_windows - win_size + 2
    # We need that start >= data_start
    # ⟹  T >= data_start + n_windows + win_size - 2
    # With data_start=1996, n_windows=10, win_size=3: min_pred_year = 1996+10+3-2 = 2007
    # But window 1 for T=2008 starts at 2008-10-3+2=1997 ≥ 1996 ✓
    # Correct formula: min_pred_year = data_start + n_windows + win_size - 2
    # Check: window1_start = T - n_windows - win_size + 2
    #   T=2007: window1_start = 2007-10-3+2 = 1996 = data_start ✓ (just barely ok)
    #   T=2008: window1_start = 1997 > 1996 ✓
    min_pred_year = data_start + n_windows + win_size - 2

    print(f"[build_year_XY_windowed] data_start={data_start}, "
          f"min_pred_year={min_pred_year}, test_year={test_year}, "
          f"win_size={win_size}, n_windows={n_windows}")

    all_windowed = {}
    for T in range(min_pred_year, test_year + 1):
        all_windowed[T] = build_merged_windows(
            X_dict, Y_dict, avail_dict, county_set,
            T, win_size=win_size, n_windows=n_windows
        )
        if (T - min_pred_year) % 5 == 0:
            print(f"  built windows for T={T}")

    print(f"[build_year_XY_windowed] built windowed XY for "
          f"{len(all_windowed)} prediction years ({min_pred_year}–{test_year})")
    return all_windowed


def _build_windowed_batch(windowed_XY, T, in_nodes, out_nodes, device, n_windows=10):
    """Extract a batch of nodes from windowed_XY for prediction year T.

    Parameters
    ----------
    windowed_XY : dict  window_idx → (X_merged, Y_merged, counties)
                  as returned by build_merged_windows for a single T
    T           : int   prediction year (for logging only)
    in_nodes    : LongTensor  indices into the county_set for the source nodes
    out_nodes   : LongTensor  indices into the county_set for the destination nodes
    device      : torch.device
    n_windows   : int   number of windows expected

    Returns
    -------
    X_batch : FloatTensor  [n_out, n_windows, 12, n_features]  (src for encoder)
    Y_batch : FloatTensor  [n_out, n_windows, 12, n_species]
    X_src   : FloatTensor  [n_in,  n_windows, 12, n_features]  (full src for GNN)
    """
    in_idx  = in_nodes.cpu().numpy()
    out_idx = out_nodes.cpu().numpy()

    X_wins, Y_wins = [], []
    X_src_wins     = []

    for w in range(1, n_windows + 1):
        X_w, Y_w, _ = windowed_XY[w]            # numpy [n_counties, 12, *]
        X_wins.append(X_w)
        Y_wins.append(Y_w)
        X_src_wins.append(X_w)

    # Stack along window dimension
    X_all     = np.stack(X_wins,     axis=1)   # [n_counties, n_windows, 12, n_feat]
    Y_all     = np.stack(Y_wins,     axis=1)   # [n_counties, n_windows, 12, n_sp]
    X_src_all = np.stack(X_src_wins, axis=1)   # [n_counties, n_windows, 12, n_feat]

    X_src_t = torch.tensor(X_src_all[in_idx],  dtype=torch.float32, device=device)
    X_dst_t = torch.tensor(X_all[out_idx],     dtype=torch.float32, device=device)
    Y_dst_t = torch.tensor(Y_all[out_idx],     dtype=torch.float32, device=device)

    return X_dst_t, Y_dst_t, X_src_t


def train_epoch_windowed(args, model, device, nodeloader, windowed_XY,
                         county_avg, year_avg_Y, optimizer, scheduler,
                         epoch, writer=None, cur_step=0):
    """Training epoch using the sliding-window data representation.

    For each prediction year T in the windowed_XY dict (excluding val/test
    years) and each mini-batch of nodes, stacks the n_windows averaged slices
    into a sequence and feeds them to model.forward().  Loss is computed on
    the final window only (the prediction for year T).

    Parameters
    ----------
    args         : namespace — must contain test_year, clip_grad, check_freq,
                               means, stds, c1, c2, output_names, length,
                               no_log_transform, batch_size
    model        : GAT_RNN_V2 instance
    device       : torch.device
    nodeloader   : DGL DataLoader yielding (in_nodes, out_nodes, blocks)
    windowed_XY  : dict  T → {w: (X_merged, Y_merged, counties)}
                   as returned by build_year_XY_windowed
    county_avg   : not used directly (kept for API symmetry)
    year_avg_Y   : not used directly (kept for API symmetry)
    optimizer    : torch optimizer
    scheduler    : lr scheduler or None
    epoch        : int  current epoch number (for printing)
    writer       : SummaryWriter or None
    cur_step     : int  global step counter for TensorBoard

    Returns
    -------
    cur_step : int
    metrics  : dict  {rmse, r2, corr, …}
    results  : pd.DataFrame  per-station per-year predictions
    """
    print(f"\n----- Epoch {epoch} (windowed) -----")
    model.train()

    n_windows = int(getattr(args, 'n_windows', 10))

    # Prediction years used for training: exclude test_year; include all earlier windowed years
    # (for the earliest rolling folds, val year may also serve as a training point)
    train_pred_years = sorted([T for T in windowed_XY if T != args.test_year])
    if not train_pred_years:
        raise ValueError("No training prediction years found in windowed_XY. "
                         "Check that build_year_XY_windowed covers years up to test_year-1.")

    np.random.shuffle(train_pred_years)

    tot_loss   = 0.0
    n_batch    = 0
    all_pred   = []
    all_Y      = []
    result_dfs = []

    for T in train_pred_years:
        win_data = windowed_XY[T]   # {w: (X_merged, Y_merged, counties)}

        # Per-T calendar-based month weights (no-op if not loaded)
        _set_window_month_weights(args, T, n_windows)

        for batch_idx, (in_nodes, out_nodes, blocks) in enumerate(nodeloader):
            # ── Assemble batch tensors ────────────────────────────────────────
            # X_dst: [n_out, n_windows, 12, n_feat]  (destination / output nodes)
            # Y_dst: [n_out, n_windows, 12, n_sp]
            # X_src: [n_in,  n_windows, 12, n_feat]  (source / input  nodes for GNN)
            X_dst, Y_dst, X_src = _build_windowed_batch(
                win_data, T, in_nodes, out_nodes, device, n_windows)

            # Skip batches where the final-window target is entirely zero
            # (avoids spurious perfect-zero updates for stations with no data)
            if Y_dst[:, -1].abs().max() < 1e-9:
                continue

            # Normalise Y for the look-back input to the model
            Y_std = (Y_dst - args.means) / args.stds  # [n_out, n_windows, 12, n_sp]

            # ── Graph blocks ──────────────────────────────────────────────────
            blocks = [b.int().to(device) for b in blocks]
            _attach_dist_weights(blocks, args)

            # model.forward expects:
            #   x : [n_src, n_years+1, 12, n_feat]   (here n_years+1 = n_windows)
            #   y : [n_dst, n_years,   12, n_sp]      (look-back labels; n_years = n_windows-1)
            #
            # We use X_src as x (all source nodes for the GNN layers) and
            # Y_std[:, :-1] as the look-back y (all windows except the last).
            pred_std = model(blocks, X_src, Y_std[:, :-1])
            # pred_std: [n_dst, n_windows, 12, n_sp]

            pred = pred_std * args.stds + args.means   # back to log1p scale

            # ── Loss: intermediate windows * c1  +  final window * c2 ─────────
            # If per-window calendar weights are loaded (args.month_weights is
            # [n_windows, 12, n_sp]), slice it for each term.
            _mw = getattr(args, 'month_weights', None)
            if _mw is not None and _mw.dim() == 3:
                mw_inter = _mw[:-1]                      # [n_windows-1, 12, n_sp]
                mw_final = _mw[-1]                       # [12, n_sp]
            else:
                mw_inter = mw_final = None               # falls back to args.month_weights

            if getattr(args, 'use_hurdle', False) and model.last_presence_logits is not None:
                pl = model.last_presence_logits          # [n_dst, n_windows, 12, n_sp]
                loss = (hurdle_loss(pred[:, :-1], Y_dst[:, :-1], args,
                                    presence_logit=pl[:, :-1], month_weights=mw_inter) * args.c1 +
                        hurdle_loss(pred[:, -1],  Y_dst[:, -1],  args,
                                    presence_logit=pl[:, -1], month_weights=mw_final) * args.c2)
                # Combine for result CSVs and metric collection
                presence_prob = torch.sigmoid(pl)
                cnt_abund     = torch.expm1(torch.clamp(pred, max=30.0))
                pred = torch.log1p(torch.clamp(presence_prob * cnt_abund, min=0.0))
            else:
                loss = (loss_fn(pred[:, :-1], Y_dst[:, :-1], args,
                                month_weights=mw_inter) * args.c1 +
                        loss_fn(pred[:, -1],  Y_dst[:, -1],  args,
                                month_weights=mw_final) * args.c2)

            # Optional Bayesian KL regularisation
            beta_kl = getattr(args, 'beta_kl', 0.0)
            if beta_kl > 0 and hasattr(model, 'bayes_kl_loss'):
                rarity_w = getattr(args, 'species_rarity_weights', None)
                kl   = model.bayes_kl_loss(rarity_w) / max(X_dst.shape[0], 1)
                loss = loss + beta_kl * kl

            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            n_batch  += 1
            tot_loss += loss.item()

            # Collect final-window predictions for epoch-level metrics
            pred_final = pred[:, -1].reshape(-1, pred.shape[-1])    # [n_out, 12, n_sp] → flatten later
            Y_final    = Y_dst[:, -1].reshape(-1, Y_dst.shape[-1])  # same shape
            # Flatten month dimension into batch for eval()
            all_pred.append(pred_final.reshape(-1, pred.shape[-1]))
            all_Y.append(   Y_final.reshape(-1,    Y_dst.shape[-1]))

            if n_batch % args.check_freq == 0:
                m = eval(pred_final.reshape(-1, pred.shape[-1]),
                         Y_final.reshape(-1,   Y_dst.shape[-1]), args)
                print(f"  batch {n_batch-1} | loss {loss.item():.4f} "
                      f"rmse {m['rmse']:.3f} r2 {m['r2']:.3f} corr {m['corr']:.3f}")

            if writer is not None:
                writer.add_scalar("Train/loss_windowed", loss.item(), cur_step)
            cur_step += 1

            # ── Record per-station predictions ────────────────────────────────
            # counties: the out_nodes already map to county_set positions;
            # retrieve station ids from the first window's county list
            counties_all = win_data[1][2]  # list of station ids
            fips_np = [counties_all[i] for i in out_nodes.cpu().numpy().tolist()]

            for w_idx in range(n_windows):
                # The actual calendar year this window's year_end corresponds to
                actual_year = T - (n_windows - (w_idx + 1))
                for mo in range(12):
                    row = {
                        "fips":  fips_np,
                        "year":  [actual_year] * len(fips_np),
                        "month": [mo + 1]      * len(fips_np),
                    }
                    for i, name in enumerate(args.output_names):
                        _pred_np = pred[:, w_idx, mo, i].detach().cpu().numpy()
                        _true_np = Y_dst[:, w_idx, mo, i].detach().cpu().numpy()
                        if getattr(args, 'no_log_transform', False):
                            row[f"predicted_{name}"] = np.clip(_pred_np, 0, None).tolist()
                            row[f"true_{name}"]       = _true_np.tolist()
                        else:
                            row[f"predicted_{name}"] = np.clip(_to_count_np(_pred_np, getattr(args,'transform','log1p')), 0, None).tolist()
                            row[f"true_{name}"]       = _to_count_np(_true_np, getattr(args,'transform','log1p')).tolist()
                    result_dfs.append(pd.DataFrame(row))

    if not result_dfs:
        empty = pd.DataFrame()
        return cur_step, {'rmse': 0, 'r2': 0, 'corr': 0}, empty

    results  = pd.concat(result_dfs, ignore_index=True)
    all_pred = torch.cat(all_pred, dim=0)
    all_Y    = torch.cat(all_Y,    dim=0)
    metrics  = eval(all_pred, all_Y, args)
    print(f"\n###### Windowed train epoch {epoch} overall metrics")
    print(f"  loss: {tot_loss / max(n_batch, 1):.4f} "
          f"rmse: {metrics['rmse']:.3f} r2: {metrics['r2']:.3f} corr: {metrics['corr']:.3f}")
    return cur_step, metrics, results


def val_epoch_windowed(args, model, device, nodeloader, windowed_XY,
                       county_avg, year_avg_Y, epoch,
                       mode="Val", writer=None, n_windows=10):
    """Validation or test epoch using the sliding-window data representation.

    Parameters
    ----------
    mode : str  "Val"  → uses windowed_XY[test_year - 1]
                "Test" → uses windowed_XY[test_year]
    (all other parameters match train_epoch_windowed)

    Returns
    -------
    metrics  : dict  {rmse, r2, corr, …}
    results  : pd.DataFrame
    """
    print(f"***** Epoch {epoch} {mode} (windowed) *****")
    model.eval()

    n_windows = int(getattr(args, 'n_windows', n_windows))
    T = args.test_year - 1 if mode == "Val" else args.test_year

    if T not in windowed_XY:
        print(f"  [warn] {mode} year {T} not found in windowed_XY — skipping.")
        return {'rmse': 0, 'r2': 0, 'corr': 0}, pd.DataFrame()

    win_data = windowed_XY[T]

    # Per-T calendar-based month weights (no-op if not loaded)
    _set_window_month_weights(args, T, n_windows)

    tot_loss   = 0.0
    all_pred   = []
    all_Y      = []
    result_dfs = []
    n_batch    = 0

    with torch.no_grad():
        for batch_idx, (in_nodes, out_nodes, blocks) in enumerate(nodeloader):
            X_dst, Y_dst, X_src = _build_windowed_batch(
                win_data, T, in_nodes, out_nodes, device, n_windows)

            Y_std    = (Y_dst - args.means) / args.stds

            blocks   = [b.int().to(device) for b in blocks]
            _attach_dist_weights(blocks, args)

            pred_std = model(blocks, X_src, Y_std[:, :-1])
            pred     = pred_std * args.stds + args.means

            # Use the final-window slice of per-window weights if loaded
            _mw       = getattr(args, 'month_weights', None)
            mw_final  = _mw[-1] if (_mw is not None and _mw.dim() == 3) else None

            # Hurdle inference: pred_final_combined = sigmoid(p) * abundance
            if getattr(args, 'use_hurdle', False) and model.last_presence_logits is not None:
                pl_full = model.last_presence_logits           # [n_dst, n_windows, 12, n_sp]
                loss = hurdle_loss(pred[:, -1], Y_dst[:, -1], args,
                                   presence_logit=pl_full[:, -1], month_weights=mw_final)
                # Combine: log1p(sigmoid(p) * expm1(abund))
                presence_prob = torch.sigmoid(pl_full)
                cnt_abund     = torch.expm1(torch.clamp(pred, max=30.0))
                pred = torch.log1p(torch.clamp(presence_prob * cnt_abund, min=0.0))
            else:
                loss = loss_fn(pred[:, -1], Y_dst[:, -1], args, month_weights=mw_final)

            tot_loss += loss.item()
            n_batch  += 1

            pred_final = pred[:, -1].reshape(-1, pred.shape[-1])
            Y_final    = Y_dst[:, -1].reshape(-1, Y_dst.shape[-1])
            all_pred.append(pred_final)
            all_Y.append(   Y_final)

            counties_all = win_data[1][2]
            fips_np = [counties_all[i] for i in out_nodes.cpu().numpy().tolist()]

            for w_idx in range(n_windows):
                actual_year = T - (n_windows - (w_idx + 1))
                for mo in range(12):
                    row = {
                        "fips":  fips_np,
                        "year":  [actual_year] * len(fips_np),
                        "month": [mo + 1]      * len(fips_np),
                    }
                    for i, name in enumerate(args.output_names):
                        _pred_np = pred[:, w_idx, mo, i].cpu().numpy()
                        _true_np = Y_dst[:, w_idx, mo, i].cpu().numpy()
                        if getattr(args, 'no_log_transform', False):
                            row[f"predicted_{name}"] = np.clip(_pred_np, 0, None).tolist()
                            row[f"true_{name}"]       = _true_np.tolist()
                        else:
                            row[f"predicted_{name}"] = np.clip(_to_count_np(_pred_np, getattr(args,'transform','log1p')), 0, None).tolist()
                            row[f"true_{name}"]       = _to_count_np(_true_np, getattr(args,'transform','log1p')).tolist()
                    result_dfs.append(pd.DataFrame(row))

    if not result_dfs:
        return {'rmse': 0, 'r2': 0, 'corr': 0}, pd.DataFrame()

    results  = pd.concat(result_dfs, ignore_index=True)
    all_pred = torch.cat(all_pred, dim=0)
    all_Y    = torch.cat(all_Y,    dim=0)
    metrics  = eval(all_pred, all_Y, args)

    print(f"  loss: {tot_loss / max(n_batch, 1):.4f} "
          f"rmse: {metrics['rmse']:.3f} r2: {metrics['r2']:.3f} "
          f"corr: {metrics['corr']:.3f}")
    print("********************")

    if writer is not None:
        writer.add_scalar(f"{mode}/rmse_windowed", metrics['rmse'], epoch)
        writer.add_scalar(f"{mode}/r2_windowed",   metrics['r2'],   epoch)
        writer.add_scalar(f"{mode}/corr_windowed", metrics['corr'], epoch)

    return metrics, results


def test_epoch_windowed(args, model, device, nodeloader, windowed_XY,
                        county_avg, year_avg_Y, epoch, writer=None, n_windows=10):
    """Test epoch — thin alias for val_epoch_windowed with mode='Test'."""
    return val_epoch_windowed(
        args, model, device, nodeloader, windowed_XY,
        county_avg, year_avg_Y, epoch,
        mode="Test", writer=writer, n_windows=n_windows,
    )


def _setup_bayesian_transfer_for_windowed(args, model, windowed_XY):
    """Replicate the Bayesian transfer learning setup that `train()` runs at
    startup, but for the rolling-windowed pipeline:

      1. Compute per-species `species_rarity_weights` from the actual nonzero
         fractions across training-year windows. Used by `model.bayes_kl_loss`
         to up-weight KL on rare species (stronger shrinkage to prior).
      2. If `--use_cross_donors` is set, install a hardcoded ecological cross-
         donor prior map onto the variational output heads via
         `model.set_transfer_priors_mapped()`.

    Both are no-ops unless `--use_bayes` and `--use_film` are True (defaults).
    """
    if not (getattr(args, 'use_bayes', True) and getattr(args, 'use_film', True)):
        return

    sp_names = getattr(args, 'output_names', [])
    n_sp     = len(sp_names)
    if n_sp == 0:
        return

    # ── 1. Species rarity weights from training-year windows ──────────────────
    train_pred_years = [T for T in windowed_XY if T < args.test_year - 1]
    if train_pred_years:
        all_Y = []
        for T in train_pred_years:
            for w in windowed_XY[T].values():
                _, Y_w, _ = w
                all_Y.append(Y_w.reshape(-1, Y_w.shape[-1]))
        if all_Y:
            Y_cat = np.concatenate(all_Y, axis=0)
            nz = (Y_cat > 0).mean(axis=0)
            nz = np.clip(nz, 1e-3, None)            # avoid div by 0
            inv_w = 1.0 / nz
            inv_w = inv_w * (n_sp / inv_w.sum())   # normalize so mean = 1
            args.species_rarity_weights = torch.tensor(
                inv_w.astype(np.float32), device=next(model.parameters()).device
            )
            print(f"  [Bayes] Species rarity weights (mean=1): "
                  f"min={inv_w.min():.2f} max={inv_w.max():.2f}  "
                  f"(rarest={sp_names[int(np.argmax(inv_w))]})")
        else:
            args.species_rarity_weights = None
    else:
        args.species_rarity_weights = None

    # ── 2. Ecological cross-donor priors ──────────────────────────────────────
    if not getattr(args, 'use_cross_donors', False):
        return
    if not hasattr(model, 'set_transfer_priors_mapped'):
        print("  [CrossDonor] Model lacks set_transfer_priors_mapped — skipping")
        return

    def _idx(fragment):
        return [i for i, n in enumerate(sp_names) if fragment in n]

    # Same hardcoded ecological pairings as `train()`
    CROSS_DONOR_RULES = [
        ('Centropomus undecimalis_A',     ['Cynoscion nebulosus_R', 'Cynoscion nebulosus_A']),
        ('Centropomus undecimalis_SA',    ['Cynoscion nebulosus_R', 'Centropomus undecimalis_A']),
        ('Sciaenops ocellatus_R',         ['Cynoscion nebulosus_R', 'Cynoscion nebulosus_A']),
        ('Sciaenops ocellatus_SA',        ['Cynoscion nebulosus_R', 'Cynoscion nebulosus_A']),
        ('Cynoscion nebulosus_A',         ['Cynoscion nebulosus_R', 'Lagodon rhomboides_A']),
        ('Mycteroperca microlepis_SA',    ['Lutjanus griseus_SA',   'Lutjanus griseus_R']),
        ('Lutjanus griseus_R',            ['Archosargus probatocephalus_A', 'Lutjanus griseus_SA']),
        ('Lutjanus griseus_SA',           ['Archosargus probatocephalus_A']),
        ('Archosargus probatocephalus_R', ['Archosargus probatocephalus_A', 'Lagodon rhomboides_A']),
    ]

    cross_map = {}
    for rec_frag, donor_frags in CROSS_DONOR_RULES:
        rec_indices = _idx(rec_frag)
        donor_indices = []
        for df_frag in donor_frags:
            donor_indices.extend(_idx(df_frag))
        # Dedup, drop self-references
        for ri in rec_indices:
            valid_donors = list(dict.fromkeys(d for d in donor_indices if d != ri))
            if valid_donors:
                cross_map[ri] = valid_donors

    if cross_map:
        print(f"  [CrossDonor] Ecological cross-donor map ({len(cross_map)} recipients):")
        for ri, di in cross_map.items():
            r_name  = sp_names[ri] if ri < len(sp_names) else ri
            d_names = [sp_names[d] if d < len(sp_names) else d for d in di]
            print(f"    {r_name}  <-  {d_names}")
        model.set_transfer_priors_mapped(
            donor_map          = cross_map,
            transfer_prior_std = getattr(args, 'transfer_prior_std', 0.3),
            warmstart_alpha    = getattr(args, 'transfer_warmstart_alpha', 0.5),
        )
    else:
        print("  [CrossDonor] No matching donors found in target list — no priors installed")


def train_rolling_windowed(args):
    """Walk-forward rolling CV using the sliding-window averaged input pipeline.

    For each prediction year T in [rolling_start, rolling_end]:

      - All 10 windows for T and for val year T-1 are built from averaged
        win_size-year slices.  The outermost window covers up to T-1, so the
        model never sees data from T itself during training.
      - Training years: all T' in windowed_XY where T' < T-1
      - Val  year: T-1
      - Test year: T

    Window schedule (n_windows=10, win_size=3):
        earliest window 1 for T starts at  T - 12
        ⟹ data from at least year  T - 12  must exist
        ⟹ minimum T = data_start + 11

    Results saved to:
        results/{dataset}/{test_year}/{param_setting}/   (per-year)
        results/{dataset}/rolling_windowed/rolling_windowed_results.csv
        results/{dataset}/rolling_windowed/rolling_windowed_metrics.png

    Args controlled by args:
        rolling_start : int (default 2008)
        rolling_end   : int (default 2024)
        win_size      : int window size in years (default 3)
        n_windows     : int number of windows (default 10)
        max_epoch     : int epochs per fold (default 30)
        All other args match the standard train() call.
    """
    import copy

    rolling_start = int(getattr(args, 'rolling_start', 2008))
    rolling_end   = int(getattr(args, 'rolling_end',   2024))
    win_size      = int(getattr(args, 'win_size',       3))
    n_windows_arg = int(getattr(args, 'n_windows',      10))

    rolling_dir = f'results/{args.dataset}/rolling_windowed'
    build_path(rolling_dir)

    all_rows = []

    for test_year in range(rolling_start, rolling_end + 1):
        val_year = test_year - 1

        print("\n" + "=" * 64)
        print(f"Rolling-windowed  test_year={test_year}")
        print(f"  Val  : {val_year}")
        print(f"  Test : {test_year}")
        print(f"  win_size={win_size}  n_windows={n_windows_arg}")
        print("=" * 64)

        window_args = copy.deepcopy(args)
        window_args.test_year       = test_year
        window_args.win_size        = win_size
        window_args.n_windows       = n_windows_arg
        window_args.train_start_year = None  # use all history; windowed builder handles the window logic

        # ── Data loading (same as train()) ────────────────────────────────────
        global device
        device = _resolve_device(getattr(window_args, 'device', 'auto'))

        raw_data = np.load(window_args.data_dir)
        data     = raw_data['data']
        window_args.nested = True

        (X_dict, Y_dict, avail_dict, adj, order_map,
         min_year, max_year, county_set,
         county_avg, year_avg_Y) = get_X_Y_2D(data, window_args, device)

        for year in year_avg_Y:
            year_avg_Y[year] = torch.tensor(year_avg_Y[year], device=device)

        # ── Build windowed XY for all prediction years ─────────────────────────
        windowed_XY = build_year_XY_windowed(
            X_dict, Y_dict, avail_dict, county_set,
            test_year=test_year,
            win_size=win_size,
            n_windows=n_windows_arg,
        )

        if test_year not in windowed_XY:
            print(f"  [skip] test_year={test_year} has no windowed data "
                  "(not enough history). Increase rolling_start.")
            continue

        # ── Graph setup ───────────────────────────────────────────────────────
        sp_adj = sp.coo_matrix(adj)
        g      = dgl.from_scipy(sp_adj).to(device)
        g      = dgl.add_self_loop(g)

        N = adj.shape[0]
        sampler    = dgl.dataloading.MultiLayerNeighborSampler([10, 10])
        nodeloader = dgl.dataloading.DataLoader(
            g, torch.arange(N).to(device), sampler,
            batch_size=window_args.batch_size,
            shuffle=True, drop_last=False, num_workers=0, device=device,
        )

        # ── Model ─────────────────────────────────────────────────────────────
        # Infer in_dim / out_dim from a sample window
        _X_sample = windowed_XY[test_year][1][0]  # [n_counties, 12, n_feat]
        in_dim  = _X_sample.shape[-1]
        out_dim = len(window_args.output_names)

        dist_pkl = getattr(window_args, 'dist_weights_path', '')
        if dist_pkl and os.path.exists(dist_pkl):
            import pickle as _pkl
            with open(dist_pkl, 'rb') as _f:
                window_args.dist_weights = _pkl.load(_f)
            _sample = next(iter(window_args.dist_weights.values()))
            window_args.edge_feat_dim = int(np.asarray(_sample).shape[0]) if np.asarray(_sample).ndim >= 1 else 1
        else:
            window_args.dist_weights  = None
            window_args.edge_feat_dim = getattr(window_args, 'edge_feat_dim', 3)

        model = GAT_RNN_V2(window_args, in_dim, out_dim).to(device)

        # ── Bayesian transfer learning setup (mirrors `train()`) ──────────────
        # Computes per-species rarity weights for the KL term and installs an
        # ecological cross-donor prior map on the variational output heads.
        _setup_bayesian_transfer_for_windowed(window_args, model, windowed_XY)

        # ── Optimizer / scheduler ─────────────────────────────────────────────
        optimizer = optim.Adam(
            model.parameters(),
            lr=window_args.learning_rate,
            weight_decay=getattr(window_args, 'weight_decay', 0.0),
        )
        if window_args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=window_args.T0, eta_min=window_args.eta_min)
        elif window_args.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, window_args.lrsteps, window_args.gamma)
        else:
            scheduler = None

        # ── Dirs ──────────────────────────────────────────────────────────────
        param_setting = (
            "gat-rnn-v2-windowed_bs-{}_lr-{}_maxepoch-{}_testyear-{}"
            "_win-{}_nwin-{}_seed-{}"
        ).format(
            window_args.batch_size, window_args.learning_rate,
            window_args.max_epoch, test_year,
            win_size, n_windows_arg, window_args.seed,
        )
        results_dir = f'results/{window_args.dataset}/{test_year}/{param_setting}'
        model_dir   = f'model/{window_args.dataset}/{test_year}/{param_setting}'
        build_path(results_dir)
        build_path(model_dir)
        writer = SummaryWriter(
            log_dir=f'summary/{window_args.dataset}/{test_year}/{param_setting}')

        # ── Per-fold training loop ─────────────────────────────────────────────
        global best_val, best_test
        best_val   = {'rmse': 1e9, 'r2': -1e9, 'corr': -1e9}
        best_test  = {'rmse': 1e9, 'r2': -1e9, 'corr': -1e9}

        best_val_rmse  = 1e9
        best_model_path = os.path.join(model_dir, 'model-0')
        cur_step        = 0

        # Early stopping
        early_stop_patience  = int(getattr(window_args, 'early_stop_patience', 0) or 0)
        early_stop_min_delta = float(getattr(window_args, 'early_stop_min_delta', 1e-4) or 1e-4)
        epochs_since_best    = 0

        for epoch in range(window_args.max_epoch):
            epoch_start = time()

            cur_step, train_metrics, train_results = train_epoch_windowed(
                window_args, model, device, nodeloader, windowed_XY,
                county_avg, year_avg_Y,
                optimizer, scheduler, epoch, writer, cur_step,
            )
            val_metrics,  val_results  = val_epoch_windowed(
                window_args, model, device, nodeloader, windowed_XY,
                county_avg, year_avg_Y, epoch, mode="Val", writer=writer,
                n_windows=n_windows_arg,
            )
            test_metrics, test_results = val_epoch_windowed(
                window_args, model, device, nodeloader, windowed_XY,
                county_avg, year_avg_Y, epoch, mode="Test", writer=writer,
                n_windows=n_windows_arg,
            )

            print(f"  Epoch {epoch} done in {time() - epoch_start:.1f}s")

            improved = val_metrics['rmse'] < (best_val_rmse - early_stop_min_delta)
            if improved:
                best_val_rmse     = val_metrics['rmse']
                epochs_since_best = 0
                update_metrics(val_metrics['rmse'],  val_metrics['r2'],  val_metrics['corr'],  "Val")
                update_metrics(test_metrics['rmse'], test_metrics['r2'], test_metrics['corr'], "Test")

                best_model_path = os.path.join(model_dir, f'model-{epoch}')
                torch.save(model.state_dict(), best_model_path)
                print(f"  Saved best model → {best_model_path}")

                train_results.to_csv(os.path.join(results_dir, "train_results.csv"), index=False)
                val_results.to_csv(  os.path.join(results_dir, "val_results.csv"),   index=False)
                test_results.to_csv( os.path.join(results_dir, "test_results.csv"),  index=False)
            else:
                epochs_since_best += 1

            if scheduler is not None and epoch >= getattr(window_args, 'sleep', 0) - 1:
                scheduler.step()

            # Early stopping check
            if early_stop_patience > 0 and epochs_since_best >= early_stop_patience:
                print(f"  [EarlyStop] No val_rmse improvement for "
                      f"{epochs_since_best} epochs (patience={early_stop_patience}). "
                      f"Stopping fold at epoch {epoch}.")
                break

        writer.close()
        print(f"Fold done — best val RMSE={best_val_rmse:.4f} "
              f"test R²={best_test['r2']:.4f} corr={best_test['corr']:.4f}")

        all_rows.append({
            'test_year':  test_year,
            'val_year':   val_year,
            'win_size':   win_size,
            'n_windows':  n_windows_arg,
            'test_rmse':  best_test['rmse'],
            'test_r2':    best_test['r2'],
            'test_corr':  best_test['corr'],
        })

    # ── Aggregated results ─────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(rolling_dir, 'rolling_windowed_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nRolling-windowed CV results saved to {csv_path}")
    if not df.empty:
        print(df.to_string(index=False))

    # ── Summary plot ──────────────────────────────────────────────────────────
    if _HAS_MPL and not df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        years = df['test_year'].values
        for ax, (col, label, color) in zip(axes, [
            ('test_rmse', 'RMSE',              'crimson'),
            ('test_r2',   'R²',                'steelblue'),
            ('test_corr', 'Pearson Corr',      'seagreen'),
        ]):
            vals  = df[col].values.astype(float)
            valid = ~np.isnan(vals)
            ax.plot(years[valid], vals[valid], marker='o', color=color, linewidth=2)
            if valid.any():
                ax.axhline(float(np.nanmean(vals)), color=color, linestyle='--',
                           alpha=0.5, label=f'Mean={float(np.nanmean(vals)):.3f}')
            ax.set_title(label)
            ax.set_xlabel('Test Year')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        fig.suptitle(
            f'GAT-RNN-V2 Rolling CV (windowed, win={win_size}, n_win={n_windows_arg})',
            fontsize=13)
        plt.tight_layout()
        fig_path = os.path.join(rolling_dir, 'rolling_windowed_metrics.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Rolling-windowed metrics plot saved to {fig_path}")

    return df
