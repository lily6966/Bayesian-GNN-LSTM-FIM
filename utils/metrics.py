"""
metrics.py — Evaluation metrics for species distribution model.

Provides:
  compute_metrics(pred, true)           → dict with rmse, r2, corr, mae
  compute_species_metrics(pred_df, true_df, species_names) → DataFrame
"""

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import r2_score, mean_absolute_error as MAE
    from sklearn.metrics import roc_auc_score, average_precision_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def compute_metrics(pred, true):
    """
    Compute aggregate regression metrics over all predictions.

    Parameters
    ----------
    pred : array-like, any shape — predicted values (count scale, not log1p)
    true : array-like, same shape — ground-truth values

    Returns
    -------
    dict with keys: rmse, r2, corr, mae
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    true = np.asarray(true, dtype=np.float64).ravel()

    # Remove NaN pairs
    valid = ~(np.isnan(pred) | np.isnan(true))
    pred  = pred[valid]
    true  = true[valid]

    if len(pred) < 2:
        return {'rmse': float('nan'), 'r2': float('nan'),
                'corr': float('nan'), 'mae': float('nan')}

    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))

    if _HAS_SKLEARN:
        r2  = float(r2_score(true, pred))
        mae = float(MAE(true, pred))
    else:
        ss_res = float(np.sum((pred - true) ** 2))
        ss_tot = float(np.sum((true - np.mean(true)) ** 2))
        r2  = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
        mae = float(np.mean(np.abs(pred - true)))

    if np.std(pred) < 1e-9 or np.std(true) < 1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(pred, true)[0, 1])

    return {'rmse': rmse, 'r2': r2, 'corr': corr, 'mae': mae}


def compute_species_metrics(pred_df, true_df, species_names):
    """
    Compute per-species evaluation metrics.

    Parameters
    ----------
    pred_df : pd.DataFrame with columns 'predicted_{species}' for each species
    true_df : pd.DataFrame with columns 'true_{species}' for each species
              (or same DataFrame with both column sets)
    species_names : list[str] — species names

    Returns
    -------
    pd.DataFrame with columns:
      species, pearson, rmse, r2, roc_auc, pr_auc, prevalence
    """
    # Support passing a single merged DataFrame
    if true_df is pred_df or true_df is None:
        merged = pred_df
    else:
        # Merge on index
        merged = pred_df.join(true_df, how='inner', rsuffix='_true')

    rows = []
    for sp in species_names:
        pc = f"predicted_{sp}"
        tc = f"true_{sp}"

        if pc not in merged.columns or tc not in merged.columns:
            rows.append({
                'species':    sp,
                'pearson':    float('nan'),
                'rmse':       float('nan'),
                'r2':         float('nan'),
                'roc_auc':    float('nan'),
                'pr_auc':     float('nan'),
                'prevalence': float('nan'),
            })
            continue

        pred_arr = merged[pc].values.astype(float)
        true_arr = merged[tc].values.astype(float)

        valid = ~(np.isnan(pred_arr) | np.isnan(true_arr))
        pred_v = pred_arr[valid]
        true_v = true_arr[valid]

        n = len(pred_v)
        if n < 5:
            rows.append({
                'species':    sp,
                'pearson':    float('nan'),
                'rmse':       float('nan'),
                'r2':         float('nan'),
                'roc_auc':    float('nan'),
                'pr_auc':     float('nan'),
                'prevalence': float(np.mean(true_v > 0)) if n > 0 else float('nan'),
            })
            continue

        # Pearson correlation
        if np.std(pred_v) < 1e-9 or np.std(true_v) < 1e-9:
            pearson = 0.0
        else:
            pearson = float(np.corrcoef(pred_v, true_v)[0, 1])

        # RMSE
        rmse = float(np.sqrt(np.mean((pred_v - true_v) ** 2)))

        # R²
        if _HAS_SKLEARN:
            r2 = float(r2_score(true_v, pred_v))
        else:
            ss_res = np.sum((pred_v - true_v) ** 2)
            ss_tot = np.sum((true_v - np.mean(true_v)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

        # Prevalence (fraction of hauls with at least one individual)
        prevalence = float(np.mean(true_v > 0))

        # AUC metrics (presence/absence)
        binary_true = (true_v > 0).astype(int)
        n_pos = int(binary_true.sum())
        n_neg = n - n_pos
        if _HAS_SKLEARN and n_pos >= 2 and n_neg >= 2:
            try:
                roc_auc = float(roc_auc_score(binary_true, pred_v))
                pr_auc  = float(average_precision_score(binary_true, pred_v))
            except Exception:
                roc_auc = pr_auc = float('nan')
        else:
            roc_auc = pr_auc = float('nan')

        rows.append({
            'species':    sp,
            'pearson':    pearson,
            'rmse':       rmse,
            'r2':         r2,
            'roc_auc':    roc_auc,
            'pr_auc':     pr_auc,
            'prevalence': prevalence,
        })

    return pd.DataFrame(rows)
