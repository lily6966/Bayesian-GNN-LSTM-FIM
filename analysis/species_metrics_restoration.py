"""
Per-species metrics aggregated across all rolling CV windows for the
restoration dataset (GAT_RNN_2D + species heads + OLS calibration,
21 windows 2004-2024).

Compares raw vs calibrated test predictions side-by-side.

Outputs:
  analysis/figures/restoration/species_metrics_table.csv
  analysis/figures/restoration/species_metrics_barplot.png
  analysis/figures/restoration/species_scatter_grid.png
"""

import os, glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'gnn-rnn',
                           'results', 'FIM_restoration_stations_monthly')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures', 'restoration')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Collect test_results.csv (raw) and calibrated_test_results.csv ────────────
raw_files = glob.glob(os.path.join(RESULTS_DIR, '**', 'test_results.csv'), recursive=True)
cal_files = glob.glob(os.path.join(RESULTS_DIR, '**', 'calibrated_test_results.csv'), recursive=True)
raw_files = sorted([f for f in raw_files if 'maxepoch-10' in f])
cal_files = sorted([f for f in cal_files if 'maxepoch-10' in f])
print(f"Raw files: {len(raw_files)} | Calibrated files: {len(cal_files)}")

raw_data = pd.concat([pd.read_csv(f) for f in raw_files], ignore_index=True)
cal_data = pd.concat([pd.read_csv(f) for f in cal_files], ignore_index=True) if cal_files else None
print(f"Raw rows: {len(raw_data)} | Cal rows: {len(cal_data) if cal_data is not None else 0}")

pred_cols = [c for c in raw_data.columns if c.startswith('predicted_')]
true_cols  = [c.replace('predicted_', 'true_') for c in pred_cols]
species    = [c.replace('predicted_', '') for c in pred_cols]
print(f"Species: {len(species)}")

# ── Compute per-species metrics for a dataframe ───────────────────────────────
def compute_metrics(df, pred_cols, true_cols, species):
    records = []
    for sp, pc, tc in zip(species, pred_cols, true_cols):
        pred = df[pc].values.astype(float)
        true = df[tc].values.astype(float)
        mask = ~(np.isnan(pred) | np.isnan(true))
        pred, true = pred[mask], true[mask]
        if len(pred) < 5:
            continue
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        mae  = float(np.mean(np.abs(pred - true)))
        bias = float(np.mean(pred - true))
        ss_res = np.sum((pred - true) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        corr = float(np.corrcoef(pred, true)[0, 1]) if np.std(pred) > 1e-9 else np.nan
        suffix = sp.split('_')[-1] if '_' in sp else ''
        common = sp.rsplit('_', 1)[0] if '_' in sp else sp
        records.append({'species': sp, 'scientific_name': common, 'stage': suffix,
                        'corr': corr, 'r2': r2, 'rmse': rmse, 'mae': mae, 'bias': bias})
    return pd.DataFrame(records).sort_values('corr', ascending=False)

raw_metrics = compute_metrics(raw_data, pred_cols, true_cols, species)
cal_metrics = compute_metrics(cal_data, pred_cols, true_cols, species) if cal_data is not None else None

# Save table (raw)
raw_metrics.to_csv(os.path.join(OUT_DIR, 'species_metrics_table.csv'), index=False)

# ── Print comparison table ─────────────────────────────────────────────────────
def fmt(v): return f"{v:+.3f}" if not np.isnan(v) else "   NaN"

print("\n" + "=" * 100)
header = f"{'Species':<45} {'Stage':>5}  {'Corr(raw)':>10}  {'Corr(cal)':>10}  {'RMSE':>8}  {'Bias(raw)':>10}"
print(header)
print("=" * 100)
for _, row in raw_metrics.iterrows():
    cal_corr = np.nan
    if cal_metrics is not None:
        match = cal_metrics[cal_metrics['species'] == row['species']]
        if len(match):
            cal_corr = match['corr'].values[0]
    delta = f"  Δ={cal_corr - row['corr']:+.3f}" if not np.isnan(cal_corr) else ""
    print(f"{row['scientific_name']:<45} {row['stage']:>5}  {fmt(row['corr']):>10}  "
          f"{fmt(cal_corr):>10}{delta}  {row['rmse']:>8.2f}  {fmt(row['bias']):>10}")
print("=" * 100)

mean_raw = raw_metrics['corr'].mean()
mean_cal = cal_metrics['corr'].mean() if cal_metrics is not None else np.nan
print(f"\nMean corr  → RAW: {mean_raw:.4f}   CAL: {mean_cal:.4f}   "
      f"Δ={mean_cal - mean_raw:+.4f}")
print(f"Corr > 0.3 → RAW: {(raw_metrics['corr']>0.3).sum()}/{len(raw_metrics)}  "
      f"CAL: {(cal_metrics['corr']>0.3).sum() if cal_metrics is not None else '?'}/{len(raw_metrics)}")
print(f"Corr > 0.0 → RAW: {(raw_metrics['corr']>0.0).sum()}/{len(raw_metrics)}  "
      f"CAL: {(cal_metrics['corr']>0.0).sum() if cal_metrics is not None else '?'}/{len(raw_metrics)}")

# ── Bar plot: raw vs calibrated correlation ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

def _barplot(ax, metrics, title):
    m = metrics.dropna(subset=['corr']).sort_values('corr')
    colors = ['#e74c3c' if s == 'R' else '#2980b9' for s in m['stage']]
    labels = [f"{r['scientific_name'].split()[-1]}_{r['stage']}" for _, r in m.iterrows()]
    bars = ax.barh(range(len(m)), m['corr'], color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(m))); ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.set_xlabel('Pearson Correlation', fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.5, 0.8)
    ax.grid(axis='x', alpha=0.3)
    legend_elements = [Patch(facecolor='#2980b9', label='Adult (_A)'),
                       Patch(facecolor='#e74c3c', label='Recruit (_R)')]
    ax.legend(handles=legend_elements, fontsize=9)
    # Annotate values
    for i, (_, row) in enumerate(m.iterrows()):
        ax.text(row['corr'] + 0.01, i, f"{row['corr']:.3f}", va='center', fontsize=7.5)

_barplot(axes[0], raw_metrics, f"Raw predictions\n(mean corr={mean_raw:.3f})")
if cal_metrics is not None:
    _barplot(axes[1], cal_metrics, f"After OLS val calibration\n(mean corr={mean_cal:.3f})")
else:
    axes[1].set_visible(False)

plt.suptitle('Per-Species Correlation — GAT_RNN_2D + Species Heads\n21 Rolling CV Windows (2004–2024)', fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'species_metrics_barplot.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Scatter grid (raw) ────────────────────────────────────────────────────────
ncols = 4
nrows = int(np.ceil(len(species) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
axes_flat = axes.flatten()

for idx, (sp, pc, tc) in enumerate(zip(species, pred_cols, true_cols)):
    ax = axes_flat[idx]
    pred = raw_data[pc].values.astype(float)
    true = raw_data[tc].values.astype(float)
    mask = ~(np.isnan(pred) | np.isnan(true))
    pred, true = pred[mask], true[mask]
    row_m = raw_metrics[raw_metrics['species'] == sp]
    corr_val = row_m['corr'].values[0] if len(row_m) else np.nan
    r2_val   = row_m['r2'].values[0]   if len(row_m) else np.nan
    ax.scatter(true, pred, alpha=0.15, s=6, color='#2c7bb6', rasterized=True)
    lim = max(true.max(), pred.max(), 0.1) * 1.05
    ax.plot([0, lim], [0, lim], 'r--', lw=1, alpha=0.7)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('True (log1p)', fontsize=7); ax.set_ylabel('Pred (log1p)', fontsize=7)
    short = sp.rsplit(' ', 1)[-1] if ' ' in sp else sp
    ax.set_title(f"{short}\nr={corr_val:.3f}  R²={r2_val:.3f}", fontsize=8)
    ax.tick_params(labelsize=6)

for idx in range(len(species), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle('Raw Predictions vs True (log1p) — All 21 CV Windows', fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'species_scatter_grid.png'), dpi=130, bbox_inches='tight')
plt.close(fig)

print(f"\nFigures saved to {OUT_DIR}")
