"""
Predicted vs Observed — FIM GNN-RNN Rolling Window Results
===========================================================
Generates:
  1. Scatter plots (predicted vs observed) per species — all test years combined
  2. Time series: mean observed vs mean predicted per year per species
  3. Monthly seasonal cycle: mean observed vs mean predicted by month
  4. Overall summary table

Usage:
    python analysis/compare_pred_obs.py \
        --results_dir gnn-rnn/results/FIM_stations_monthly \
        --out_dir     analysis/figures/pred_vs_obs
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── helpers ───────────────────────────────────────────────────────────────────

SPECIES = [
    'Pinfish_j', 'Pinfish_a',
    'Spotted Seatrout_j', 'Spotted Seatrout_a',
    'Red Drum_j', 'Red Drum_a',
    'Common Snook_j', 'Common Snook_a',
    'Blue Crab_j', 'Blue Crab_a',
    'Sheepshead_j', 'Sheepshead_a',
    'Gray Snapper_j', 'Gray Snapper_a',
    'Gag_j', 'Gag_a',
    'Tarpon_j', 'Tarpon_a',
]

MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']


def load_all_test(results_dir, epoch_tag='maxepoch-3'):
    """Load and concatenate all per-year test_results.csv files."""
    pattern = os.path.join(results_dir, f'*/gnn-rnn*{epoch_tag}*/test_results.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching: {pattern}")
    print(f"Loading {len(files)} test result files ...")
    dfs = []
    for f in files:
        test_year = int(f.split(os.sep)[-3])
        df = pd.read_csv(f)
        # Keep only rows where year == test_year (not look-back years)
        df = df[df['year'] == test_year]
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    print(f"  Total rows: {len(data):,}  |  years: {sorted(data.year.unique())}")
    return data


def metrics(obs, pred):
    mask = ~(np.isnan(obs) | np.isnan(pred))
    obs, pred = obs[mask], pred[mask]
    if len(obs) < 2:
        return dict(r2=np.nan, rmse=np.nan, mae=np.nan, corr=np.nan, n=0)
    r2   = r2_score(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    mae  = mean_absolute_error(obs, pred)
    corr, _ = pearsonr(obs, pred) if np.std(pred) > 1e-9 else (0, 1)
    return dict(r2=round(r2,3), rmse=round(rmse,2), mae=round(mae,2),
                corr=round(corr,3), n=int(mask.sum()))


# ── Plot 1: scatter grids ─────────────────────────────────────────────────────

def plot_scatter_grid(data, out_dir):
    """3×6 scatter grid — one panel per species."""
    n_cols, n_rows = 6, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    axes = axes.flatten()

    for ax, sp in zip(axes, SPECIES):
        pred_col = f'predicted_{sp}'
        true_col = f'true_{sp}'
        if pred_col not in data.columns:
            ax.set_visible(False)
            continue

        obs  = data[true_col].values.astype(float)
        pred = data[pred_col].values.astype(float)
        mask = ~(np.isnan(obs) | np.isnan(pred))
        obs, pred = obs[mask], pred[mask]

        # Clip extreme outliers for visual clarity (99th percentile)
        cap = max(np.percentile(obs, 99), np.percentile(pred, 99), 1)
        obs_p  = np.clip(obs,  0, cap)
        pred_p = np.clip(pred, 0, cap)

        ax.scatter(obs_p, pred_p, alpha=0.15, s=3, color='steelblue', rasterized=True)
        ax.plot([0, cap], [0, cap], 'r--', linewidth=1, label='1:1')

        m = metrics(obs, pred)
        ax.set_title(f'{sp}\nR²={m["r2"]:.2f}  r={m["corr"]:.2f}', fontsize=8)
        ax.set_xlabel('Observed count', fontsize=7)
        ax.set_ylabel('Predicted count', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlim(0, cap); ax.set_ylim(0, cap)

    for ax in axes[len(SPECIES):]:
        ax.set_visible(False)

    fig.suptitle('Predicted vs Observed Fish Counts — All Test Years (2004–2024)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, 'scatter_all_species.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Plot 2: time series ───────────────────────────────────────────────────────

def plot_timeseries(data, out_dir, top_n=6):
    """Annual mean predicted vs observed for top N species by abundance."""
    # Rank species by mean observed count
    means = {sp: data[f'true_{sp}'].mean() for sp in SPECIES
             if f'true_{sp}' in data.columns}
    top_species = sorted(means, key=means.get, reverse=True)[:top_n]

    years = sorted(data['year'].unique())
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=False)
    axes = axes.flatten()

    for ax, sp in zip(axes, top_species):
        obs_yr, pred_yr = [], []
        for yr in years:
            sub = data[data['year'] == yr]
            obs_yr.append(sub[f'true_{sp}'].mean())
            pred_yr.append(sub[f'predicted_{sp}'].mean())

        ax.plot(years, obs_yr,  'o-', color='steelblue', linewidth=2,
                markersize=5, label='Observed')
        ax.plot(years, pred_yr, 's--', color='darkorange', linewidth=2,
                markersize=5, label='Predicted')
        ax.fill_between(years, obs_yr, pred_yr, alpha=0.1, color='gray')
        ax.set_title(sp, fontsize=10)
        ax.set_xlabel('Year', fontsize=8)
        ax.set_ylabel('Mean count', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle('Annual Mean Predicted vs Observed — Top 6 Species (Test Years 2004–2024)',
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, 'timeseries_top6.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Plot 3: seasonal cycle ────────────────────────────────────────────────────

def plot_seasonal(data, out_dir, top_n=6):
    """Monthly mean predicted vs observed — seasonal cycle."""
    means = {sp: data[f'true_{sp}'].mean() for sp in SPECIES
             if f'true_{sp}' in data.columns}
    top_species = sorted(means, key=means.get, reverse=True)[:top_n]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    for ax, sp in zip(axes, top_species):
        obs_mo, pred_mo = [], []
        for mo in range(1, 13):
            sub = data[data['month'] == mo]
            obs_mo.append(sub[f'true_{sp}'].mean())
            pred_mo.append(sub[f'predicted_{sp}'].mean())

        ax.plot(range(1,13), obs_mo,  'o-', color='steelblue', linewidth=2,
                markersize=6, label='Observed')
        ax.plot(range(1,13), pred_mo, 's--', color='darkorange', linewidth=2,
                markersize=6, label='Predicted')
        ax.fill_between(range(1,13), obs_mo, pred_mo, alpha=0.1, color='gray')
        ax.set_xticks(range(1,13))
        ax.set_xticklabels(MONTH_NAMES, fontsize=7)
        ax.set_title(sp, fontsize=10)
        ax.set_ylabel('Mean count', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle('Seasonal Cycle: Predicted vs Observed — Top 6 Species',
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, 'seasonal_top6.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Plot 4: bias plot ─────────────────────────────────────────────────────────

def plot_bias(data, out_dir):
    """Mean bias (predicted - observed) per species as a bar chart."""
    bias = {}
    pct_bias = {}
    for sp in SPECIES:
        if f'true_{sp}' not in data.columns:
            continue
        obs  = data[f'true_{sp}'].values.astype(float)
        pred = data[f'predicted_{sp}'].values.astype(float)
        mask = ~(np.isnan(obs) | np.isnan(pred))
        if mask.sum() == 0:
            continue
        bias[sp]     = (pred[mask] - obs[mask]).mean()
        pct_bias[sp] = bias[sp] / (obs[mask].mean() + 1e-5) * 100

    spp   = list(bias.keys())
    vals  = [bias[s] for s in spp]
    pvals = [pct_bias[s] for s in spp]
    colors = ['tomato' if v > 0 else 'steelblue' for v in vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    bars = ax1.barh(spp, vals, color=colors, alpha=0.8, edgecolor='white')
    ax1.axvline(0, color='black', linewidth=1)
    ax1.set_xlabel('Mean bias (predicted − observed) [counts]')
    ax1.set_title('Absolute Bias per Species')
    ax1.grid(axis='x', alpha=0.3)

    bars2 = ax2.barh(spp, pvals,
                     color=['tomato' if v > 0 else 'steelblue' for v in pvals],
                     alpha=0.8, edgecolor='white')
    ax2.axvline(0, color='black', linewidth=1)
    ax2.set_xlabel('Relative bias [%]')
    ax2.set_title('Relative Bias per Species (% of mean observed)')
    ax2.grid(axis='x', alpha=0.3)

    fig.suptitle('Prediction Bias — All Test Years 2004–2024', fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, 'bias_per_species.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Summary table ─────────────────────────────────────────────────────────────

def save_summary_table(data, out_dir):
    rows = []
    for sp in SPECIES:
        if f'true_{sp}' not in data.columns:
            continue
        obs  = data[f'true_{sp}'].values.astype(float)
        pred = data[f'predicted_{sp}'].values.astype(float)
        m = metrics(obs, pred)
        bias = np.nanmean(pred - obs)
        rows.append({
            'species': sp,
            'mean_obs': round(float(np.nanmean(obs)), 3),
            'mean_pred': round(float(np.nanmean(pred)), 3),
            'bias': round(float(bias), 3),
            'pct_bias': round(float(bias / (np.nanmean(obs) + 1e-5) * 100), 1),
            **m
        })
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, 'metrics_summary.csv')
    df.to_csv(path, index=False)
    print(f'  Saved: {path}')
    print('\n' + df[['species','mean_obs','mean_pred','bias','r2','corr','rmse']].to_string(index=False))
    return df


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='gnn-rnn/results/FIM_stations_monthly')
    parser.add_argument('--out_dir',     default='analysis/figures/pred_vs_obs')
    parser.add_argument('--epoch_tag',   default='maxepoch-3',
                        help='String to match in results folder name')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = load_all_test(args.results_dir, epoch_tag=args.epoch_tag)

    print('\n=== 1. Scatter plots ===')
    plot_scatter_grid(data, args.out_dir)

    print('\n=== 2. Time series ===')
    plot_timeseries(data, args.out_dir)

    print('\n=== 3. Seasonal cycle ===')
    plot_seasonal(data, args.out_dir)

    print('\n=== 4. Bias ===')
    plot_bias(data, args.out_dir)

    print('\n=== 5. Summary table ===')
    save_summary_table(data, args.out_dir)

    print(f'\nAll plots saved to: {args.out_dir}/')


if __name__ == '__main__':
    main()
