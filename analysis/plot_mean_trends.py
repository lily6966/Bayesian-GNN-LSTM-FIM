"""
plot_mean_trends.py
-------------------
Plots mean predicted vs observed population trends for GAT-RNN-V2 Run H.

For each species:
  • Annual trend   — mean count per station across years
  • Seasonal trend — mean count per station across calendar months

Both predicted and observed get matching uncertainty bands:
  • Light shading (alpha 0.10): 5–95% station-level quantiles
  • Dark  shading (alpha 0.30): 25–75% station-level IQR

Outputs (PNG + PDF) saved to:
  analysis/figures/<run_id>/trends_mean/
    trend_annual_raw.{png,pdf}
    trend_annual_winsor95.{png,pdf}
    trend_seasonal_raw.{png,pdf}
    trend_seasonal_winsor95.{png,pdf}
    trend_metrics.csv

Usage:
    cd gnn-lstm-v4
    python analysis/plot_mean_trends.py                        # default: run_h
    python analysis/plot_mean_trends.py --run_id run_g
    python analysis/plot_mean_trends.py --run_id run_h --winsor_pct 99
"""

import argparse
import glob
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_GNN_DIR = os.path.dirname(_HERE)

# ── Species display names ─────────────────────────────────────────────────────
COMMON = {
    'Archosargus probatocephalus_A':  'Sheepshead (A)',
    'Archosargus probatocephalus_R':  'Sheepshead (R)',
    'Callinectes sapidus_R':          'Blue Crab (R)',
    'Centropomus undecimalis_A':      'Snook (A)',
    'Centropomus undecimalis_SA':     'Snook (SA)',
    'Cynoscion nebulosus_A':          'Seatrout (A)',
    'Cynoscion nebulosus_R':          'Seatrout (R)',
    'Lagodon rhomboides_A':           'Pinfish (A)',
    'Lagodon rhomboides_R':           'Pinfish (R)',
    'Lutjanus griseus_R':             'Gray Snapper (R)',
    'Lutjanus griseus_SA':            'Gray Snapper (SA)',
    'Mycteroperca microlepis_SA':     'Gag Grouper (SA)',
    'Sciaenops ocellatus_R':          'Red Drum (R)',
    'Sciaenops ocellatus_SA':         'Red Drum (SA)',
}

MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']


# ── I/O helpers ───────────────────────────────────────────────────────────────

def savefig(fig, base_path):
    """Save figure as both PNG and PDF."""
    base = os.path.splitext(base_path)[0]
    for ext in ('png', 'pdf'):
        fig.savefig(f'{base}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {base}.png / .pdf')


def find_test_csv(results_dir, year):
    candidates = sorted(
        glob.glob(os.path.join(results_dir, str(year), 'gat-rnn-v2-windowed_*')),
        key=os.path.getmtime, reverse=True
    )
    for cand in candidates:
        csv = os.path.join(cand, 'test_results.csv')
        if os.path.exists(csv):
            return csv
    return None


def discover_species(results_dir):
    """Find all target species from the first available test_results.csv."""
    for yr in range(2009, 2030):
        csv = find_test_csv(results_dir, yr)
        if csv:
            cols = pd.read_csv(csv, nrows=0).columns
            return sorted({c[len('predicted_'):] for c in cols
                           if c.startswith('predicted_')})
    raise RuntimeError(f'No test_results.csv found under {results_dir}')


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_folds(results_dir, species):
    """
    Pool all fold CSVs (one per test year).
    Returns a DataFrame with columns:
      fips, year, month, test_year,
      cnt_pred_{sp}, cnt_true_{sp}  for each species.
    """
    frames = []
    for yr in range(2009, 2030):
        csv = find_test_csv(results_dir, yr)
        if csv is None:
            continue
        df = pd.read_csv(csv)
        df['year'] = df['year'].astype(int)
        holdout = df[df['year'] == yr].copy()
        if holdout.empty:
            continue
        holdout['test_year'] = yr
        for sp in species:
            pc, tc = f'predicted_{sp}', f'true_{sp}'
            if pc not in holdout.columns:
                continue
            holdout[f'cnt_pred_{sp}'] = holdout[pc].astype(float).clip(lower=0)
            holdout[f'cnt_true_{sp}'] = holdout[tc].astype(float).clip(lower=0)
        frames.append(holdout)
        print(f'  loaded year {yr}: {len(holdout):,} rows')
    if not frames:
        raise RuntimeError(f'No data found under {results_dir}')
    return pd.concat(frames, ignore_index=True)


def winsorize(df, species, pct=95.0):
    """
    Clip both predicted and observed at the pct-th percentile of nonzero
    observed values (computed per species).
    """
    df = df.copy()
    for sp in species:
        pc, tc = f'cnt_pred_{sp}', f'cnt_true_{sp}'
        if pc not in df.columns:
            continue
        nz = df[tc].dropna().values
        nz = nz[nz > 0]
        if nz.size == 0:
            continue
        cap = float(np.percentile(nz, min(100.0, pct)))
        if cap > 0:
            df[pc] = df[pc].clip(upper=cap)
            df[tc] = df[tc].clip(upper=cap)
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def _bands(data, group_col, val_col, x_vals):
    """
    For each x in x_vals, compute station-level quantiles of val_col.
    Returns (iqr_lo, iqr_hi, out_lo, out_hi) as lists.
    """
    iqr_lo, iqr_hi, out_lo, out_hi = [], [], [], []
    for x in x_vals:
        vals = data.loc[data[group_col] == x, val_col].dropna().values
        if len(vals) >= 2:
            iqr_lo.append(float(np.quantile(vals, 0.25)))
            iqr_hi.append(float(np.quantile(vals, 0.75)))
            out_lo.append(float(np.quantile(vals, 0.05)))
            out_hi.append(float(np.quantile(vals, 0.95)))
        else:
            v = float(vals.mean()) if len(vals) else 0.0
            iqr_lo.append(v); iqr_hi.append(v)
            out_lo.append(v); out_hi.append(v)
    return iqr_lo, iqr_hi, out_lo, out_hi


def plot_trend(data, species, out_path, group_col, x_label,
               x_ticks_fn, title):
    """
    Multi-panel trend figure.

    One subplot per species.
    Both predicted (blue) and observed (orange) show:
      - Central line: mean across stations
      - Dark band (alpha=0.30): 25–75% station IQR
      - Light band (alpha=0.10): 5–95% station range
    """
    n_sp   = len(species)
    n_cols = 4
    n_rows = int(np.ceil(n_sp / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    for i, sp in enumerate(species):
        ax = axes[i]
        pc = f'cnt_pred_{sp}'
        tc = f'cnt_true_{sp}'
        if pc not in data.columns:
            ax.set_visible(False)
            continue

        agg = (data.groupby(group_col)[[pc, tc]]
                   .mean().reset_index().sort_values(group_col))
        x_vals = agg[group_col].values

        # Predicted bands
        p_iqr_lo, p_iqr_hi, p_out_lo, p_out_hi = _bands(data, group_col, pc, x_vals)
        ax.fill_between(x_vals, p_out_lo, p_out_hi,
                        color='steelblue', alpha=0.10, label='Pred 5–95%')
        ax.fill_between(x_vals, p_iqr_lo, p_iqr_hi,
                        color='steelblue', alpha=0.30, label='Pred 25–75%')

        # Observed bands
        o_iqr_lo, o_iqr_hi, o_out_lo, o_out_hi = _bands(data, group_col, tc, x_vals)
        ax.fill_between(x_vals, o_out_lo, o_out_hi,
                        color='darkorange', alpha=0.10, label='Obs 5–95%')
        ax.fill_between(x_vals, o_iqr_lo, o_iqr_hi,
                        color='darkorange', alpha=0.30, label='Obs 25–75%')

        # Central lines
        ax.plot(x_vals, agg[pc], color='steelblue',
                marker='o', lw=1.8, ms=4, label='Pred mean', zorder=3)
        ax.plot(x_vals, agg[tc], color='darkorange',
                marker='s', lw=1.8, ms=4, label='Obs mean', zorder=3)

        ax.set_title(COMMON.get(sp, sp), fontsize=10, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel('Mean catch (fish / station)', fontsize=8)
        ax.tick_params(labelsize=8)
        x_ticks_fn(ax, data[group_col])
        if i == 0:
            ax.legend(fontsize=6.5, loc='upper left', ncol=2)

    for j in range(n_sp, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    savefig(fig, out_path)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(data, species):
    rows = []
    for sp in species:
        pc, tc = f'cnt_pred_{sp}', f'cnt_true_{sp}'
        if pc not in data.columns:
            continue
        pred = data[pc].values.astype(float)
        true = data[tc].values.astype(float)
        ok   = np.isfinite(pred) & np.isfinite(true)
        if ok.sum() < 20:
            continue
        p, t = pred[ok], true[ok]
        corr = pearsonr(p, t)[0] if p.std() > 1e-10 and t.std() > 1e-10 else 0.0
        rows.append(dict(
            species=sp,
            common=COMMON.get(sp, sp),
            n=int(ok.sum()),
            prevalence=float((t > 0).mean()),
            pearson_r=float(corr),
            r2=float(r2_score(t, p)),
            rmse=float(np.sqrt(mean_squared_error(t, p))),
        ))
    return pd.DataFrame(rows).sort_values('pearson_r', ascending=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Plot mean population trends with 25/75 and 5/95 CI bands.'
    )
    p.add_argument('--run_id',      default='run_h',
                   help='Run identifier (default: run_h)')
    p.add_argument('--results_dir', default=None,
                   help='Path to results directory (auto-detected if omitted)')
    p.add_argument('--winsor_pct',  type=float, default=95.0,
                   help='Winsorization percentile of nonzero observed (default: 95)')
    args = p.parse_args()

    results_dir = (args.results_dir or
                   os.path.join(_GNN_DIR,
                                'results/FIM_restoration_0412G_stations_monthly'))
    out_dir = os.path.join(_HERE, 'figures', args.run_id, 'trends_mean')
    os.makedirs(out_dir, exist_ok=True)

    print(f'=== plot_mean_trends.py  |  run_id={args.run_id} ===')
    print(f'  results_dir : {results_dir}')
    print(f'  out_dir     : {out_dir}')

    # ── Load data ──────────────────────────────────────────────────────────────
    species = discover_species(results_dir)
    print(f'\nDiscovered {len(species)} target species')

    print('\nLoading fold CSVs …')
    raw_df  = load_all_folds(results_dir, species)
    win_df  = winsorize(raw_df, species, pct=args.winsor_pct)
    print(f'Total rows: {len(raw_df):,}  |  folds: {raw_df["test_year"].nunique()}')

    # ── x-axis tick helpers ────────────────────────────────────────────────────
    def annual_ticks(ax, s):
        yrs = sorted(s.unique())
        ax.set_xticks(yrs[::2])
        ax.tick_params(axis='x', rotation=45)

    def seasonal_ticks(ax, _s):
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_LABELS, rotation=45, fontsize=7)

    # ── Generate plots ─────────────────────────────────────────────────────────
    runs = [
        ('raw',
         raw_df,
         'Mean predicted vs observed  [annual, raw counts]',
         'Mean predicted vs observed  [seasonal, raw counts]'),
        (f'winsor{int(args.winsor_pct)}',
         win_df,
         f'Mean predicted vs observed  [annual, winsorized @ {args.winsor_pct:.0f}th pct]',
         f'Mean predicted vs observed  [seasonal, winsorized @ {args.winsor_pct:.0f}th pct]'),
    ]

    for tag, df, annual_title, seasonal_title in runs:
        print(f'\n--- {tag} ---')
        plot_trend(
            df, species,
            out_path   = os.path.join(out_dir, f'trend_annual_{tag}'),
            group_col  = 'test_year',
            x_label    = 'Year',
            x_ticks_fn = annual_ticks,
            title      = annual_title,
        )
        plot_trend(
            df, species,
            out_path   = os.path.join(out_dir, f'trend_seasonal_{tag}'),
            group_col  = 'month',
            x_label    = 'Month',
            x_ticks_fn = seasonal_ticks,
            title      = seasonal_title,
        )

    # ── Metrics CSV ────────────────────────────────────────────────────────────
    metrics = compute_metrics(raw_df, species)
    csv_out = os.path.join(out_dir, 'trend_metrics.csv')
    metrics.to_csv(csv_out, index=False)
    print(f'\nMetrics saved → {csv_out}')

    print(f'\nAll figures saved to: {out_dir}/')
    print('Files: trend_annual_raw, trend_annual_winsor95,',
          'trend_seasonal_raw, trend_seasonal_winsor95  (each .png + .pdf)')


if __name__ == '__main__':
    main()
