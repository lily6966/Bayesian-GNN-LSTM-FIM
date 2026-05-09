"""
trend_analysis.py
-----------------
Rolling-CV trend plots for a windowed run:

  • Annual trend     — mean count per station-month, observed vs predicted
  • Seasonal trend   — mean count per station-month by calendar month
  • Per-species CSV  — pooled metrics

All figures saved as both PNG and PDF under
analysis/figures/<run_id>/trends/.

Outlier handling for both predicted and observed counts (default ON):
  --pct 95     winsorize both at 95th percentile of observed
  --no_winsorize_observed     leave observed raw

Aggregator:
  --aggregator mean (default) | median

Usage:
    python analysis/trend_analysis.py
    python analysis/trend_analysis.py --run_id run_g
    python analysis/trend_analysis.py --run_id run_f --results_dir results/FIM_restoration_0412_stations_monthly
"""
import argparse
import glob
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

warnings.filterwarnings("ignore")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GNN_DIR    = os.path.dirname(_SCRIPT_DIR)


COMMON = {
    'Archosargus probatocephalus_A': 'Sheepshead (A)',
    'Archosargus probatocephalus_R': 'Sheepshead (R)',
    'Callinectes sapidus_R':         'Blue Crab (R)',
    'Centropomus undecimalis_A':     'Snook (A)',
    'Centropomus undecimalis_SA':    'Snook (SA)',
    'Cynoscion nebulosus_A':         'Seatrout (A)',
    'Cynoscion nebulosus_R':         'Seatrout (R)',
    'Lagodon rhomboides_A':          'Pinfish (A)',
    'Lagodon rhomboides_R':          'Pinfish (R)',
    'Lutjanus griseus_R':            'Gray Snapper (R)',
    'Lutjanus griseus_SA':           'Gray Snapper (SA)',
    'Mycteroperca microlepis_SA':    'Gag Grouper (SA)',
    'Sciaenops ocellatus_R':         'Red Drum (R)',
    'Sciaenops ocellatus_SA':        'Red Drum (SA)',
}


def savefig_pngpdf(fig, base_path):
    base_path = os.path.splitext(base_path)[0]
    for ext in ('png', 'pdf'):
        fig.savefig(f'{base_path}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {base_path}.png + .pdf')


def find_test_csv(results_dir, year):
    candidates = sorted(
        glob.glob(os.path.join(results_dir, str(year), 'gat-rnn-v2-windowed_*')),
        key=os.path.getmtime,
        reverse=True,
    )
    for cand in candidates:
        csv = os.path.join(cand, 'test_results.csv')
        if os.path.exists(csv):
            return csv
    return None


def auto_discover_targets(results_dir, year):
    csv = find_test_csv(results_dir, year)
    if csv is None:
        return []
    cols = pd.read_csv(csv, nrows=0).columns
    return sorted({c[len('predicted_'):] for c in cols if c.startswith('predicted_')})


def load_all_folds(results_dir, target_species, raw_npz=None, raw_spp_pkl=None):
    """Load fold CSVs and optionally replace `cnt_true_X` with the
    **raw observed counts from the NPZ** (per station-year-month, NOT the
    train-time window-averaged values). This makes the observed line show
    actual fish catches.

    If `raw_npz` and `raw_spp_pkl` are provided, raw observations are
    merged in via (station_id, year, month) keys.
    """
    frames = []
    found_years = []
    for yr in range(2009, 2030):
        csv = find_test_csv(results_dir, yr)
        if csv is None: continue
        df = pd.read_csv(csv)
        if 'year' in df.columns:
            df['year'] = df['year'].astype(int)
        holdout = df[df['year'] == yr].copy()
        if holdout.empty:
            continue
        holdout['test_year'] = yr
        for sp in target_species:
            pred_col, true_col = f'predicted_{sp}', f'true_{sp}'
            if pred_col not in holdout.columns:
                continue
            # Predictions: use CSV (back-transformed model output)
            holdout[f'cnt_pred_{sp}'] = np.clip(holdout[pred_col].astype(float).values, 0, None)
            holdout[f'cnt_true_{sp}'] = np.clip(holdout[true_col].astype(float).values, 0, None)
        frames.append(holdout)
        found_years.append(yr)
        print(f'  loaded year {yr}: {len(holdout):,} rows')
    if not frames:
        raise RuntimeError(f'No test_results.csv files under {results_dir}')
    pooled = pd.concat(frames, ignore_index=True)

    # Optionally overlay RAW observed counts (no window averaging)
    if raw_npz and raw_spp_pkl and os.path.exists(raw_npz) and os.path.exists(raw_spp_pkl):
        import pickle
        print(f'  overlaying RAW observed counts from {raw_npz}')
        with open(raw_spp_pkl, 'rb') as f:
            spp = pickle.load(f)
        orig_targets = list(spp['target_species'])
        raw_data = np.load(raw_npz)['data']
        # node_id encodes station_id*12 + (month-1)
        node_ids = raw_data[:, 0].astype(int)
        station_ids_raw = node_ids // 12
        months_raw      = (node_ids % 12) + 1
        years_raw       = raw_data[:, 1].astype(int)

        for sp in target_species:
            if sp not in orig_targets:
                continue
            col = 2 + orig_targets.index(sp)
            # NPZ stores log1p; convert back to raw count
            raw_count = np.expm1(np.clip(raw_data[:, col].astype(float), None, 30))
            raw_df = pd.DataFrame(dict(
                station_id=station_ids_raw, year=years_raw, month=months_raw,
                raw_true=raw_count
            ))
            # Merge into pooled by (fips=station_id, year, month)
            pooled = pooled.merge(
                raw_df, how='left',
                left_on=['fips', 'year', 'month'],
                right_on=['station_id', 'year', 'month'],
            )
            pooled[f'cnt_true_{sp}'] = pooled['raw_true'].fillna(0).clip(lower=0)
            pooled = pooled.drop(columns=['station_id', 'raw_true'])
        print(f'  raw observed counts overlaid for {sum(1 for s in target_species if s in orig_targets)} species')

    return pooled, found_years


def bootstrap_sum_ci(values, n_boot=200, qlo=0.025, qhi=0.975, seed=0):
    """Bootstrap CI for the sum of `values`."""
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return (0.0, 0.0)
    n = len(values)
    sums = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        sums[b] = values[idx].sum()
    return float(np.quantile(sums, qlo)), float(np.quantile(sums, qhi))


def winsorize_both(df, sp, pct=95.0, also_observed=True):
    """Winsorize at the `pct`-th percentile of NONZERO observed values
    (computed per species). Computing over all values would clip everything
    to 0 for rare species, so nonzero is the right reference set."""
    cnt_pred = f'cnt_pred_{sp}'
    cnt_true = f'cnt_true_{sp}'
    if cnt_pred not in df.columns or cnt_true not in df.columns:
        return df
    valid_true = df[cnt_true].dropna().values
    nz = valid_true[valid_true > 0]
    if nz.size == 0:
        return df
    high = float(np.percentile(nz, min(100.0, pct)))
    if high <= 0:
        return df
    df[cnt_pred] = df[cnt_pred].clip(upper=high)
    if also_observed:
        df[cnt_true] = df[cnt_true].clip(upper=high)
    return df


def _quantile_bands(all_data, group_col, val_col, x_vals):
    """Return (iqr_lo, iqr_hi, out_lo, out_hi) lists — station-level quantiles."""
    iqr_lo, iqr_hi, out_lo, out_hi = [], [], [], []
    for x in x_vals:
        sub = all_data.loc[all_data[group_col] == x, val_col].dropna().values
        if len(sub) >= 2:
            iqr_lo.append(float(np.quantile(sub, 0.25)))
            iqr_hi.append(float(np.quantile(sub, 0.75)))
            out_lo.append(float(np.quantile(sub, 0.05)))
            out_hi.append(float(np.quantile(sub, 0.95)))
        else:
            v = float(sub.mean()) if len(sub) else 0.0
            iqr_lo.append(v); iqr_hi.append(v)
            out_lo.append(v); out_hi.append(v)
    return iqr_lo, iqr_hi, out_lo, out_hi


def plot_trend(all_data, sp_list, out_path, group_col, x_label, x_ticks_fn,
               aggregator='mean', title_suffix='', n_boot=200):
    """Generic trend plot. group_col is 'test_year' or 'month'.

    aggregator='mean' (default): mean count per station per (year/month) bucket.
    Both predicted and observed lines get matching 25/75 and 5/95 station-level
    quantile bands.
    """
    n = len(sp_list)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    agg_fn = aggregator if aggregator in ('mean', 'median') else 'mean'

    for i, sp in enumerate(sp_list):
        ax = axes[i]
        pred_col, true_col = f'cnt_pred_{sp}', f'cnt_true_{sp}'
        if pred_col not in all_data.columns:
            ax.set_visible(False); continue

        agg = (all_data.groupby(group_col)[[pred_col, true_col]]
                       .agg(agg_fn).reset_index().sort_values(group_col))
        x_vals = agg[group_col].values

        # ── Predicted bands (station-level quantiles) ─────────────────────────
        p_iqr_lo, p_iqr_hi, p_out_lo, p_out_hi = _quantile_bands(
            all_data, group_col, pred_col, x_vals)
        ax.fill_between(x_vals, p_out_lo, p_out_hi,
                        color='steelblue', alpha=0.10, label='Pred 5–95%')
        ax.fill_between(x_vals, p_iqr_lo, p_iqr_hi,
                        color='steelblue', alpha=0.30, label='Pred 25–75% (IQR)')

        # ── Observed bands (station-level quantiles) ──────────────────────────
        o_iqr_lo, o_iqr_hi, o_out_lo, o_out_hi = _quantile_bands(
            all_data, group_col, true_col, x_vals)
        ax.fill_between(x_vals, o_out_lo, o_out_hi,
                        color='darkorange', alpha=0.10, label='Obs 5–95%')
        ax.fill_between(x_vals, o_iqr_lo, o_iqr_hi,
                        color='darkorange', alpha=0.30, label='Obs 25–75% (IQR)')

        # ── Central lines ─────────────────────────────────────────────────────
        ax.plot(x_vals, agg[pred_col], color='steelblue',
                marker='o', linewidth=1.8, markersize=4, label=f'Pred mean', zorder=3)
        ax.plot(x_vals, agg[true_col], color='darkorange',
                marker='s', linewidth=1.8, markersize=4, label=f'Obs mean', zorder=3)

        ax.set_title(COMMON.get(sp, sp), fontsize=10, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel('Mean catch (fish/station)', fontsize=8)
        ax.tick_params(labelsize=8)
        x_ticks_fn(ax, all_data[group_col])
        if i == 0:
            ax.legend(fontsize=7, loc='upper left', ncol=2)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Predicted vs observed{title_suffix}',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    savefig_pngpdf(fig, out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_id', default='run_g',
                   help='Subfolder under analysis/figures/ for outputs (default: run_g)')
    p.add_argument('--results_dir', default=None)
    p.add_argument('--raw_npz', default=None,
                   help='If provided, observed line uses RAW NPZ counts '
                        '(not window-averaged). Default: use Run G NPZ.')
    p.add_argument('--raw_spp_pkl', default=None,
                   help='Species pkl matching --raw_npz')
    p.add_argument('--aggregator', choices=['mean', 'median'], default='mean',
                   help='Aggregator across stations within each year/month bucket '
                        '(default: mean per-station catch).')
    args = p.parse_args()

    results_dir = args.results_dir or os.path.join(_GNN_DIR, 'results/FIM_restoration_0412G_stations_monthly')
    out_dir = os.path.join(_SCRIPT_DIR, 'figures', args.run_id, 'trends_mean')
    os.makedirs(out_dir, exist_ok=True)

    print(f'=== Run: {args.run_id} ===')
    print(f'  Note: CSV values are already in count scale (back-transformed at training time)')
    discovered = []
    for yr in range(2009, 2030):
        csv = find_test_csv(results_dir, yr)
        if csv is not None:
            discovered = auto_discover_targets(results_dir, yr)
            break
    print(f'Discovered {len(discovered)} target species')

    raw_npz = args.raw_npz or os.path.join(_GNN_DIR, '..', 'data',
                                            'FIM_restoration_0412G_stations_monthly.npz')
    raw_pkl = args.raw_spp_pkl or os.path.join(_GNN_DIR, '..', 'data',
                                                'FIM_restoration_0412G_species_names.pkl')
    all_data, years = load_all_folds(results_dir, discovered,
                                      raw_npz=raw_npz, raw_spp_pkl=raw_pkl)
    print(f'\nTotal rows pooled: {len(all_data):,} across {len(years)} folds')

    # Build BOTH a raw version and a winsorized version for side-by-side
    raw_df = all_data
    win_df = all_data.copy()
    print(f'\n  Building winsorized copy (95th pct of nonzero, pred+obs)')
    for sp in discovered:
        win_df = winsorize_both(win_df, sp, pct=95.0, also_observed=True)

    runs = [
        ('raw',        raw_df, f'  [{args.aggregator}, raw counts]'),
        ('winsor95',   win_df, f'  [{args.aggregator}, winsorized @ 95th pct of nonzero]'),
    ]

    # Annual + seasonal trends — generate BOTH raw and winsorized versions
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    for tag, df_v, suffix in runs:
        plot_trend(
            df_v, discovered,
            out_path=os.path.join(out_dir, f'trend_annual_{tag}'),
            group_col='test_year',
            x_label='Year',
            x_ticks_fn=lambda ax, s: (ax.set_xticks(sorted(s.unique())[::2]),
                                      ax.tick_params(axis='x', rotation=45)),
            aggregator=args.aggregator,
            title_suffix=' (annual)' + suffix,
        )
        plot_trend(
            df_v, discovered,
            out_path=os.path.join(out_dir, f'trend_seasonal_{tag}'),
            group_col='month',
            x_label='Month',
            x_ticks_fn=lambda ax, s: (ax.set_xticks(range(1,13)),
                                      ax.set_xticklabels(month_labels, rotation=45, fontsize=7)),
            aggregator=args.aggregator,
            title_suffix=' (seasonal)' + suffix,
        )

    # Per-species pooled metrics (computed on raw)
    rows = []
    for sp in discovered:
        pred_col, true_col = f'predicted_{sp}', f'true_{sp}'
        if pred_col not in raw_df.columns: continue
        pred = raw_df[pred_col].values.astype(float)
        true = raw_df[true_col].values.astype(float)
        valid = np.isfinite(pred) & np.isfinite(true)
        if valid.sum() < 20: continue
        p, t = pred[valid], true[valid]
        corr = pearsonr(p, t)[0] if (p.std() > 1e-10 and t.std() > 1e-10) else 0.0
        rows.append(dict(
            species=sp, common=COMMON.get(sp, sp),
            n=int(valid.sum()), prevalence=float((t > 0).mean()),
            corr=float(corr), r2=float(r2_score(t, p)),
            rmse=float(np.sqrt(mean_squared_error(t, p))),
        ))
    metrics = pd.DataFrame(rows).sort_values('corr', ascending=False)
    csv_out = os.path.join(out_dir, 'trend_metrics.csv')
    metrics.to_csv(csv_out, index=False)
    print(f'\nSaved: {csv_out}')


if __name__ == '__main__':
    main()
