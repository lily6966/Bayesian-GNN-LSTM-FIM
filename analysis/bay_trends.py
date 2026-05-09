"""
bay_trends.py
-------------
Bay-level annual and seasonal abundance trends, predicted vs observed.

For each (Bay × species), produces:
    • annual trend  (year on x-axis,  one line per species, panel per bay)
    • seasonal trend (month on x-axis, one line per species, panel per bay)

Stations are mapped to bays via FIM_restoration_0412*_station_metadata.pkl.
Station id is derived from the predictions CSV's `fips` column (which holds
node_id = station_id * 12 + (month - 1) → station_id = node_id // 12).

Outputs (analysis/figures/<run_id>/bay_trends/):
    bay_annual_<bay>.png/pdf       (one panel per species, mean over bay+year)
    bay_seasonal_<bay>.png/pdf     (one panel per species, mean over bay+month)
    bay_annual_combined.png/pdf    (one panel per species, multi-line by bay)
    bay_seasonal_combined.png/pdf  (one panel per species, multi-line by bay)
    bay_summary.csv                 long-format mean/median by bay × species × year/month

Usage:
    python analysis/bay_trends.py --run_id run_g --transform root4
    python analysis/bay_trends.py --run_id run_f --transform log1p \\
        --results_dir results/FIM_restoration_0412_stations_monthly \\
        --meta_pkl ../data/FIM_restoration_0412_station_metadata.pkl
"""
import argparse
import glob
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

BAY_NAMES = {
    'TB': 'Tampa Bay',
    'CH': 'Charlotte Harbor',
    'AP': 'Apalachicola Bay',
    'CK': 'Cedar Key',
    'JX': 'Jacksonville (NE)',
    'IR': 'Indian River',
}

BAY_COLORS = {
    'TB': '#1f77b4',
    'CH': '#ff7f0e',
    'AP': '#2ca02c',
    'CK': '#d62728',
    'JX': '#9467bd',
    'IR': '#8c564b',
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
        key=os.path.getmtime, reverse=True,
    )
    for cand in candidates:
        csv = os.path.join(cand, 'test_results.csv')
        if os.path.exists(csv):
            return csv
    return None


def load_with_bay(results_dir, meta_pkl, raw_npz=None, raw_spp_pkl=None):
    """Pool fold CSVs, attach Bay, optionally overlay RAW observed counts
    (true catches per (station, year, month), not window-averaged)."""
    with open(meta_pkl, 'rb') as f:
        meta = pickle.load(f)
    if not isinstance(meta, pd.DataFrame):
        meta = pd.DataFrame([
            {'station_id': sid, 'Bay': v.get('bay', 'UNK')}
            for sid, v in meta.items()
        ])
    bay_lookup = dict(zip(meta['station_id'], meta['Bay']))

    discovered = []
    for yr in range(2009, 2030):
        csv = find_test_csv(results_dir, yr)
        if csv is not None:
            discovered = sorted({c[len('predicted_'):]
                                  for c in pd.read_csv(csv, nrows=0).columns
                                  if c.startswith('predicted_')})
            break

    frames = []
    for yr in range(2009, 2030):
        csv = find_test_csv(results_dir, yr)
        if csv is None: continue
        df = pd.read_csv(csv)
        if 'year' in df.columns:
            df['year'] = df['year'].astype(int)
        holdout = df[df['year'] == yr].copy()
        if holdout.empty: continue
        holdout['test_year'] = yr
        holdout['station_id'] = holdout['fips'].astype(int)
        holdout['Bay'] = holdout['station_id'].map(bay_lookup).fillna('UNK')
        for sp in discovered:
            pred_col, true_col = f'predicted_{sp}', f'true_{sp}'
            if pred_col not in holdout.columns: continue
            holdout[f'cnt_pred_{sp}'] = np.clip(holdout[pred_col].astype(float).values, 0, None)
            holdout[f'cnt_true_{sp}'] = np.clip(holdout[true_col].astype(float).values, 0, None)
        frames.append(holdout)
        print(f'  loaded year {yr}: {len(holdout):,} rows')
    if not frames:
        raise RuntimeError(f'No test_results.csv found under {results_dir}')
    pooled = pd.concat(frames, ignore_index=True)

    # Optionally overlay RAW observed counts (no window averaging)
    if raw_npz and raw_spp_pkl and os.path.exists(raw_npz) and os.path.exists(raw_spp_pkl):
        print(f'  overlaying RAW observed counts from {raw_npz}')
        with open(raw_spp_pkl, 'rb') as f:
            spp = pickle.load(f)
        orig_targets = list(spp['target_species'])
        raw = np.load(raw_npz)['data']
        node_ids = raw[:, 0].astype(int)
        st_ids = node_ids // 12
        mos    = (node_ids % 12) + 1
        yrs_   = raw[:, 1].astype(int)
        for sp in discovered:
            if sp not in orig_targets: continue
            col = 2 + orig_targets.index(sp)
            cnt = np.expm1(np.clip(raw[:, col].astype(float), None, 30))
            raw_df = pd.DataFrame(dict(station_id=st_ids, year=yrs_,
                                        month=mos, raw_true=cnt))
            pooled = pooled.merge(raw_df, how='left',
                                   left_on=['station_id', 'year', 'month'],
                                   right_on=['station_id', 'year', 'month'])
            pooled[f'cnt_true_{sp}'] = pooled['raw_true'].fillna(0).clip(lower=0)
            pooled = pooled.drop(columns=['raw_true'])
        print(f'  raw observed overlaid for {sum(1 for s in discovered if s in orig_targets)} species')
    return pooled, discovered


def bootstrap_sum_ci(values, n_boot=200, qlo=0.025, qhi=0.975, seed=0):
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return (0.0, 0.0)
    n = len(values)
    sums = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        sums[b] = values[idx].sum()
    return float(np.quantile(sums, qlo)), float(np.quantile(sums, qhi))


def winsorize_both(df, sp, pct=95.0):
    """Winsorize at the pct-th percentile of NONZERO observed values per
    species (avoids clipping rare species to 0)."""
    cnt_pred = f'cnt_pred_{sp}'
    cnt_true = f'cnt_true_{sp}'
    if cnt_pred not in df.columns or cnt_true not in df.columns:
        return df
    valid = df[cnt_true].dropna().values
    nz = valid[valid > 0]
    if nz.size == 0: return df
    high = float(np.percentile(nz, min(100.0, pct)))
    if high <= 0: return df
    df[cnt_pred] = df[cnt_pred].clip(upper=high)
    df[cnt_true] = df[cnt_true].clip(upper=high)
    return df


def plot_per_bay(all_data, sp_list, bay, group_col, x_label, x_ticks_fn,
                  out_path, aggregator='sum', n_boot=200):
    sub = all_data[all_data['Bay'] == bay]
    n = len(sp_list)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    agg_fn = aggregator if aggregator in ('sum', 'mean', 'median') else 'sum'

    for i, sp in enumerate(sp_list):
        ax = axes[i]
        pcol, tcol = f'cnt_pred_{sp}', f'cnt_true_{sp}'
        if pcol not in sub.columns:
            ax.set_visible(False); continue
        agg = (sub.groupby(group_col)[[pcol, tcol]]
                  .agg(agg_fn).reset_index().sort_values(group_col))

        x_vals = agg[group_col].values
        ci_lo, ci_hi = [], []      # 25–75% IQR band (predicted)
        ci95_lo, ci95_hi = [], []  # 5–95% outer band (predicted)
        for x in x_vals:
            sub_pred = sub.loc[sub[group_col] == x, pcol].dropna().values
            if agg_fn == 'sum':
                lo, hi = bootstrap_sum_ci(sub_pred, n_boot=n_boot, qlo=0.25, qhi=0.75)
                lo95, hi95 = bootstrap_sum_ci(sub_pred, n_boot=n_boot, qlo=0.05, qhi=0.95)
            else:
                if len(sub_pred) >= 2:
                    lo = float(np.quantile(sub_pred, 0.25))
                    hi = float(np.quantile(sub_pred, 0.75))
                    lo95 = float(np.quantile(sub_pred, 0.05))
                    hi95 = float(np.quantile(sub_pred, 0.95))
                else:
                    lo = hi = lo95 = hi95 = float(sub_pred.mean()) if len(sub_pred) else 0.0
            ci_lo.append(lo); ci_hi.append(hi)
            ci95_lo.append(lo95); ci95_hi.append(hi95)
        ax.fill_between(x_vals, ci95_lo, ci95_hi,
                         color='steelblue', alpha=0.10, label='Pred 5–95%')
        ax.fill_between(x_vals, ci_lo, ci_hi,
                         color='steelblue', alpha=0.30, label='Pred 25–75% (IQR)')
        ax.plot(agg[group_col], agg[pcol], color='steelblue',
                marker='o', lw=1.6, markersize=4, label='Pred')
        ax.plot(agg[group_col], agg[tcol], color='darkorange',
                marker='s', lw=1.6, markersize=4, label='Obs')
        ax.set_title(COMMON.get(sp, sp), fontsize=10, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=8)
        y_label = ('Total catch (fish)' if agg_fn == 'sum'
                   else f'{agg_fn.capitalize()} catch (fish/station)')
        ax.set_ylabel(y_label, fontsize=8)
        ax.tick_params(labelsize=8)
        x_ticks_fn(ax, sub[group_col])
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n, len(axes)): axes[j].set_visible(False)
    fig.suptitle(f'{BAY_NAMES.get(bay, bay)} ({bay}) — predicted vs observed',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    savefig_pngpdf(fig, out_path)


def plot_bays_combined(all_data, sp_list, group_col, x_label, x_ticks_fn,
                        out_path, aggregator='sum', plot_observed=False):
    """One panel per species; multiple lines (one per bay) showing predicted
    (or observed) bay-total over time/month."""
    n = len(sp_list)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    agg_fn = aggregator if aggregator in ('sum', 'mean', 'median') else 'sum'
    bays = sorted(all_data['Bay'].dropna().unique())
    target_col_kind = 'cnt_true_' if plot_observed else 'cnt_pred_'

    for i, sp in enumerate(sp_list):
        ax = axes[i]
        col = f'{target_col_kind}{sp}'
        if col not in all_data.columns:
            ax.set_visible(False); continue
        for bay in bays:
            if bay == 'UNK': continue
            sub = all_data[all_data['Bay'] == bay]
            if sub.empty: continue
            agg = (sub.groupby(group_col)[col]
                       .agg(agg_fn).reset_index().sort_values(group_col))
            x_vals = agg[group_col].values
            # 25–75% IQR band per bay (predicted only — observed bands cluttered)
            if not plot_observed:
                lo, hi = [], []
                for x in x_vals:
                    vals = sub.loc[sub[group_col] == x, col].dropna().values
                    if agg_fn == 'sum' and len(vals) > 1:
                        l, h = bootstrap_sum_ci(vals, n_boot=100, qlo=0.25, qhi=0.75)
                    elif len(vals) >= 2:
                        l = float(np.quantile(vals, 0.25))
                        h = float(np.quantile(vals, 0.75))
                    else:
                        l = h = float(vals.mean()) if len(vals) else 0.0
                    lo.append(l); hi.append(h)
                ax.fill_between(x_vals, lo, hi,
                                color=BAY_COLORS.get(bay, '#000'), alpha=0.15)
            ax.plot(x_vals, agg[col],
                    color=BAY_COLORS.get(bay, '#000'),
                    marker='o', lw=1.4, markersize=3,
                    label=f'{bay} ({BAY_NAMES.get(bay, bay)})')
        ax.set_title(COMMON.get(sp, sp), fontsize=10, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=8)
        y_label = ('Total catch (fish)' if agg_fn == 'sum'
                   else f'{agg_fn.capitalize()} catch (fish/station)')
        ax.set_ylabel(y_label, fontsize=8)
        ax.tick_params(labelsize=8)
        x_ticks_fn(ax, all_data[group_col])
        if i == 0:
            ax.legend(fontsize=7, loc='best')

    for j in range(n, len(axes)): axes[j].set_visible(False)
    kind = 'Observed' if plot_observed else 'Predicted'
    fig.suptitle(f'Bay-level {kind} trends', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    savefig_pngpdf(fig, out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_id', default='run_g')
    p.add_argument('--results_dir', default=None)
    p.add_argument('--meta_pkl', default=None)
    p.add_argument('--raw_npz', default=None)
    p.add_argument('--raw_spp_pkl', default=None)
    p.add_argument('--pct', type=float, default=95.0)
    p.add_argument('--aggregator', choices=['sum', 'mean', 'median'], default='sum',
                   help='Aggregator across stations within each bay × year/month '
                        'bucket. Default sum gives total catch per bay (true population '
                        'index). mean/median give per-station averages.')
    args = p.parse_args()

    results_dir = args.results_dir or os.path.join(_GNN_DIR,
        'results/FIM_restoration_0412G_stations_monthly')
    meta_pkl    = args.meta_pkl    or os.path.join(_GNN_DIR,
        '..', 'data', 'FIM_restoration_0412G_station_metadata.pkl')
    out_dir = os.path.join(_SCRIPT_DIR, 'figures', args.run_id, 'bay_trends')
    os.makedirs(out_dir, exist_ok=True)

    raw_npz = args.raw_npz or os.path.join(_GNN_DIR, '..', 'data',
                                            'FIM_restoration_0412G_stations_monthly.npz')
    raw_pkl = args.raw_spp_pkl or os.path.join(_GNN_DIR, '..', 'data',
                                                'FIM_restoration_0412G_species_names.pkl')
    print(f'=== {args.run_id} bay-level trends ===')
    all_data, sp_list = load_with_bay(results_dir, meta_pkl,
                                       raw_npz=raw_npz, raw_spp_pkl=raw_pkl)
    print(f'\nPooled rows: {len(all_data):,}  Bays: {sorted(all_data.Bay.unique())}')
    print(f'Targets: {len(sp_list)}')

    for sp in sp_list:
        all_data = winsorize_both(all_data, sp, pct=args.pct)

    # Per-bay panels (each bay → one figure with all species)
    bays_avail = [b for b in sorted(all_data['Bay'].unique()) if b != 'UNK']
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']
    for bay in bays_avail:
        plot_per_bay(
            all_data, sp_list, bay,
            group_col='test_year', x_label='Year',
            x_ticks_fn=lambda ax, s: (ax.set_xticks(sorted(s.unique())[::2]),
                                       ax.tick_params(axis='x', rotation=45)),
            out_path=os.path.join(out_dir, f'bay_annual_{bay}'),
            aggregator=args.aggregator,
        )
        plot_per_bay(
            all_data, sp_list, bay,
            group_col='month', x_label='Month',
            x_ticks_fn=lambda ax, s: (ax.set_xticks(range(1, 13)),
                                       ax.set_xticklabels(month_labels, rotation=45, fontsize=7)),
            out_path=os.path.join(out_dir, f'bay_seasonal_{bay}'),
            aggregator=args.aggregator,
        )

    # Combined: all bays on the same axes per species, predicted only and observed only
    for kind, plot_obs in [('predicted', False), ('observed', True)]:
        plot_bays_combined(
            all_data, sp_list,
            group_col='test_year', x_label='Year',
            x_ticks_fn=lambda ax, s: (ax.set_xticks(sorted(s.unique())[::2]),
                                       ax.tick_params(axis='x', rotation=45)),
            out_path=os.path.join(out_dir, f'bay_annual_combined_{kind}'),
            aggregator=args.aggregator,
            plot_observed=plot_obs,
        )
        plot_bays_combined(
            all_data, sp_list,
            group_col='month', x_label='Month',
            x_ticks_fn=lambda ax, s: (ax.set_xticks(range(1, 13)),
                                       ax.set_xticklabels(month_labels, rotation=45, fontsize=7)),
            out_path=os.path.join(out_dir, f'bay_seasonal_combined_{kind}'),
            aggregator=args.aggregator,
            plot_observed=plot_obs,
        )

    # Long-format CSV — includes sum (population), mean, median
    rows = []
    for bay in bays_avail:
        sub = all_data[all_data['Bay'] == bay]
        for sp in sp_list:
            pcol, tcol = f'cnt_pred_{sp}', f'cnt_true_{sp}'
            if pcol not in sub.columns: continue
            for ty, group in sub.groupby('test_year'):
                rows.append(dict(bay=bay, species=sp, year=int(ty),
                                  pred_total=float(group[pcol].sum()),
                                  obs_total=float(group[tcol].sum()),
                                  pred_mean=float(group[pcol].mean()),
                                  obs_mean=float(group[tcol].mean()),
                                  pred_median=float(group[pcol].median()),
                                  obs_median=float(group[tcol].median()),
                                  n=len(group)))
            for mo, group in sub.groupby('month'):
                rows.append(dict(bay=bay, species=sp, month=int(mo),
                                  pred_total=float(group[pcol].sum()),
                                  obs_total=float(group[tcol].sum()),
                                  pred_mean=float(group[pcol].mean()),
                                  obs_mean=float(group[tcol].mean()),
                                  pred_median=float(group[pcol].median()),
                                  obs_median=float(group[tcol].median()),
                                  n=len(group)))
    sum_csv = os.path.join(out_dir, 'bay_summary.csv')
    pd.DataFrame(rows).to_csv(sum_csv, index=False)
    print(f'\nSaved: {sum_csv}')


if __name__ == '__main__':
    main()
