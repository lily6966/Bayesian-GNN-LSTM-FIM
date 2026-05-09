"""
species_metrics.py — Per-species evaluation metrics from test_results.csv.

Computes Pearson correlation, RMSE, R², ROC-AUC, PR-AUC and prevalence for
each species from the saved test_results.csv produced by train.py / test.py.

Outputs (written to --out_dir):
  species_metrics.csv          — full per-species metrics table
  species_pearson_bar.png      — ranked bar chart of Pearson r
  species_roc_auc_bar.png      — ranked bar chart of ROC-AUC
  rmse_vs_prevalence.png       — scatter: RMSE vs prevalence, coloured by R²
  pearson_vs_prevalence.png    — scatter: Pearson r vs prevalence

Usage (from gnn-rnn-v2/ directory):
    python species_metrics.py \\
        --results_dir results/FIM_restoration_0412_stations_monthly/2024/<param_dir> \\
        --species_pkl ../data/FIM_restoration_0412_species_names.pkl \\
        --out_dir     species_analysis/2024
"""

import argparse
import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import compute_species_metrics, build_path


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='',
                   help='Directory containing test_results.csv (or path to the CSV itself)')
    p.add_argument('--species_pkl', default='',
                   help='Path to FIM species_names pkl. Auto-detected if empty.')
    p.add_argument('--out_dir',     default='species_analysis',
                   help='Directory to write output files')
    p.add_argument('--year',        type=int, default=-1,
                   help='Filter results to a specific test year (-1 = all rows)')
    p.add_argument('--top_n',       type=int, default=30,
                   help='Number of species to show in bar charts (default: 30)')
    return p.parse_args()


# ── Short display name ────────────────────────────────────────────────────────
COMMON = {
    'Archosargus probatocephalus': 'Sheepshead',
    'Callinectes sapidus':         'Blue Crab',
    'Centropomus undecimalis':     'Snook',
    'Cynoscion nebulosus':         'Spotted Seatrout',
    'Lagodon rhomboides':          'Pinfish',
    'Lutjanus griseus':            'Gray Snapper',
    'Mycteroperca microlepis':     'Gag Grouper',
    'Sciaenops ocellatus':         'Red Drum',
}

def common_name(sp):
    for sci, com in COMMON.items():
        if sp.startswith(sci.split()[0]) and sci.split()[1][:3] in sp:
            stage = sp.rsplit('_', 1)[-1]
            return f'{com} ({stage})'
    # Shorten to genus_epithet_stage
    parts = sp.split('_')
    if len(parts) >= 3:
        return f'{parts[0][:8]}_{parts[-1]}'
    return sp


# ── Load species names from pkl ───────────────────────────────────────────────
def load_species_names(spp_pkl):
    with open(spp_pkl, 'rb') as f:
        spp = pickle.load(f)
    if isinstance(spp, dict):
        return spp['target_species']
    else:
        return [s for s in spp if s.endswith(('_A', '_R', '_J', '_SA',
                                               '_a', '_r', '_j'))]


# ── Auto-detect species pkl ───────────────────────────────────────────────────
def auto_spp_pkl(results_dir):
    """Walk up from results_dir to find the data dir and species pkl."""
    # results/<dataset>/<year>/<param>/ → look for ../../data/<dataset>_species_names.pkl
    parts = results_dir.rstrip('/').split(os.sep)
    # Try to find 'results' in the path
    try:
        ri = parts.index('results')
        dataset = parts[ri + 1] if ri + 1 < len(parts) else ''
        root    = os.sep.join(parts[:ri])
    except ValueError:
        root    = os.path.dirname(results_dir)
        dataset = ''

    candidates = []
    if dataset:
        prefix = dataset.replace('_stations_monthly', '')
        candidates.append(os.path.join(root, 'data', f'{prefix}_species_names.pkl'))
        candidates.append(os.path.join(root, '..', 'data', f'{prefix}_species_names.pkl'))
    candidates.append(os.path.join(root, '..', 'data', 'FIM_restoration_0412_species_names.pkl'))
    candidates.append(os.path.join(root, '..', 'data', 'FIM_species_names.pkl'))

    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


# ── Load test_results.csv ─────────────────────────────────────────────────────
def load_results(results_dir, year=-1):
    if os.path.isfile(results_dir):
        csv_path = results_dir
    else:
        csv_path = os.path.join(results_dir, 'test_results.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'test_results.csv not found at {csv_path}')

    df = pd.read_csv(csv_path)
    print(f'[metrics] Loaded {len(df)} rows from {csv_path}')

    if year > 0 and 'year' in df.columns:
        df = df[df['year'] == year]
        print(f'[metrics] Filtered to year={year}: {len(df)} rows')

    return df


# ── Bar chart helper ──────────────────────────────────────────────────────────
def bar_chart(values, labels, title, xlabel, out_path, color_by=None, top_n=None):
    if top_n is not None and len(values) > top_n:
        idx    = np.argsort(np.abs(np.nan_to_num(values)))[::-1][:top_n]
        values = values[idx]
        labels = [labels[i] for i in idx]
        if color_by is not None:
            color_by = color_by[idx]

    n = len(values)
    if color_by is not None:
        norm  = plt.Normalize(vmin=float(np.nanmin(color_by)),
                              vmax=float(np.nanmax(color_by)))
        cmap  = cm.get_cmap('RdYlGn')
        colors = [cmap(norm(v)) if not np.isnan(v) else 'lightgrey'
                  for v in color_by]
    else:
        colors = ['steelblue'] * n

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.35)))
    y_pos = np.arange(n)
    bars  = ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axvline(0, color='black', lw=0.8)
    ax.invert_yaxis()   # highest at top

    if color_by is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='R²', shrink=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[metrics] Saved → {out_path}')


# ── Scatter plot helper ───────────────────────────────────────────────────────
def scatter_plot(x, y, labels, xlabel, ylabel, title, out_path,
                 color_by=None, color_label=''):
    fig, ax = plt.subplots(figsize=(8, 6))

    if color_by is not None:
        norm  = plt.Normalize(vmin=float(np.nanmin(color_by)),
                              vmax=float(np.nanmax(color_by)))
        cmap  = cm.get_cmap('RdYlGn')
        colors = np.array([cmap(norm(v)) if not np.isnan(v) else (0.8, 0.8, 0.8, 1.0)
                           for v in color_by])
        sc = ax.scatter(x, y, c=color_by, cmap='RdYlGn',
                        vmin=norm.vmin, vmax=norm.vmax, s=60, zorder=3)
        plt.colorbar(sc, ax=ax, label=color_label)
    else:
        ax.scatter(x, y, s=60, color='steelblue', zorder=3)

    # Annotate notable species
    sorted_idx = np.argsort(np.nan_to_num(y))[::-1]
    for i in sorted_idx[:8]:
        if not (np.isnan(x[i]) or np.isnan(y[i])):
            ax.annotate(labels[i], (x[i], y[i]),
                        xytext=(4, 2), textcoords='offset points', fontsize=7)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[metrics] Saved → {out_path}')


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    cli = parse_args()
    build_path(cli.out_dir)

    # Load results
    df = load_results(cli.results_dir, year=cli.year)

    # Detect species from column names
    pred_cols = [c for c in df.columns if c.startswith('predicted_')]
    true_cols = [c for c in df.columns if c.startswith('true_')]
    if not pred_cols:
        raise ValueError('No predicted_* columns found in test_results.csv')

    detected_species = [c[len('predicted_'):] for c in pred_cols]
    print(f'[metrics] Detected {len(detected_species)} species from column names')

    # Optionally load species pkl for canonical ordering
    spp_pkl = cli.species_pkl
    if not spp_pkl:
        spp_pkl = auto_spp_pkl(cli.results_dir)
    if spp_pkl and os.path.exists(spp_pkl):
        canonical = load_species_names(spp_pkl)
        # Use canonical ordering, keeping only detected species
        species_names = [s for s in canonical if s in set(detected_species)]
        extra = [s for s in detected_species if s not in set(canonical)]
        species_names = species_names + extra
        print(f'[metrics] Using canonical ordering: {len(species_names)} species')
    else:
        species_names = detected_species
        print(f'[metrics] No species pkl found, using column-order: {len(species_names)} species')

    # Compute per-species metrics using utils.compute_species_metrics
    metrics_df = compute_species_metrics(df, df, species_names)
    print(f'\n[metrics] Species-level metrics summary:')
    print(metrics_df[['species', 'pearson', 'rmse', 'r2', 'roc_auc', 'prevalence']].to_string(index=False))

    # Save metrics CSV
    out_csv = os.path.join(cli.out_dir, 'species_metrics.csv')
    metrics_df.to_csv(out_csv, index=False)
    print(f'\n[metrics] Saved → {out_csv}')

    # Prepare display names
    display_names = [common_name(s) for s in metrics_df['species'].tolist()]

    pearson_arr    = metrics_df['pearson'].values.astype(float)
    rmse_arr       = metrics_df['rmse'].values.astype(float)
    r2_arr         = metrics_df['r2'].values.astype(float)
    roc_arr        = metrics_df['roc_auc'].values.astype(float)
    prevalence_arr = metrics_df['prevalence'].values.astype(float)

    # Bar chart: Pearson r
    bar_chart(
        pearson_arr, display_names,
        title='Per-species Pearson r (test year)',
        xlabel='Pearson r',
        out_path=os.path.join(cli.out_dir, 'species_pearson_bar.png'),
        color_by=r2_arr,
        top_n=cli.top_n,
    )

    # Bar chart: ROC-AUC
    valid_roc = ~np.isnan(roc_arr)
    if valid_roc.sum() >= 3:
        bar_chart(
            roc_arr, display_names,
            title='Per-species ROC-AUC (presence/absence, test year)',
            xlabel='ROC-AUC',
            out_path=os.path.join(cli.out_dir, 'species_roc_auc_bar.png'),
            color_by=prevalence_arr,
            top_n=cli.top_n,
        )

    # Scatter: RMSE vs prevalence (coloured by R²)
    valid = ~(np.isnan(rmse_arr) | np.isnan(prevalence_arr))
    if valid.sum() >= 3:
        scatter_plot(
            prevalence_arr[valid], rmse_arr[valid],
            [display_names[i] for i in np.where(valid)[0]],
            xlabel='Prevalence (fraction of hauls present)',
            ylabel='RMSE',
            title='Per-species RMSE vs prevalence (count scale)',
            out_path=os.path.join(cli.out_dir, 'rmse_vs_prevalence.png'),
            color_by=r2_arr[valid],
            color_label='R²',
        )

    # Scatter: Pearson r vs prevalence
    valid2 = ~(np.isnan(pearson_arr) | np.isnan(prevalence_arr))
    if valid2.sum() >= 3:
        scatter_plot(
            prevalence_arr[valid2], pearson_arr[valid2],
            [display_names[i] for i in np.where(valid2)[0]],
            xlabel='Prevalence (fraction of hauls present)',
            ylabel='Pearson r',
            title='Per-species Pearson r vs prevalence',
            out_path=os.path.join(cli.out_dir, 'pearson_vs_prevalence.png'),
            color_by=roc_arr[valid2],
            color_label='ROC-AUC',
        )

    # Aggregate summary
    print('\n' + '='*60)
    print('SPECIES METRICS SUMMARY')
    print('='*60)
    print(f'  N species              : {len(metrics_df)}')
    print(f'  Mean Pearson r         : {np.nanmean(pearson_arr):.4f}')
    print(f'  Median Pearson r       : {np.nanmedian(pearson_arr):.4f}')
    print(f'  Mean RMSE              : {np.nanmean(rmse_arr):.4f}')
    print(f'  Mean R²                : {np.nanmean(r2_arr):.4f}')
    print(f'  Mean ROC-AUC           : {np.nanmean(roc_arr):.4f}')
    print(f'  N with ROC-AUC         : {np.sum(~np.isnan(roc_arr))}')

    top5_pearson = metrics_df.dropna(subset=['pearson']).nlargest(5, 'pearson')
    print(f'\n  Top-5 Pearson r:')
    for _, row in top5_pearson.iterrows():
        print(f'    {common_name(row["species"]):<35s}  r={row["pearson"]:.4f}  '
              f'prev={row["prevalence"]:.3f}')

    bot5_pearson = metrics_df.dropna(subset=['pearson']).nsmallest(5, 'pearson')
    print(f'\n  Bottom-5 Pearson r:')
    for _, row in bot5_pearson.iterrows():
        print(f'    {common_name(row["species"]):<35s}  r={row["pearson"]:.4f}  '
              f'prev={row["prevalence"]:.3f}')

    print(f'\n  All outputs saved to: {os.path.abspath(cli.out_dir)}/')
    print('='*60)


if __name__ == '__main__':
    main()
