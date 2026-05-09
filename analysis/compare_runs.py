"""
compare_runs.py
---------------
Side-by-side comparison of two FIM runs (default: Run G in v4 vs Run F in v3).

For the species that appear in BOTH runs' target lists, computes per-fold and
aggregated metrics (corr, R², F1_best) for each run, then shows the delta.

Outputs (analysis/figures/comparisons/<run_a>_vs_<run_b>/):
    fold_summary.csv               per-(year,species) metrics for both runs + delta
    species_aggregate.csv          per-species mean across overlap folds + delta
    delta_corr_per_species.png/pdf
    delta_r2_per_species.png/pdf
    delta_F1_per_species.png/pdf
    fold_means_overlay.png/pdf     mean metric across species per year, A vs B

Usage:
    python analysis/compare_runs.py                            # G vs F default
    python analysis/compare_runs.py --run_a_id run_g \\
        --run_a_results results/FIM_restoration_0412G_stations_monthly \\
        --run_b_id run_f \\
        --run_b_results ../gnn-lstm-v3/results/FIM_restoration_0412_stations_monthly
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import f1_score

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


def best_f1(y_true, y_score):
    thrs = np.unique(np.concatenate([
        [0.0, 0.1, 0.25, 0.5, 1.0],
        np.quantile(y_score, np.linspace(0.5, 0.99, 20))
    ]))
    bf = 0.0
    for t in thrs:
        pred = (y_score > t).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        f = f1_score(y_true, pred, zero_division=0)
        if f > bf:
            bf = float(f)
    return bf


def find_test_csv(results_dir, year, pattern='gat-rnn-v2-windowed_*'):
    candidates = sorted(
        glob.glob(os.path.join(results_dir, str(year), pattern)),
        key=os.path.getmtime,
        reverse=True,
    )
    for cand in candidates:
        csv = os.path.join(cand, 'test_results.csv')
        if os.path.exists(csv):
            return csv
    return None


def discover_targets(results_dir, pattern='gat-rnn-v2-windowed_*'):
    for yr in range(2009, 2030):
        csv = find_test_csv(results_dir, yr, pattern)
        if csv is not None:
            cols = pd.read_csv(csv, nrows=0).columns
            return sorted({c[len('predicted_'):] for c in cols if c.startswith('predicted_')})
    return []


def aggregate(results_dir, label, target_list, pattern='gat-rnn-v2-windowed_*'):
    rows = []
    for ty in range(2009, 2030):
        csv = find_test_csv(results_dir, ty, pattern)
        if csv is None: continue
        df = pd.read_csv(csv)
        df = df[df.year == ty]
        for s in target_list:
            pcol, tcol = f'predicted_{s}', f'true_{s}'
            if pcol not in df.columns: continue
            pr = df[pcol].values; tr = df[tcol].values
            m = np.isfinite(pr) & np.isfinite(tr)
            p, y = pr[m], tr[m]
            nz = (y > 0).mean()
            if nz == 0 or nz == 1: continue
            corr = pearsonr(p, y)[0] if (y.std() > 1e-9 and p.std() > 1e-9) else np.nan
            ss = ((y - y.mean()) ** 2).sum()
            r2 = (1 - ((y - p) ** 2).sum() / ss) if ss > 0 else np.nan
            f1b = best_f1((y > 0).astype(int), p)
            rows.append((ty, s, nz, corr, r2, f1b))
    return pd.DataFrame(
        rows, columns=['year', 'species', 'nz', 'corr', 'r2', 'F1_best']
    ).assign(run=label)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_a_id', default='run_g')
    p.add_argument('--run_a_results', default=os.path.join(
        _GNN_DIR, 'results/FIM_restoration_0412G_stations_monthly'))
    p.add_argument('--run_a_pattern', default='gat-rnn-v2-windowed_*',
                   help="Glob pattern under <year>/ to find Run A's checkpoint dir")
    p.add_argument('--run_b_id', default='run_f')
    p.add_argument('--run_b_results', default=os.path.join(
        _GNN_DIR, '..', 'gnn-lstm-v3',
        'results/FIM_restoration_0412_stations_monthly'))
    p.add_argument('--run_b_pattern', default='gat-rnn-v2-windowed_*',
                   help="Glob pattern under <year>/ to find Run B's checkpoint dir")
    args = p.parse_args()

    out_dir = os.path.join(_SCRIPT_DIR, 'figures', 'comparisons',
                           f'{args.run_a_id}_vs_{args.run_b_id}')
    os.makedirs(out_dir, exist_ok=True)

    print(f'=== Compare {args.run_a_id} vs {args.run_b_id} ===')
    print(f'  A results: {args.run_a_results}')
    print(f'  B results: {args.run_b_results}')

    # Targets in each + intersection
    A_tgts = set(discover_targets(args.run_a_results, args.run_a_pattern))
    B_tgts = set(discover_targets(args.run_b_results, args.run_b_pattern))
    common = sorted(A_tgts & B_tgts)
    print(f'  A targets ({len(A_tgts)}), B targets ({len(B_tgts)}), '
          f'common ({len(common)}): {common}')
    if not common:
        raise SystemExit('No common target species — nothing to compare.')

    # Aggregate metrics for each
    A = aggregate(args.run_a_results, args.run_a_id, common, args.run_a_pattern)
    B = aggregate(args.run_b_results, args.run_b_id, common, args.run_b_pattern)
    overlap_years = sorted(set(A.year) & set(B.year))
    print(f'  Overlap folds: {overlap_years}')
    A_o = A[A.year.isin(overlap_years)]
    B_o = B[B.year.isin(overlap_years)]

    # Per-(year,species) wide
    fold_long = pd.concat([A_o, B_o], ignore_index=True)
    fold_long.to_csv(os.path.join(out_dir, 'fold_long.csv'), index=False)

    fold_summary = (fold_long.pivot_table(
        index=['year', 'species'], columns='run',
        values=['corr', 'r2', 'F1_best']
    ))
    fold_summary.columns = [f'{m}_{r}' for m, r in fold_summary.columns]
    for m in ['corr', 'r2', 'F1_best']:
        fold_summary[f'd{m}'] = fold_summary[f'{m}_{args.run_a_id}'] - fold_summary[f'{m}_{args.run_b_id}']
    fold_summary = fold_summary.reset_index()
    fold_summary.to_csv(os.path.join(out_dir, 'fold_summary.csv'), index=False)
    print(f'\nSaved: {os.path.join(out_dir, "fold_summary.csv")}')

    # Species-level aggregation (mean across overlap folds)
    A_g = A_o.groupby('species').agg(corr_A=('corr', 'mean'), r2_A=('r2', 'mean'),
                                      F1_A=('F1_best', 'mean'),
                                      nz=('nz', 'mean')).reset_index()
    B_g = B_o.groupby('species').agg(corr_B=('corr', 'mean'), r2_B=('r2', 'mean'),
                                      F1_B=('F1_best', 'mean')).reset_index()
    M = A_g.merge(B_g, on='species', how='outer')
    M['common']  = M['species'].map(lambda s: COMMON.get(s, s))
    M['dcorr']   = M['corr_A'] - M['corr_B']
    M['dR2']     = M['r2_A']   - M['r2_B']
    M['dF1']     = M['F1_A']   - M['F1_B']
    M = M.sort_values('dcorr', ascending=False)

    sp_csv = os.path.join(out_dir, 'species_aggregate.csv')
    M.to_csv(sp_csv, index=False)
    print(f'Saved: {sp_csv}')

    # Print summary
    print('\n=== Mean across species (overlap folds) ===')
    print(f'  {args.run_a_id} corr={M.corr_A.mean():.3f}  R²={M.r2_A.mean():.3f}  F1_best={M.F1_A.mean():.3f}')
    print(f'  {args.run_b_id} corr={M.corr_B.mean():.3f}  R²={M.r2_B.mean():.3f}  F1_best={M.F1_B.mean():.3f}')
    print(f'  Δ           Δcorr={M.dcorr.mean():+.3f}  ΔR²={M.dR2.mean():+.3f}  ΔF1={M.dF1.mean():+.3f}')

    # ── Delta plots ────────────────────────────────────────────────────────
    for metric_label, dcol, full_a, full_b in [
        ('corr',     'dcorr', 'corr_A', 'corr_B'),
        ('R²',       'dR2',   'r2_A',   'r2_B'),
        ('F1_best',  'dF1',   'F1_A',   'F1_B'),
    ]:
        sorted_M = M.sort_values(dcol, ascending=True)
        labels = sorted_M['common'].fillna(sorted_M['species'])
        colors = ['#2E86AB' if d > 0 else '#A23B72' for d in sorted_M[dcol]]
        fig, ax = plt.subplots(figsize=(11, max(4, 0.4 * len(sorted_M))))
        ax.barh(range(len(sorted_M)), sorted_M[dcol], color=colors, edgecolor='none')
        ax.axvline(0, color='k', lw=0.5)
        ax.set_yticks(range(len(sorted_M)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(f'Δ{metric_label}  ({args.run_a_id} − {args.run_b_id})')
        ax.set_title(f'Per-species Δ{metric_label}: '
                     f'{args.run_a_id} vs {args.run_b_id}  ({len(overlap_years)} folds)',
                     fontsize=11, fontweight='bold')
        ax.legend(handles=[
            plt.Rectangle((0, 0), 1, 1, color='#2E86AB',
                           label=f'{args.run_a_id} better'),
            plt.Rectangle((0, 0), 1, 1, color='#A23B72',
                           label=f'{args.run_b_id} better'),
        ], loc='lower right', fontsize=9)
        fig.tight_layout()
        savefig_pngpdf(fig, os.path.join(out_dir, f'delta_{metric_label.replace("²","2")}_per_species'))

    # ── Fold-mean overlay (averaged across species, per year) ──────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for ax, metric in zip(axes, ['corr', 'r2', 'F1_best']):
        a_year = A_o.groupby('year')[metric].mean()
        b_year = B_o.groupby('year')[metric].mean()
        ax.plot(a_year.index, a_year.values, marker='o', color='#2E86AB',
                lw=2, label=args.run_a_id)
        ax.plot(b_year.index, b_year.values, marker='s', color='#A23B72',
                lw=2, label=args.run_b_id)
        ax.set_xlabel('Test year')
        ax.set_ylabel(f'Mean {metric} across species')
        ax.set_title(metric)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize=9)
    fig.suptitle(f'Per-fold mean: {args.run_a_id} vs {args.run_b_id}',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    savefig_pngpdf(fig, os.path.join(out_dir, 'fold_means_overlay'))


if __name__ == '__main__':
    main()
