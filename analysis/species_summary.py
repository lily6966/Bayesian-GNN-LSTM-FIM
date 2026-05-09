"""
species_summary.py
------------------
Aggregate per-species statistics for a rolling-windowed FIM run.

Outputs (analysis/figures/<run_id>/species_summary/):
    species_summary.csv        full species table (target + feature)
    target_metrics.csv         target-only fold-aggregated metrics
    species_prevalence.png/pdf
    target_metrics.png/pdf
    role_summary.png/pdf

Usage (defaults to Run G in v4):
    python analysis/species_summary.py
    python analysis/species_summary.py --run_id run_g
    python analysis/species_summary.py --run_id run_f \\
        --results_dir results/FIM_restoration_0412_stations_monthly \\
        --spp_pkl data/FIM_restoration_0412_species_names.pkl \\
        --data_npz data/FIM_restoration_0412_stations_monthly.npz
"""
import argparse
import glob
import os
import pickle

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


def best_f1_thr(y_true, y_score):
    thrs = np.unique(np.concatenate([
        [0.0, 0.1, 0.25, 0.5, 1.0],
        np.quantile(y_score, np.linspace(0.5, 0.99, 20))
    ]))
    best_thr, best_f = 0.5, 0.0
    for t in thrs:
        pred = (y_score > t).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f:
            best_thr, best_f = float(t), float(f)
    return best_thr, best_f


def savefig_pngpdf(fig, base_path):
    base_path = os.path.splitext(base_path)[0]
    for ext in ('png', 'pdf'):
        fig.savefig(f'{base_path}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {base_path}.png  +  .pdf')


def auto_discover_folds(results_dir):
    folds = []
    for d in sorted(glob.glob(os.path.join(results_dir, '*', '*windowed*', 'test_results.csv'))):
        try:
            yr = int(os.path.basename(os.path.dirname(os.path.dirname(d))))
            folds.append(yr)
        except ValueError:
            continue
    return sorted(set(folds))


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_id', default='run_g',
                   help='Subfolder under analysis/figures/ for outputs (default: run_g)')
    p.add_argument('--results_dir', default=None,
                   help='Path to results dir (default: results/FIM_restoration_0412G_stations_monthly)')
    p.add_argument('--spp_pkl', default=None,
                   help='Path to species_names.pkl')
    p.add_argument('--data_npz', default=None,
                   help='Path to NPZ')
    p.add_argument('--train_year_max', type=int, default=2021,
                   help='Years <= this used for prevalence/abundance stats (default 2021)')
    p.add_argument('--prev_promote', type=float, default=0.10,
                   help='Prevalence threshold for "promoted" target classification in summary')
    args = p.parse_args()

    results_dir = args.results_dir or os.path.join(_GNN_DIR, 'results/FIM_restoration_0412G_stations_monthly')
    spp_pkl     = args.spp_pkl     or os.path.join(_GNN_DIR, '..', 'data', 'FIM_restoration_0412G_species_names.pkl')
    data_npz    = args.data_npz    or os.path.join(_GNN_DIR, '..', 'data', 'FIM_restoration_0412G_stations_monthly.npz')
    out_dir     = os.path.join(_SCRIPT_DIR, 'figures', args.run_id, 'species_summary')
    os.makedirs(out_dir, exist_ok=True)

    folds = auto_discover_folds(results_dir)
    print(f'=== Run id: {args.run_id} ===')
    print(f'Auto-discovered {len(folds)} folds: {folds}')

    with open(spp_pkl, 'rb') as f:
        spp = pickle.load(f)
    orig_targets   = list(spp['target_species'])
    community      = list(spp['feature_species'])
    n_env          = int(spp['n_env_features'])
    community_base = 2 + len(orig_targets) + n_env

    raw = np.load(data_npz)
    data = raw['data']
    years = data[:, 1].astype(int)
    train_mask = years <= args.train_year_max
    train_data = data[train_mask]

    # ── Master species table ───────────────────────────────────────────────
    rows = []
    for i, name in enumerate(orig_targets):
        rows.append(dict(species=name, common=COMMON.get(name, name),
                         col=2 + i, origin='orig-target'))
    for i, name in enumerate(community):
        rows.append(dict(species=name, common=COMMON.get(name, name),
                         col=community_base + i, origin='community'))
    df = pd.DataFrame(rows)

    Y_t = train_data[:, df['col'].values].astype(np.float32, copy=True)
    raw_cols_idx = np.where(np.nanmax(Y_t, axis=0) > 15.0)[0]
    if len(raw_cols_idx) > 0:
        Y_t[:, raw_cols_idx] = np.log1p(np.clip(Y_t[:, raw_cols_idx], 0.0, None))

    df['prevalence']    = (Y_t > 0).mean(axis=0)
    df['mean_log1p']    = Y_t.mean(axis=0)
    df['std_log1p']     = Y_t.std(axis=0)
    df['max_log1p']     = Y_t.max(axis=0)
    df['median_log1p']  = np.median(Y_t, axis=0)
    nz_means = []
    for j in range(Y_t.shape[1]):
        col = Y_t[:, j]
        nz_means.append(col[col > 0].mean() if (col > 0).any() else 0.0)
    df['mean_log1p_nz'] = nz_means
    Y_count = np.expm1(np.clip(Y_t, None, 30.0))
    df['mean_count'] = Y_count.mean(axis=0)
    df['max_count']  = Y_count.max(axis=0)

    # Discover actual targets from first fold's CSV
    first_csv = find_test_csv(results_dir, folds[0]) if folds else None
    actual_targets = set()
    if first_csv:
        cols = pd.read_csv(first_csv, nrows=0).columns
        actual_targets = {c[len('predicted_'):] for c in cols if c.startswith('predicted_')}

    def role(r):
        if r['species'] in actual_targets:
            kind = 'orig-target' if r['origin'] == 'orig-target' else 'promoted-target'
            return f'{kind} (target)'
        return 'feature'
    df['role'] = df.apply(role, axis=1)

    # ── Aggregate model metrics across folds ───────────────────────────────
    fold_rows = []
    target_species = list(actual_targets)
    for ty in folds:
        csv = find_test_csv(results_dir, ty)
        if csv is None: continue
        rdf = pd.read_csv(csv)
        rdf = rdf[rdf.year == ty]
        for s in target_species:
            pcol, tcol = f'predicted_{s}', f'true_{s}'
            if pcol not in rdf.columns: continue
            pr = rdf[pcol].values; tr = rdf[tcol].values
            m = np.isfinite(pr) & np.isfinite(tr)
            p_, y_ = pr[m], tr[m]
            nz = (y_ > 0).mean()
            if nz == 0 or nz == 1: continue
            corr = (pearsonr(p_, y_)[0]
                    if (y_.std() > 1e-9 and p_.std() > 1e-9) else np.nan)
            ss = ((y_ - y_.mean()) ** 2).sum()
            r2 = (1 - ((y_ - p_) ** 2).sum() / ss) if ss > 0 else np.nan
            y_true = (y_ > 0).astype(int)
            f1_05  = f1_score(y_true, (p_ > 0.5).astype(int), zero_division=0)
            _, f1_b = best_f1_thr(y_true, p_)
            rmse = float(np.sqrt(((p_ - y_) ** 2).mean()))
            fold_rows.append((ty, s, nz, corr, r2, f1_05, f1_b, rmse))

    fold_df = pd.DataFrame(
        fold_rows,
        columns=['year','species','nonzero','corr','r2','F1_t0.5','F1_best','rmse']
    )
    metrics_agg = (fold_df.groupby('species')
                   .agg(n_folds=('year','count'),
                        mean_corr=('corr','mean'),
                        median_corr=('corr','median'),
                        std_corr=('corr','std'),
                        mean_r2=('r2','mean'),
                        median_r2=('r2','median'),
                        mean_F1_t0_5=('F1_t0.5','mean'),
                        mean_F1_best=('F1_best','mean'),
                        mean_rmse=('rmse','mean')).reset_index())
    df = df.merge(metrics_agg, on='species', how='left')

    out_csv = os.path.join(out_dir, 'species_summary.csv')
    df_sorted = df.sort_values(['role', 'prevalence'], ascending=[True, False])
    df_sorted.to_csv(out_csv, index=False)
    print(f'\nSaved: {out_csv}  ({len(df_sorted)} species)')

    target_csv = os.path.join(out_dir, 'target_metrics.csv')
    target_df = (df[df['role'].str.endswith('(target)')]
                 .sort_values('mean_F1_best', ascending=False))
    target_df.to_csv(target_csv, index=False)
    print(f'Saved: {target_csv}  ({len(target_df)} target species)')

    role_colors = {
        'orig-target (target)':     '#2E86AB',
        'promoted-target (target)': '#A23B72',
        'feature':                  '#C0C0C0',
    }

    # ── Plot 1: prevalence bars ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 18))
    ord_df = df.sort_values('prevalence', ascending=True)
    bar_colors = [role_colors.get(r, '#999999') for r in ord_df['role']]
    ax.barh(range(len(ord_df)), ord_df['prevalence'], color=bar_colors, edgecolor='none')
    ax.axvline(args.prev_promote, ls='--', c='red', lw=1)
    ax.set_yticks(range(len(ord_df)))
    ax.set_yticklabels(ord_df['common'], fontsize=7)
    ax.set_xlabel('Prevalence (fraction of training rows with count > 0)')
    ax.set_title(f'Species prevalence — {args.run_id} (training years ≤ {args.train_year_max})')
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=k) for k, c in role_colors.items()
    ] + [plt.Line2D([], [], ls='--', c='red', label=f'threshold {args.prev_promote}')]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=9)
    ax.set_xlim(0, max(0.4, ord_df['prevalence'].max() * 1.05))
    fig.tight_layout()
    savefig_pngpdf(fig, os.path.join(out_dir, 'species_prevalence'))

    # ── Plot 2: target metrics ─────────────────────────────────────────────
    if not target_df.empty:
        tgt = target_df.sort_values('mean_F1_best', ascending=True).copy()
        fig, axes = plt.subplots(1, 3, figsize=(18, max(6, 0.45 * len(tgt))), sharey=True)
        tcolors = [role_colors.get(r, '#999999') for r in tgt['role']]
        common_labels = tgt['common'].fillna(tgt['species'])

        axes[0].barh(range(len(tgt)), tgt['mean_corr'], color=tcolors, edgecolor='none')
        axes[0].axvline(0, color='k', lw=0.5)
        axes[0].set_yticks(range(len(tgt)))
        axes[0].set_yticklabels(common_labels, fontsize=8)
        axes[0].set_xlabel('Mean Pearson corr'); axes[0].set_title('Correlation')
        axes[0].set_xlim(min(-0.1, tgt['mean_corr'].min() - 0.05),
                         max(0.8, tgt['mean_corr'].max() + 0.05))

        axes[1].barh(range(len(tgt)), tgt['mean_r2'], color=tcolors, edgecolor='none')
        axes[1].axvline(0, color='k', lw=0.5)
        axes[1].set_xlabel('Mean R²'); axes[1].set_title('R²')
        axes[1].set_xlim(min(-0.5, tgt['mean_r2'].min() - 0.05),
                         max(0.5, tgt['mean_r2'].max() + 0.05))

        axes[2].barh(range(len(tgt)), tgt['mean_F1_t0_5'], color='lightgrey',
                     edgecolor='none', label='F1 @ 0.5')
        axes[2].barh(range(len(tgt)), tgt['mean_F1_best'], height=0.45,
                     color=tcolors, edgecolor='none', label='F1_best')
        axes[2].set_xlabel('F1 (presence/absence)'); axes[2].set_title('F1')
        axes[2].set_xlim(0, 1); axes[2].legend(loc='lower right', fontsize=8)

        fig.suptitle(f'{args.run_id} — target metrics across {fold_df.year.nunique()} folds',
                     fontsize=12, y=0.995)
        fig.tight_layout()
        savefig_pngpdf(fig, os.path.join(out_dir, 'target_metrics'))

    # ── Plot 3: role summary boxplots ──────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    role_order = sorted(df['role'].unique())
    bp_prev = axes[0].boxplot(
        [df[df['role'] == r]['prevalence'].values for r in role_order],
        patch_artist=True, tick_labels=[r.split(' ')[0] for r in role_order]
    )
    for patch, r in zip(bp_prev['boxes'], role_order):
        patch.set_facecolor(role_colors.get(r, '#999999'))
    axes[0].set_ylabel('Prevalence'); axes[0].set_title('Prevalence by role')
    axes[0].axhline(args.prev_promote, ls='--', c='red', lw=1, alpha=0.7)

    bp_logp = axes[1].boxplot(
        [df[df['role'] == r]['mean_log1p'].values for r in role_order],
        patch_artist=True, tick_labels=[r.split(' ')[0] for r in role_order]
    )
    for patch, r in zip(bp_logp['boxes'], role_order):
        patch.set_facecolor(role_colors.get(r, '#999999'))
    axes[1].set_ylabel('mean log1p count'); axes[1].set_title('Abundance by role')

    target_roles = [r for r in role_order if r.endswith('(target)')]
    if target_roles:
        bp_corr = axes[2].boxplot(
            [df[df['role'] == r]['mean_corr'].dropna().values for r in target_roles],
            patch_artist=True, tick_labels=[r.split(' ')[0] for r in target_roles]
        )
        for patch, r in zip(bp_corr['boxes'], target_roles):
            patch.set_facecolor(role_colors.get(r, '#999999'))
        axes[2].set_ylabel('Mean corr'); axes[2].set_title('Corr by role')
        axes[2].axhline(0, color='k', lw=0.5)

        bp_f1 = axes[3].boxplot(
            [df[df['role'] == r]['mean_F1_best'].dropna().values for r in target_roles],
            patch_artist=True, tick_labels=[r.split(' ')[0] for r in target_roles]
        )
        for patch, r in zip(bp_f1['boxes'], target_roles):
            patch.set_facecolor(role_colors.get(r, '#999999'))
        axes[3].set_ylabel('F1_best'); axes[3].set_title('F1 by role')
        axes[3].set_ylim(0, 1)

    fig.suptitle(f'{args.run_id} — role summary', fontsize=12)
    fig.tight_layout()
    savefig_pngpdf(fig, os.path.join(out_dir, 'role_summary'))

    print('\n=== Aggregates by role ===')
    print(df.groupby('role').agg(
        n=('species','count'),
        mean_prevalence=('prevalence','mean'),
        mean_corr=('mean_corr','mean'),
        mean_r2=('mean_r2','mean'),
        mean_F1_best=('mean_F1_best','mean'),
    ).round(3).to_string())


if __name__ == '__main__':
    main()
