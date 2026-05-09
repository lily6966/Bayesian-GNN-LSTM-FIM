"""
compute_month_weights.py
------------------------
Compute per-species, per-month loss weights from training data.

Weight for species s at month m:
    w[m, s] = mean_log1p_abundance[m, s] / sum_over_m(mean_log1p_abundance[:, s])

This upweights peak recruitment / abundance months for each species and
downweights off-season months.  Weights are normalized per species so
they sum to 1.0 across the 12 months (preserving overall loss scale).

Output: analysis/figures/month_weights.npy   shape [12, n_target_species]
        analysis/figures/month_weights.png   heatmap for visual inspection

Run from gnn-lstm-v2/:
    python analysis/compute_month_weights.py
"""

import argparse
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GNN_DIR    = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _GNN_DIR)

DATA_NPZ = os.path.join(_GNN_DIR, '../data/FIM_restoration_0412_stations_monthly.npz')
SPP_PKL  = os.path.join(_GNN_DIR, '../data/FIM_restoration_0412_species_names.pkl')
FIGURES  = os.path.join(_SCRIPT_DIR, 'figures')
os.makedirs(FIGURES, exist_ok=True)

FORAGE_SPECIES = {'Lagodon rhomboides_A', 'Lagodon rhomboides_R'}
TEST_YEAR      = 2024   # use all years < TEST_YEAR-1 as training data

COMMON_NAMES = {
    'Archosargus probatocephalus_A':  'Sheepshead (Adult)',
    'Archosargus probatocephalus_R':  'Sheepshead (Recruit)',
    'Callinectes sapidus_R':          'Blue Crab (Recruit)',
    'Centropomus undecimalis_A':      'Common Snook (Adult)',
    'Centropomus undecimalis_SA':     'Common Snook (Sub-adult)',
    'Cynoscion nebulosus_A':          'Spotted Seatrout (Adult)',
    'Cynoscion nebulosus_R':          'Spotted Seatrout (Recruit)',
    'Lutjanus griseus_R':             'Gray Snapper (Recruit)',
    'Lutjanus griseus_SA':            'Gray Snapper (Sub-adult)',
    'Mycteroperca microlepis_SA':     'Gag Grouper (Sub-adult)',
    'Sciaenops ocellatus_R':          'Red Drum (Recruit)',
    'Sciaenops ocellatus_SA':         'Red Drum (Sub-adult)',
}

MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument('--all_species_targets', action='store_true',
                      help='Include feature_species (community) in the target set so '
                           'weights match a model trained with --all_species_targets.')
    argp.add_argument('--include_forage', action='store_true',
                      help='Keep forage species (Pinfish) in the target set.')
    argp.add_argument('--nonprey_targets', action='store_true',
                      help='Promote ONLY non-prey community species (Gear-160) into '
                           'the target set (mirrors main.py --nonprey_targets).')
    argp.add_argument('--prevalence_target_min', type=float, default=0.0,
                      help='Prevalence floor for auto-promoting community species to '
                           'targets (mirrors main.py --prevalence_target_min).')
    argp.add_argument('--test_year', type=int, default=TEST_YEAR,
                      help='Test year — prevalence is computed over years <= test_year-2.')
    argp.add_argument('--per_window_calendar', action='store_true',
                      help='Compute a separate [12, n_sp] weight slab for every '
                           'window year_end in the data range. Emits a .npz with '
                           'keys "year_ends" and "weights" [n_year_ends, 12, n_sp].')
    argp.add_argument('--win_size', type=int, default=3,
                      help='Window size (years) for --per_window_calendar mode.')
    cli = argp.parse_args()

    # ── Load species metadata ──────────────────────────────────────────────────
    with open(SPP_PKL, 'rb') as f:
        spp_data = pickle.load(f)
    all_target = spp_data['target_species']       # 14 (includes forage)
    if cli.include_forage:
        target_spp = list(all_target)
    else:
        target_spp = [s for s in all_target if s not in FORAGE_SPECIES]

    if cli.all_species_targets:
        community_spp = list(spp_data.get('feature_species', []))
        target_spp = target_spp + community_spp
        print(f'[all_species_targets] Appended {len(community_spp)} community species '
              f'→ {len(target_spp)} total targets')
    elif cli.nonprey_targets:
        community_spp = list(spp_data.get('feature_species', []))
        n_prey = int(spp_data.get('n_prey_community', len(community_spp) // 2))
        nonprey_spp = community_spp[n_prey:]
        target_spp = target_spp + nonprey_spp
        print(f'[nonprey_targets] Appended {len(nonprey_spp)} non-prey community species '
              f'→ {len(target_spp)} total targets')
    prevalence_mode_extra_cols = None
    if cli.prevalence_target_min > 0:
        community_spp = list(spp_data.get('feature_species', []))
        n_env = int(spp_data.get('n_env_features', 0))
        cb = 2 + len(all_target) + n_env
        # Prevalence over years <= test_year-2
        raw = np.load(DATA_NPZ)
        _d = raw['data']
        _yrs = _d[:, 1].astype(int)
        _mask = _yrs <= (cli.test_year - 2)
        _train = _d[_mask]
        promoted_names, promoted_cols = [], []
        for i, sp_name in enumerate(community_spp):
            col = cb + i
            prev = (_train[:, col] > 0).mean() if _mask.any() else 0.0
            if prev >= cli.prevalence_target_min:
                promoted_names.append(sp_name)
                promoted_cols.append(col)
        target_spp = target_spp + promoted_names
        prevalence_mode_extra_cols = promoted_cols
        print(f'[prevalence_target_min={cli.prevalence_target_min}] Promoted '
              f'{len(promoted_names)} community species → {len(target_spp)} total targets')
    n_sp = len(target_spp)
    print(f'Target species: {n_sp}')

    # ── Load raw data ──────────────────────────────────────────────────────────
    raw = np.load(DATA_NPZ)
    data = raw['data']   # [N, cols]  col0=node_id, col1=year, col2..=species_log1p
    print(f'Raw data shape: {data.shape}')

    # NPZ column layout:
    #   cols 0-1:                                node_id, Year
    #   cols 2..2+n_target-1:                    target species
    #   cols 2+n_target..2+n_target+n_env-1:     env features
    #   cols 2+n_target+n_env..end:              community species (prey + nonprey)
    n_env = int(spp_data.get('n_env_features', 0))
    community_base = 2 + len(all_target) + n_env
    if cli.include_forage:
        base_cols = list(range(2, 2 + len(all_target)))
    else:
        base_cols = [2 + i for i, s in enumerate(all_target) if s not in FORAGE_SPECIES]
    if cli.all_species_targets:
        community_cols = list(range(community_base,
                                    community_base + len(spp_data.get('feature_species', []))))
        target_col_idx = base_cols + community_cols
    elif cli.nonprey_targets:
        _feat = spp_data.get('feature_species', [])
        _n_prey = int(spp_data.get('n_prey_community', len(_feat) // 2))
        nonprey_cols = list(range(community_base + _n_prey,
                                  community_base + len(_feat)))
        target_col_idx = base_cols + nonprey_cols
    elif prevalence_mode_extra_cols is not None:
        target_col_idx = base_cols + prevalence_mode_extra_cols
    else:
        target_col_idx = base_cols

    # node_id encodes station*12 + (month-1)
    node_ids = data[:, 0].astype(int)
    months   = (node_ids % 12) + 1    # 1-indexed
    years    = data[:, 1].astype(int)

    # Data hygiene: log1p any target cols stored as raw counts (max > 15).
    Y_all = data[:, target_col_idx].astype(np.float32, copy=True)
    col_max = np.nanmax(Y_all, axis=0)
    raw_cols = np.where(col_max > 15.0)[0]
    if len(raw_cols) > 0:
        print(f'[data hygiene] {len(raw_cols)} target col(s) look like raw counts; applying log1p')
        Y_all[:, raw_cols] = np.log1p(np.clip(Y_all[:, raw_cols], 0.0, None))

    # ── Per-window calendar mode: emit [n_year_ends, 12, n_sp] ────────────────
    if cli.per_window_calendar:
        win_size = int(cli.win_size)
        all_years = np.unique(years)
        min_yr, max_yr = int(all_years.min()), int(all_years.max())
        year_ends = list(range(min_yr + win_size - 1, max_yr + 1))
        print(f'[per_window_calendar] win_size={win_size}, '
              f'year_ends={year_ends[0]}..{year_ends[-1]} ({len(year_ends)} windows)')
        weights_cube = np.zeros((len(year_ends), 12, n_sp), dtype=np.float32)

        for idx, ye in enumerate(year_ends):
            ys = ye - win_size + 1
            w_mask = (years >= ys) & (years <= ye)
            if w_mask.sum() == 0:
                weights_cube[idx] = 1.0  # uniform fallback
                continue
            Y_w = Y_all[w_mask]
            m_w = months[w_mask]
            mm  = np.zeros((12, n_sp), dtype=np.float64)
            for m in range(1, 13):
                mm_mask = m_w == m
                if mm_mask.sum() > 0:
                    mm[m-1] = np.nanmean(Y_w[mm_mask], axis=0)
            col_sums = mm.sum(axis=0, keepdims=True)
            col_sums[col_sums < 1e-10] = 1.0
            w = (mm / col_sums) * 12.0   # mean across months = 1.0
            weights_cube[idx] = w.astype(np.float32)
            print(f'  year_end={ye} ({ys}-{ye}): rows={int(w_mask.sum())}')

        npz_path = os.path.join(FIGURES, 'month_weights_per_window.npz')
        np.savez(npz_path,
                 year_ends=np.asarray(year_ends, dtype=np.int32),
                 weights=weights_cube,
                 win_size=np.int32(win_size),
                 species=np.asarray(target_spp, dtype=object))
        print(f'\nSaved: {npz_path}  weights.shape={weights_cube.shape}')
        print(f'\nDone. Use in run_train.sh:')
        print(f'  --month_weight_path {os.path.abspath(npz_path)}')
        return

    # ── Default: single [12, n_sp] slab from TEST_YEAR-1 training window ─────
    # Use only training years (< TEST_YEAR - 1)
    train_mask = years < (TEST_YEAR - 1)
    Y_train    = data[train_mask][:, target_col_idx]   # [N_train, 12_sp]
    m_train    = months[train_mask]

    print(f'Training rows: {train_mask.sum()}  ({years[train_mask].min()}–{years[train_mask].max()})')

    # ── Compute mean log1p abundance per month per species ────────────────────
    monthly_mean = np.zeros((12, n_sp), dtype=np.float64)
    for m in range(1, 13):
        mask = m_train == m
        if mask.sum() > 0:
            monthly_mean[m-1] = np.nanmean(Y_train[mask], axis=0)

    print('\nMonthly mean log1p abundance (rows=month, cols=species):')
    df_mm = pd.DataFrame(monthly_mean,
                         index=MONTH_LABELS,
                         columns=[COMMON_NAMES.get(s, s) for s in target_spp])
    print(df_mm.round(4).to_string())

    # ── Normalize per species → weights sum to 1.0 across months ─────────────
    col_sums = monthly_mean.sum(axis=0, keepdims=True)
    col_sums[col_sums < 1e-10] = 1.0   # avoid divide-by-zero for zero-prevalence species
    weights = monthly_mean / col_sums   # [12, n_sp], each column sums to 1

    # Scale so mean weight = 1.0 (preserves expected loss magnitude)
    weights = weights * 12.0

    print('\nMonth weights (×12, mean=1.0):')
    df_w = pd.DataFrame(weights,
                        index=MONTH_LABELS,
                        columns=[COMMON_NAMES.get(s, s) for s in target_spp])
    print(df_w.round(3).to_string())

    # ── Save .npy ─────────────────────────────────────────────────────────────
    npy_path = os.path.join(FIGURES, 'month_weights.npy')
    np.save(npy_path, weights.astype(np.float32))
    print(f'\nSaved: {npy_path}  shape={weights.shape}')

    # ── Heatmap ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: raw monthly mean abundance
    im0 = axes[0].imshow(monthly_mean, aspect='auto', cmap='YlOrRd')
    axes[0].set_xticks(range(n_sp))
    axes[0].set_xticklabels([COMMON_NAMES.get(s, s) for s in target_spp],
                            rotation=45, ha='right', fontsize=8)
    axes[0].set_yticks(range(12))
    axes[0].set_yticklabels(MONTH_LABELS, fontsize=9)
    axes[0].set_title('Mean log1p Abundance by Month (training data)', fontsize=11)
    plt.colorbar(im0, ax=axes[0], label='Mean log1p count')

    # Right: normalized weights
    im1 = axes[1].imshow(weights, aspect='auto', cmap='RdYlGn',
                         vmin=0, vmax=weights.max())
    axes[1].set_xticks(range(n_sp))
    axes[1].set_xticklabels([COMMON_NAMES.get(s, s) for s in target_spp],
                            rotation=45, ha='right', fontsize=8)
    axes[1].set_yticks(range(12))
    axes[1].set_yticklabels(MONTH_LABELS, fontsize=9)
    axes[1].set_title('Month Weights for Loss Function (×12, mean=1)', fontsize=11)
    plt.colorbar(im1, ax=axes[1], label='Weight')

    # Annotate weight values
    for i in range(12):
        for j in range(n_sp):
            axes[1].text(j, i, f'{weights[i,j]:.2f}',
                        ha='center', va='center', fontsize=6,
                        color='black' if weights[i,j] < weights.max()*0.7 else 'white')

    fig.suptitle('Species × Month Loss Weights — FIM Restoration Model',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    png_path = os.path.join(FIGURES, 'month_weights.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {png_path}')

    # ── Peak months summary ───────────────────────────────────────────────────
    print('\nPeak recruitment/abundance month per species:')
    for j, sp in enumerate(target_spp):
        peak_m = int(np.argmax(weights[:, j]))
        print(f'  {COMMON_NAMES.get(sp, sp):<40s}  peak = {MONTH_LABELS[peak_m]}  '
              f'(weight={weights[peak_m, j]:.2f})')

    print(f'\nDone. Use in run_train.sh:')
    print(f'  --month_weight_path {os.path.abspath(npy_path)}')


if __name__ == '__main__':
    main()
