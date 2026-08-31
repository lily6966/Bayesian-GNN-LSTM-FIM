"""
Focused 2-D PDP driver for Run H.
Generates Temperature × Salinity 2-D PDPs for:
  - Red Drum (Sub-adult)      Sciaenops ocellatus_SA
  - Spotted Seatrout (Recruit) Cynoscion nebulosus_R
Reuses all heavy-lifting setup from pdp_analysis.py.
"""
import os
import numpy as np
import torch
import pickle

import pdp_analysis as P


def main():
    os.makedirs(P.FIGURES_DIR, exist_ok=True)
    print('=' * 64)
    print('Run H — Focused 2-D PDP (Red Drum Sub-adult, Seatrout Recruit)')
    print('=' * 64)

    args = P.FIMArgs()
    args = P.load_species_info(P.SPP_PKL, args)
    print(f'[pdp] {len(args.output_names)} target species')

    raw_data = np.load(P.DATA_NPZ)['data']
    X_mean_raw, X_std_raw = P.recover_X_stats(raw_data, args)

    (X_dict, Y_dict, _, adj, _,
     min_year, max_year, county_set, _, _) = \
        P.get_X_Y_2D(raw_data, args, P.device)
    print(f'[pdp] Stations: {len(county_set)}, years: {min_year}–{max_year}')

    g, nodeloader = P.build_graph_and_loader(adj, batch_size=128)
    print(f'[pdp] Graph: {g.num_nodes()} nodes, {g.num_edges()} edges')

    dist_weights = None
    if os.path.exists(P.DIST_PKL):
        with open(P.DIST_PKL, 'rb') as f:
            dist_weights = pickle.load(f)
        _s = next(iter(dist_weights.values()))
        args.edge_feat_dim = int(np.asarray(_s).ravel().shape[0])
        print(f'[pdp] Edge feat dim: {args.edge_feat_dim}')

    _s0     = county_set[0]
    _y0     = next(iter(X_dict[_s0]))
    in_dim  = X_dict[_s0][_y0].shape[-1]
    out_dim = len(args.output_names)
    ckpt_path = P.find_latest_checkpoint(P.CKPT_DIR)
    model     = P.load_model(ckpt_path, in_dim, out_dim, args)

    windowed_XY_T = P.build_merged_windows(
        X_dict, Y_dict, county_set, P.TEST_YEAR,
        win_size=P.WIN_SIZE, n_windows=P.N_WINDOWS
    )

    # ── Target species: Red Drum Sub-adult, Seatrout Recruit ──────────────────
    drum_sa_idx = next((i for i, s in enumerate(args.output_names)
                        if 'ocellatus_SA' in s), None)
    trout_r_idx = next((i for i, s in enumerate(args.output_names)
                        if 'nebulosus_R' in s), None)

    targets = [
        (drum_sa_idx, 'reddrum_SA', 'Red Drum (Sub-adult)'),
        (trout_r_idx, 'seatrout_R', 'Spotted Seatrout (Recruit)'),
    ]

    for sp_idx, sp_tag, sp_label in targets:
        if sp_idx is None:
            print(f'[pdp] WARNING: {sp_label} not found in output_names — skipping')
            continue
        print(f'\n[pdp] 2-D PDP Temperature × Salinity for {sp_label} '
              f'(idx={sp_idx}) …')
        g1_orig, g2_orig, pdp2d = P.compute_2d_pdp(
            model, nodeloader, windowed_XY_T, county_set,
            args, dist_weights,
            feat_idx_1=P.FEATURES['Temperature']['idx'],
            feat_idx_2=P.FEATURES['Salinity']['idx'],
            n_grid=P.N_GRID_2D,
            X_mean_raw=X_mean_raw, X_std_raw=X_std_raw,
            sp_idx=sp_idx
        )
        out_path = os.path.join(P.FIGURES_DIR, f'pdp_2d_{sp_tag}.png')
        P.plot_2d_pdp(g1_orig, g2_orig, pdp2d,
                      'Temperature (°C)', 'Salinity (PSU)', sp_label, out_path)

    print('\n[pdp] Done.')


if __name__ == '__main__':
    main()
