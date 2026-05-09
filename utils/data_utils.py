"""
data_utils.py — Data loading utilities for GAT_RNN_V2.

Copies and re-exports get_X_Y, get_X_Y_2D, and helpers from the base utils.py.
These functions handle:
  - FIM station-month data loading
  - Feature standardisation
  - Missing data imputation (neighbor/year average)
  - Adjacency loading
"""

import os
import pickle
import torch
import numpy as np
import warnings
import subprocess


def build_path(path):
    """Recursively create directories in path."""
    path_levels = path.split('/')
    cur_path = ""
    for path_seg in path_levels:
        if len(cur_path):
            cur_path = cur_path + "/" + path_seg
        else:
            cur_path = path_seg
        if not os.path.exists(cur_path):
            os.mkdir(cur_path)


def get_git_revision_hash():
    """Return the current git commit hash, or 'no-git' if not in a repo."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
    except Exception:
        return "no-git"


def fill_missing(X_dict, Y_dict, county_set, node_neighbors, year_avg, avg_Y, min_year, max_year, verbose=False):
    """
    Fill missing (node, year) entries with neighbor average or year average.

    Parameters
    ----------
    X_dict       : dict[node][year] → feature array
    Y_dict       : dict[node][year] → label array
    county_set   : sorted list of nodes
    node_neighbors: list[list[int]] — neighbor indices per node in county_set
    year_avg     : dict[year] → mean X feature array
    avg_Y        : dict[year] → mean Y label array
    min_year, max_year : int

    Modifies X_dict and Y_dict in-place.
    """
    n_missing = 0
    for year in range(min_year, max_year + 1):
        for i, county in enumerate(county_set):
            if year not in X_dict[county]:
                n_missing += 1
                if verbose:
                    print(f"No data for county {county}, year {year}")
                X_nbs = []
                Y_nbs = []
                for j in node_neighbors[i]:
                    nb = county_set[j]
                    if year in X_dict[nb]:
                        X_nbs.append(X_dict[nb][year])
                        Y_nbs.append(Y_dict[nb][year])
                if len(X_nbs):
                    X_dict[county][year] = np.mean(X_nbs, axis=0)
                    Y_dict[county][year] = np.mean(Y_nbs, axis=0)
                else:
                    X_dict[county][year] = year_avg[year]
                    Y_dict[county][year] = avg_Y[year]
    if n_missing > 0:
        print(f"Filled {n_missing} missing (node, year) entries from neighbors or year averages")
    return X_dict, Y_dict


def get_X_Y(data, args, device):
    """
    Load and preprocess flat (non-nested) data from an NPZ array.

    Returns
    -------
    X_dict, Y_dict, avail_dict, sub_adj, order_map,
    min_year, max_year, county_set, county_avg, avg_Y
    """
    import scipy.sparse as _sp

    print("Initially data", data.shape)
    counties_all = data[:, 0].astype(int)
    years_all    = data[:, 1].astype(int)

    row_mask = years_all <= args.test_year
    if not getattr(args, 'no_soil', False) and 'stations_monthly' not in args.data_dir:
        row_mask = row_mask & (counties_all != 25019)
    train_start_year = getattr(args, 'train_start_year', None)
    if train_start_year is not None:
        row_mask = row_mask & (
            (years_all >= train_start_year) | (years_all >= args.test_year - 1)
        )
    excluded_sids = getattr(args, 'excluded_station_ids', set())
    if excluded_sids:
        station_ids_all = counties_all // 12
        row_mask = row_mask & ~np.isin(station_ids_all, list(excluded_sids))
    data = data[row_mask]
    print("After filtering", data.shape)
    counties = data[:, 0].astype(int)
    years    = data[:, 1].astype(int)

    other_cols = getattr(args, 'other_species_cols', [])
    if isinstance(args.output_idx, list):
        Y = data[:, args.output_idx].astype(np.float32, copy=True)
        # Data hygiene: some community-species cols in the NPZ are stored as
        # raw counts rather than log1p. log1p(3.3M)≈15, so any col with
        # max > 15 must be raw-count. Apply log1p so every target is on a
        # consistent scale.
        if not getattr(args, 'no_log_transform', False):
            col_max = np.nanmax(Y, axis=0)
            raw_cols = np.where(col_max > 15.0)[0]
            if len(raw_cols) > 0:
                print(f"[data hygiene] {len(raw_cols)} target column(s) look "
                      f"like raw counts (max > 15); applying log1p")
                Y[:, raw_cols] = np.log1p(np.clip(Y[:, raw_cols], 0.0, None))
        if getattr(args, 'no_log_transform', False):
            Y = np.expm1(Y)
        else:
            # Optional alternative transforms (Y is currently log1p-scale)
            transform = getattr(args, 'transform', 'log1p')
            if transform == 'root4':
                # log1p → counts → y^0.25
                Y_count = np.expm1(np.clip(Y, a_min=None, a_max=30.0))
                Y_count = np.clip(Y_count, a_min=0.0, a_max=None)
                Y = np.power(Y_count, 0.25).astype(np.float32)
                print(f"[transform] Applied root-4: y → expm1(log1p_y)^(1/4); "
                      f"new range [{Y.min():.2f}, {Y.max():.2f}]")
        explicit_feature_cols = getattr(args, 'feature_cols', None)
        if explicit_feature_cols:
            X = data[:, explicit_feature_cols]
            feature_start = None
        elif other_cols:
            feature_start = 2 + len(args.output_idx) + len(other_cols)
            X = data[:, feature_start:]
        else:
            feature_start = max(8, 2 + len(args.output_idx))
            X = data[:, feature_start:]
    else:
        Y = data[:, [args.output_idx]]
        feature_start = 8
        X = data[:, feature_start:]

    if other_cols:
        X = np.concatenate([data[:, other_cols], X], axis=1)

    print("get_X_Y")
    print("X shape", X.shape)
    print("Y shape", Y.shape)

    min_year   = int(min(years))
    max_year   = int(max(years))
    county_set = sorted(list(set(counties)))

    avg_Y     = {}
    avg_Y_lst = []
    for year in range(min_year, max_year + 1):
        avg_Y[year] = np.nanmean(Y[years == year, :], axis=0)
        avg_Y_lst.append(avg_Y[year])
    avg_Y[min_year - 1] = avg_Y[min_year]

    Ybar = []
    for year in years:
        Ybar.append(avg_Y[year - 1])
    Ybar = np.array(Ybar)
    X = np.concatenate((X, Ybar), axis=1)

    known_years = data[:, 1] < (args.test_year - 1)
    if train_start_year is not None:
        known_years = known_years & (data[:, 1] >= train_start_year)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        X_mean = np.nanmean(X[known_years], axis=0, keepdims=True)
        X_std  = np.nanstd( X[known_years], axis=0, keepdims=True)
        X_std[(X_std < 1e-6) | np.isnan(X_std)] = 1
        X_mean[np.isnan(X_mean)] = 0

    X = (X - X_mean) / (X_std + 1e-10)
    X = np.nan_to_num(X)

    Y_mean = np.nanmean(Y[known_years], axis=0, keepdims=True)
    Y_std  = np.nanstd( Y[known_years], axis=0, keepdims=True)
    Y_std[(Y_std < 1e-6) | np.isnan(Y_std)] = 1
    Y_mean[np.isnan(Y_mean)] = 0
    args.means = torch.tensor(Y_mean, device=device)
    args.stds  = torch.tensor(Y_std,  device=device)
    print("Y (output) means", args.means, "stds", args.stds)

    if getattr(args, 'no_log_transform', False):
        Y_count = Y
    else:
        Y_clipped = np.clip(Y, a_min=None, a_max=30.0)
        Y_count   = np.expm1(Y_clipped)
    Y_count_mean = np.nanmean(Y_count[known_years], axis=0, keepdims=True)
    Y_count_std  = np.nanstd( Y_count[known_years], axis=0, keepdims=True)
    bad_std = (Y_count_std < 1e-6) | np.isnan(Y_count_std) | np.isinf(Y_count_std)
    Y_count_std[bad_std] = 1
    Y_count_mean[np.isnan(Y_count_mean) | np.isinf(Y_count_mean)] = 0
    args.count_means = torch.tensor(Y_count_mean, device=device)
    args.count_stds  = torch.tensor(Y_count_std,  device=device)
    print("Y count-scale means", args.count_means, "stds", args.count_stds)

    month_weight_path = getattr(args, 'month_weight_path', None) or None
    args.window_month_weights_by_yearend = None
    if month_weight_path and os.path.exists(month_weight_path):
        n_out = len(args.output_names)
        if month_weight_path.endswith('.npz'):
            npz       = np.load(month_weight_path, allow_pickle=True)
            year_ends = npz['year_ends'].astype(int).tolist()
            cube      = npz['weights'].astype(np.float32)   # [n_ye, 12, n_sp]
            if cube.shape[2] != n_out:
                print(f"per-window month_weight shape mismatch "
                      f"({cube.shape[2]} vs {n_out}) — using uniform weights")
                args.month_weights = None
            else:
                args.window_month_weights_by_yearend = {
                    int(ye): torch.tensor(cube[i], device=device)
                    for i, ye in enumerate(year_ends)
                }
                args.month_weights = None
                print(f"Loaded per-window (calendar) month weights from "
                      f"{month_weight_path}: cube.shape={cube.shape}")
        else:
            mw = np.load(month_weight_path).astype(np.float32)
            if mw.shape[1] != n_out:
                mw = np.ones((12, n_out), dtype=np.float32) / 12.0
                print(f"month_weight shape mismatch — using uniform weights")
            args.month_weights = torch.tensor(mw, device=device)
            print(f"Loaded species-month weights from {month_weight_path}: {mw.shape}")
    else:
        args.month_weights = None

    X_dict     = {}
    Y_dict     = {}
    county_set = sorted(list(set(counties)))
    year_dict  = {}
    county_dict = {}
    for county in county_set:
        X_dict[county]    = {}
        Y_dict[county]    = {}
        county_dict[county] = []
    for year in range(min_year, max_year + 1):
        year_dict[year] = []
    for county, year, x, y in zip(counties, years, X, Y):
        X_dict[county][year] = x
        Y_dict[county][year] = y
        year_dict[year].append(x)
        if year < args.test_year - 1:
            county_dict[county].append(x)

    year_avg = {}
    for year in range(min_year, max_year + 1):
        year_dict[year] = np.array(year_dict[year])
        year_avg[year]  = np.mean(year_dict[year], axis=0)

    county_avg = {}
    for county in county_set:
        county_dict[county] = np.array(county_dict[county])
        county_avg[county]  = torch.tensor(
            np.nanmean(county_dict[county], axis=0).astype(np.float32)
        ).to(device)

    avail_dict = {}
    for year in range(min_year, max_year + 1):
        avail_dict[year] = []
        for j, county in enumerate(county_set):
            if year in X_dict[county]:
                avail_dict[year].append(j)

    Data = pickle.load(open(args.us_adj_file, 'rb'))
    if isinstance(Data, dict):
        adj          = Data['adj']
        ctid_to_order = Data['ctid_to_order']
    else:
        adj = Data
        fid_raw = pickle.load(open(args.crop_id_to_fid, 'rb'))
        ctid_to_order = fid_raw if isinstance(fid_raw, dict) else fid_raw.get('fid_dict', {})
    crop_data  = pickle.load(open(args.crop_id_to_fid, 'rb'))
    id_to_fid  = crop_data.get('fid_dict', crop_data) if isinstance(crop_data, dict) else crop_data
    order_map  = {}
    indices    = []
    for i, loc in enumerate(county_set):
        order_map[loc] = i
        fid = loc
        indices.append(ctid_to_order[fid])
    sub_adj = adj[indices][:, indices]

    N_nodes_adj = len(county_set)
    verbose_fill = N_nodes_adj <= 500
    if _sp.issparse(sub_adj):
        sub_adj_csr   = sub_adj.tocsr()
        node_neighbors = []
        for i in range(N_nodes_adj):
            row = sub_adj_csr.getrow(i)
            _, cols = row.nonzero()
            node_neighbors.append([j for j in cols if j != i])
    else:
        node_neighbors = []
        for i in range(N_nodes_adj):
            node_neighbors.append([j for j in range(N_nodes_adj)
                                    if sub_adj[i, j] == 1 and j != i])

    X_dict, Y_dict = fill_missing(
        X_dict, Y_dict, county_set, node_neighbors,
        year_avg, avg_Y, min_year, max_year, verbose=verbose_fill
    )

    return (X_dict, Y_dict, avail_dict, sub_adj, order_map,
            min_year, max_year, county_set, county_avg, avg_Y)


def get_X_Y_2D(data, args, device):
    """
    Variant of get_X_Y for the hierarchical nested 2D LSTM (GAT_RNN_V2).

    Returns station-level data where each station has 12 monthly observations
    per year, enabling both seasonal and interannual modelling.

    Returns
    -------
    X_dict_2d, Y_dict_2d, avail_dict_2d, adj_station, order_map,
    min_year, max_year, kept_stations, county_avg, year_avg_Y
    """
    import scipy.sparse as _sp

    (X_dict_mn, Y_dict_mn, avail_dict_mn, adj_mn, order_map_mn,
     min_year, max_year, county_set_mn,
     county_avg, year_avg_Y) = get_X_Y(data, args, device)

    station_set = sorted(set(nid // 12 for nid in county_set_mn))

    _sample_x = next(iter(next(iter(X_dict_mn.values())).values()))
    n_feat  = _sample_x.shape[0]
    out_dim = len(args.output_names)

    X_dict_2d   = {}
    Y_dict_2d   = {}
    valid_stations = []

    for sid in station_set:
        x_yr, y_yr = {}, {}
        for year in range(min_year, max_year + 1):
            x_months, y_months = [], []
            for m in range(12):
                nid = sid * 12 + m
                if nid in X_dict_mn and year in X_dict_mn[nid]:
                    x_months.append(X_dict_mn[nid][year])
                    y_months.append(Y_dict_mn[nid][year])
                else:
                    x_months.append(np.zeros(n_feat,  dtype=np.float32))
                    y_months.append(np.zeros(out_dim, dtype=np.float32))
            x_yr[year] = np.stack(x_months, axis=0)   # [12, n_feat]
            y_yr[year] = np.stack(y_months, axis=0)   # [12, out_dim]
        X_dict_2d[sid] = x_yr
        Y_dict_2d[sid] = y_yr
        valid_stations.append(sid)

    Data_adj = pickle.load(open(args.us_adj_file, 'rb'))
    if isinstance(Data_adj, dict):
        adj_full      = Data_adj['adj']
        ctid_to_order = Data_adj['ctid_to_order']
    else:
        adj_full = Data_adj
        fid_dict_path = getattr(args, 'crop_id_to_fid', None)
        fid_data = pickle.load(open(fid_dict_path, 'rb'))
        ctid_to_order = (fid_data
                         if isinstance(fid_data, dict)
                         and not isinstance(next(iter(fid_data.values())), dict)
                         else fid_data.get('fid_dict', fid_data))

    m0_indices = []
    keep_mask  = []
    for sid in valid_stations:
        nid_m0 = sid * 12 + 0
        if nid_m0 in ctid_to_order:
            m0_indices.append(ctid_to_order[nid_m0])
            keep_mask.append(True)
        else:
            keep_mask.append(False)

    kept_stations = [valid_stations[i] for i in range(len(valid_stations)) if keep_mask[i]]
    kept_indices  = m0_indices

    adj_station = adj_full[kept_indices][:, kept_indices]
    order_map   = {sid: i for i, sid in enumerate(kept_stations)}

    avail_dict_2d = {
        year: list(range(len(kept_stations)))
        for year in range(min_year, max_year + 1)
    }

    X_dict_2d = {sid: X_dict_2d[sid] for sid in kept_stations}
    Y_dict_2d = {sid: Y_dict_2d[sid] for sid in kept_stations}

    print(f"[2D] Station nodes: {len(kept_stations)}, adj nnz: {adj_station.nnz}")
    return (X_dict_2d, Y_dict_2d, avail_dict_2d, adj_station, order_map,
            min_year, max_year, kept_stations, county_avg, year_avg_Y)
