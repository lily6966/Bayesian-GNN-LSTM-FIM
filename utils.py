import os
import pickle
import torch
import numpy as np
import sys
sys.path.append('../cnn-rnn')  # HACK
import visualization_utils
import warnings
import subprocess

def build_path(path):
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
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except Exception:
        return "no-git"


# # For each row in X, randomly choose a week between min_week and max_week (inclusive),
# # where weeks are indexed starting from 1.
# # If we want to fix a week, simply make min_week and max_week equal.
# # Zero out all features AFTER this week.
# # TODO - to save time, this could be put inside the DataLoader code
# def mask_end(X, args, min_week, max_week):
#     # If min_week is time_intervals, we're not masking any data, so
#     # just return the original X
#     if min_week == args.time_intervals:
#         return X

#     n_w = args.time_intervals*args.num_weather_vars  # Original: 52*6, new: 52*23
#     n_m = args.time_intervals*args.num_management_vars
#     num_vars = n_m + n_w
#     batch_size = X.shape[0]

#     # Create mask which is 1 for features up to (and incluing) the current
#     # week, and 0 afterwards. Index is 1 based
#     mask = np.zeros((batch_size, num_vars // args.time_intervals, args.time_intervals))
#     if min_week == max_week:
#         mask[:, :, :min_week] = 1
#     else:
#         weeks = np.random.randint(min_week, max_week+1, size=batch_size)
#         for i in range(batch_size):
#             mask[i, :, :weeks[i]] = 1

#     # "Flatten" mask, and then multiply each feature vector by the mask. The
#     # effect is to zero out all features after the chosen week.
#     mask = mask.reshape((batch_size, num_vars))
#     X[:, :n_w+n_m] = X[:, :n_w+n_m] * mask
#     return X


# For each row in X, randomly choose a week between min_week and max_week (inclusive),
# where weeks are indexed starting from 1.
# If we want to fix a week, simply make min_week and max_week equal.
# Zero out all features AFTER this week.
# TODO - to save time, this could be put inside the DataLoader code
def mask_end(X, counties, county_avg, args, min_week, max_week, device):
    # If min_week is time_intervals, we're not masking any data, so
    # just return the original X
    if min_week == args.time_intervals:
        return X

    n_w = args.time_intervals*args.num_weather_vars  # Original: 52*6, new: 52*23
    n_m = args.time_intervals*args.num_management_vars
    num_vars = n_m + n_w
    batch_size = X.shape[0]

    # Random boolean Tensor: True if we should mask the example, False otherwise
    examples_to_mask = (torch.rand((batch_size), device=device, dtype=float) <= args.mask_prob)

    # Create mask which is True (1) for features we want to replace/hide -
    # e.g. features after the current week. It's False (0) for features
    # up to (and including) the current week. Index is 1 based.
    # Initialize the mask to all 0. Then for the examples we want to mask,
    # set weeks after the "current week" to 1.
    mask = torch.zeros((batch_size, num_vars // args.time_intervals, args.time_intervals), dtype=bool, device=device)
    if min_week == max_week:
        mask[examples_to_mask, :, min_week:] = 1
    else:
        weeks = np.random.randint(min_week, max_week+1, size=batch_size)
        for i in range(batch_size):
            if examples_to_mask[i]:
                mask[i, :, weeks[i]:] = 1

    # Get historical average features for each county
    if args.mask_value == "county_avg":
        county_avg_matrix = torch.empty((batch_size, num_vars), device=device)
        for i in range(batch_size):
            county = counties[i].item()
            county_avg_matrix[i] = county_avg[county][:n_w+n_m]  # Only include time-dependent weather and management features

    # "Flatten" mask. Then update all indices where the mask is 1, and replace them with the county average values or 0.
    mask = mask.reshape((batch_size, num_vars))
    if args.mask_value == "zero":
        X[:, :n_w+n_m][mask] = 0
    elif args.mask_value == "county_avg":
        X[:, :n_w+n_m][mask] = county_avg_matrix[mask]
    return X


def get_X_Y(data, args, device):
    if args.data_dir == "soybean_data_full.npz":
        # Old dataset (given from CNN-RNN paper)
        counties = data[:, 0].astype(int)
        years = data[:, 1].astype(int)
        Y = data[:, 2:3]
        X = data[:, 3:]
    else:
        # Our dataset
        print("Initially data", data.shape)
        counties_all = data[:, 0].astype(int)
        years_all = data[:, 1].astype(int)

        # Only include years up to the test year.
        # Exclude county 25019 (Nantucket County) since it has no NLDAS data (crop datasets only).
        row_mask = years_all <= args.test_year
        if not getattr(args, 'no_soil', False) and 'stations_monthly' not in args.data_dir:
            row_mask = row_mask & (counties_all != 25019)
        # Rolling window: optionally restrict training data to a minimum year
        train_start_year = getattr(args, 'train_start_year', None)
        if train_start_year is not None:
            # Keep val/test years regardless; only restrict the training window
            row_mask = row_mask & (
                (years_all >= train_start_year) | (years_all >= args.test_year - 1)
            )
        # FIM bay exclusion: node_id = station_id * 12 + month → station_id = node_id // 12
        excluded_sids = getattr(args, 'excluded_station_ids', set())
        if excluded_sids:
            station_ids_all = counties_all // 12
            row_mask = row_mask & ~np.isin(station_ids_all, list(excluded_sids))
        data = data[row_mask]
        print("After filtering", data.shape)
        counties = data[:, 0].astype(int)
        years = data[:, 1].astype(int)

        # Support both single-int output_idx (crop) and list output_idx (FIM)
        other_cols = getattr(args, 'other_species_cols', [])
        if isinstance(args.output_idx, list):
            Y = data[:, args.output_idx].astype(np.float32, copy=True)
            # Data hygiene: the NPZ stores some community-species columns as raw
            # counts rather than log1p. Detect and log1p them so every target is
            # on a consistent scale. log1p(3.3M) ≈ 15; anything above that must
            # be raw counts, not log1p.
            if not getattr(args, 'no_log_transform', False):
                col_max = np.nanmax(Y, axis=0)
                raw_cols = np.where(col_max > 15.0)[0]
                if len(raw_cols) > 0:
                    print(f"[data hygiene] {len(raw_cols)} target column(s) look "
                          f"like raw counts (max > 15); applying log1p")
                    Y[:, raw_cols] = np.log1p(np.clip(Y[:, raw_cols], 0.0, None))
            # Optionally convert from log1p storage back to raw counts
            if getattr(args, 'no_log_transform', False):
                Y = np.expm1(Y)
            # Explicit feature columns override (NPZ layout with non-contiguous species)
            explicit_feature_cols = getattr(args, 'feature_cols', None)
            if explicit_feature_cols:
                X = data[:, explicit_feature_cols]
                feature_start = None
            else:
                if other_cols:
                    feature_start = 2 + len(args.output_idx) + len(other_cols)
                else:
                    feature_start = max(8, 2 + len(args.output_idx))
                X = data[:, feature_start:]
        else:
            Y = data[:, [args.output_idx]]
            feature_start = 8
            X = data[:, feature_start:]

        # FIM: prepend non-target species as additional input features
        if other_cols:
            X = np.concatenate([data[:, other_cols], X], axis=1)

    print("get_X_Y")
    print("X shape", X.shape)
    print("Y shape", Y.shape)

    # Compute the unique years and counties
    min_year = int(min(years))
    max_year = int(max(years))
    county_set = sorted(list(set(counties)))

    # Compute average yield of each year (to detect underlying yearly trends)
    avg_Y = {}
    avg_Y_lst = []
    for year in range(min_year, max_year+1):
        avg_Y[year] = np.nanmean(Y[years == year, :], axis=0)
        avg_Y_lst.append(avg_Y[year])
    '''mean_Y = np.mean(avg_Y_lst)
    std_Y = np.std(avg_Y_lst)
    for year in range(min_year, max_year+1):
        avg_Y[year] = (avg_Y[year] - mean_Y) / std_Y'''
    avg_Y[min_year-1] = avg_Y[min_year]

    # For each row in X, get the average yield(s) of the previous year, and add this as column(s) of X
    Ybar = []
    for year in years:
        Ybar.append(avg_Y[year-1])
    Ybar = np.array(Ybar)  #.reshape(-1, 1) - removed this because we may have multiple outputs
    X = np.concatenate((X, Ybar), axis=1)

    # Compute the mean and standard deviation of each feature (over non-NaN values), over the train years.
    # We will use these to standardize the features later.
    # If the feature is NaN everywhere, return 0 for mean/std (the "nan_to_num" function does this)
    train_start_year = getattr(args, 'train_start_year', None)
    known_years = (data[:, 1] < (args.test_year - 1))
    if train_start_year is not None:
        known_years = known_years & (data[:, 1] >= train_start_year)
    with warnings.catch_warnings():  # Supress warning about columns being NaN
        warnings.simplefilter("ignore", category=RuntimeWarning)
        X_mean = np.nanmean(X[known_years], axis=0, keepdims=True)
        X_std = np.nanstd(X[known_years], axis=0, keepdims=True)

        # HACK: If the standard deviation of a feature on train set is 0 (e.g. all
        # values are the same), then standardizing anything apart from that value
        # will produce a super extreme z-score. So just replace those standard 
        # deviations with 1. Also, if the mean/std are NaN, replace with 0 and 1. 
        X_std[(X_std < 1e-6) | np.isnan(X_std)] = 1
        X_mean[np.isnan(X_mean)] = 0

    # Standardize each feature (column of X)
    X = (X - X_mean) / (X_std + 1e-10)

    # # Check for extreme values in X (after standardization)
    # print('==============================')
    # indices = np.argwhere((X > 100) | (X < -100))
    # for i in range(indices.shape[0]):
    #     row, col = indices[i, 0], indices[i, 1]
    #     print("Extreme value indices", row, col + 7, "- yr", years[row])

    # For now, replace all NA with 0.
    X = np.nan_to_num(X)

    # # Fill in gaps for progress data
    # assert ((args.progress_indices[-1] + 1 - args.progress_indices[0]) % args.time_intervals == 0)
    # # Loop through each example
    # for i in range(X.shape[0]):
    #     # Loop through each progress variable (which is itself a range of "args.time_intervals" variables)
    #     for progress_var_start in range(args.progress_indices[0], args.progress_indices[-1] + 1, args.time_intervals):
    #         current_progress = 0
    #         for progress_idx in range(progress_var_start, progress_var_start + args.time_intervals):
    #             if np.isnan(X[i, progress_idx]):
    #                 X[i, progress_idx] = current_progress
    #             else:
    #                 current_progress = X[i, progress_idx]


    # Compute average of each output
    Y_mean = np.nanmean(Y[known_years], axis=0, keepdims=True)
    Y_std = np.nanstd(Y[known_years], axis=0, keepdims=True)
    # Guard: if std is 0 (species never varies) or NaN (species never observed),
    # set to 1 so standardization doesn't produce NaN or Inf.
    Y_std[(Y_std < 1e-6) | np.isnan(Y_std)] = 1
    Y_mean[np.isnan(Y_mean)] = 0
    args.means = torch.tensor(Y_mean, device=device)
    args.stds = torch.tensor(Y_std, device=device)
    print("Y (output) means", args.means, "stds", args.stds)

    # Compute count-scale statistics for loss and evaluation.
    # If no_log_transform, Y is already on count scale; otherwise apply expm1.
    if getattr(args, 'no_log_transform', False):
        Y_count = Y  # already raw counts
    else:
        # Clip log1p before expm1 to avoid float32 overflow for rare
        # very-high-count community species (log1p > ~40 → expm1 = +inf).
        Y_clipped = np.clip(Y, a_min=None, a_max=30.0)
        Y_count   = np.expm1(Y_clipped)
    Y_count_mean = np.nanmean(Y_count[known_years], axis=0, keepdims=True)
    Y_count_std  = np.nanstd( Y_count[known_years], axis=0, keepdims=True)
    # Guard NaN *and* inf in both stats
    bad_std = (Y_count_std < 1e-6) | np.isnan(Y_count_std) | np.isinf(Y_count_std)
    Y_count_std[bad_std] = 1
    Y_count_mean[np.isnan(Y_count_mean) | np.isinf(Y_count_mean)] = 0
    args.count_means = torch.tensor(Y_count_mean, device=device)
    args.count_stds  = torch.tensor(Y_count_std,  device=device)
    print("Y count-scale means", args.count_means, "stds", args.count_stds)

    # ── Species-month importance weights for the loss function ─────────────────
    # Shape [12, n_species]: peak recruitment months get higher weight so the
    # model is trained harder on ecologically meaningful seasonal patterns.
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
                args.month_weights = None   # set per-T in the training loop
                print(f"Loaded per-window (calendar) month weights from "
                      f"{month_weight_path}: cube.shape={cube.shape}, "
                      f"year_ends={year_ends[0]}..{year_ends[-1]}")
        else:
            mw = np.load(month_weight_path).astype(np.float32)   # (12, N_sp_trained)
            if mw.shape[1] != n_out:
                mw = np.ones((12, n_out), dtype=np.float32) / 12.0
                print(f"month_weight shape mismatch ({mw.shape[1]} vs {n_out}) — using uniform weights")
            args.month_weights = torch.tensor(mw, device=device)
            print(f"Loaded species-month weights from {month_weight_path}: {mw.shape}")
    else:
        args.month_weights = None

    # Create dictionaries mapping from (county + year) to features/labels
    X_dict = {}
    Y_dict = {}
    county_set = sorted(list(set(counties)))
    year_dict = {}
    county_dict = {}
    for county in county_set:
        X_dict[county] = {}
        Y_dict[county] = {}
        county_dict[county] = []
    for year in range(min_year, max_year+1):
        year_dict[year] = []
    for county, year, x, y in zip(counties, years, X, Y):
        X_dict[county][year] = x
        Y_dict[county][year] = y
        year_dict[year].append(x)
        if year < args.test_year - 1:
            county_dict[county].append(x)

    # Compute average features for each year (to use if there's missing data)
    year_avg = {}
    for year in range(min_year, max_year+1):
        year_dict[year] = np.array(year_dict[year])
        year_avg[year] = np.mean(year_dict[year], axis=0)

    # Compute average features per COUNTY (that can be used when we're doing early prediction
    # and don't have complete weather data)
    county_avg = {}
    for county in county_set:
        county_dict[county] = np.array(county_dict[county])
        county_avg[county] = torch.tensor(np.nanmean(county_dict[county], axis=0).astype(np.float32)).to(device)

    #l = args.length
    #print(min_year, max_year) # 1980, 2018
    #print(county_set) # n_counties

    avail_dict = {}
    for year in range(min_year, max_year+1):
        avail_dict[year] = []
        for j, county in enumerate(county_set):
            if year in X_dict[county]:
                avail_dict[year].append(j)

    # Adjacency — support both dict {'adj':csr,'ctid_to_order':dict} and raw sparse
    Data = pickle.load(open(args.us_adj_file, 'rb'))
    if isinstance(Data, dict):
        adj = Data['adj']
        ctid_to_order = Data['ctid_to_order']
    else:
        adj = Data
        fid_raw = pickle.load(open(args.crop_id_to_fid, 'rb'))
        ctid_to_order = fid_raw if isinstance(fid_raw, dict) else fid_raw.get('fid_dict', {})
    crop_data = pickle.load(open(args.crop_id_to_fid, 'rb'))
    id_to_fid = crop_data.get('fid_dict', crop_data) if isinstance(crop_data, dict) else crop_data
    order_map = {}
    indices = []
    for i, loc in enumerate(county_set):
        order_map[loc] = i
        if args.data_dir == "soybean_data_full.npz":
            fid = id_to_fid[loc]
        else:
            fid = loc
        indices.append(ctid_to_order[fid])
    sub_adj = adj[indices][:, indices]

    # Precompute neighbor lists from adjacency (works for both dense and sparse adj)
    import scipy.sparse as _sp
    N_nodes_adj = len(county_set)
    verbose_fill = N_nodes_adj <= 500  # suppress prints for large datasets like FIM
    if _sp.issparse(sub_adj):
        sub_adj_csr = sub_adj.tocsr()
        node_neighbors = []
        for i in range(N_nodes_adj):
            row = sub_adj_csr.getrow(i)
            _, cols = row.nonzero()
            node_neighbors.append([j for j in cols if j != i])
    else:
        node_neighbors = []
        for i in range(N_nodes_adj):
            node_neighbors.append([j for j in range(N_nodes_adj) if sub_adj[i, j] == 1 and j != i])

    n_missing = 0
    for year in range(min_year, max_year+1):
        for i, county in enumerate(county_set):
            # If data isn't present, fill in node features with the average feature values
            # of neighbors, or if no neighbors have data, replace with the year average.
            if year not in X_dict[county]:
                n_missing += 1
                if verbose_fill:
                    print("No data for county", county, "year", year)
                X_nbs = []
                Y_nbs = []
                for j in node_neighbors[i]:
                    nb = county_set[j]
                    if year in X_dict[nb]:
                        if verbose_fill:
                            print("--> Adding data from neighboring county", nb)
                        X_nbs.append(X_dict[nb][year])
                        Y_nbs.append(Y_dict[nb][year])
                if len(X_nbs):
                    X_dict[county][year] = np.mean(X_nbs, axis=0)
                    Y_dict[county][year] = np.mean(Y_nbs, axis=0)
                else:
                    if verbose_fill:
                        print("--> Not even neighboring counties have data :O")
                    X_dict[county][year] = year_avg[year]
                    Y_dict[county][year] = avg_Y[year]
    if n_missing > 0:
        print(f"Filled {n_missing} missing (node, year) entries from neighbors or year averages")

    '''loc1 = 300
    o1 = order_map[loc1]
    fid1 = id_to_fid[loc1]
    print("###", fid1)
    for year in range(2010, 2015):
        if year in Y_dict[loc1] and year+1 in Y_dict[loc1]:
            print("{:.2f}".format(Y_dict[loc1][year+1] - Y_dict[loc1][year]), end=',')
        else:
            print("-1.00", end=',')
    print()
    for i, loc2 in enumerate(county_set):
        if loc2 == loc1: continue
        o2 = order_map[loc2]
        if sub_adj[o1, o2] == 1:
            fid2 = id_to_fid[loc2]
            print("###", fid2)
            for year in range(2010, 2015):
                if year in Y_dict[loc2] and year+1 in Y_dict[loc2]:
                    print("{:.2f}".format(Y_dict[loc2][year+1] - Y_dict[loc2][year]), end=',')
                else:
                    print("-1.00", end=",")
            print()
    exit()'''
    ######

    return X_dict, Y_dict, avail_dict, sub_adj, order_map, min_year, max_year, county_set, county_avg, avg_Y


def get_X_Y_2D(data, args, device):
    """
    Variant of get_X_Y for the hierarchical nested 2D LSTM (SAGE_RNN_2D).

    Returns station-level data where each station has 12 monthly observations
    per year, enabling both seasonal and interannual modelling.

    Key differences from get_X_Y:
      - county_set   : list of station IDs (station_id = node_id // 12)
      - X_dict[s][y] : array [12, n_features]  (12 months for station s, year y)
      - Y_dict[s][y] : array [12, out_dim]
      - adj           : station-level sparse adjacency (month=0 slice of full adj)
    """
    import scipy.sparse as _sp

    # ── Reuse the existing per-month pipeline ─────────────────────────────────
    (X_dict_mn, Y_dict_mn, avail_dict_mn, adj_mn, order_map_mn,
     min_year, max_year, county_set_mn,
     county_avg, year_avg_Y) = get_X_Y(data, args, device)

    # county_set_mn holds station-month node IDs: node_id = station_id * 12 + month
    station_set = sorted(set(nid // 12 for nid in county_set_mn))

    # ── Reorganise dicts from (station_month_node, year) → (station, year, month) ──
    # Determine feature/output sizes from an existing entry
    _sample_x = next(iter(next(iter(X_dict_mn.values())).values()))
    n_feat  = _sample_x.shape[0]
    out_dim = len(args.output_names)

    X_dict_2d = {}
    Y_dict_2d = {}
    valid_stations = []  # stations that have at least some data

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

    # ── Build station-level adjacency from the month=0 slice ──────────────────
    Data_adj = pickle.load(open(args.us_adj_file, 'rb'))
    # Support both dict format {'adj':csr,'ctid_to_order':dict} (original FIM)
    # and raw sparse matrix format (restoration dataset, fid_dict loaded separately)
    if isinstance(Data_adj, dict):
        adj_full      = Data_adj['adj']
        ctid_to_order = Data_adj['ctid_to_order']
    else:
        adj_full = Data_adj
        fid_dict_path = getattr(args, 'crop_id_to_fid', None)
        fid_data = pickle.load(open(fid_dict_path, 'rb'))
        # fid_dict maps node_id → sequential index in adj
        ctid_to_order = fid_data if isinstance(fid_data, dict) and not isinstance(next(iter(fid_data.values())), dict) \
                        else fid_data.get('fid_dict', fid_data)

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
    kept_indices  = m0_indices  # already filtered to valid only

    adj_station = adj_full[kept_indices][:, kept_indices]
    order_map   = {sid: i for i, sid in enumerate(kept_stations)}

    # ── Availability: all stations available for all years ────────────────────
    avail_dict_2d = {
        year: list(range(len(kept_stations)))
        for year in range(min_year, max_year + 1)
    }

    # Rebuild X_dict_2d / Y_dict_2d to use kept_stations only
    X_dict_2d = {sid: X_dict_2d[sid] for sid in kept_stations}
    Y_dict_2d = {sid: Y_dict_2d[sid] for sid in kept_stations}

    print(f"[2D] Station nodes: {len(kept_stations)}, adj nnz: {adj_station.nnz}")
    return (X_dict_2d, Y_dict_2d, avail_dict_2d, adj_station, order_map,
            min_year, max_year, kept_stations, county_avg, year_avg_Y)
