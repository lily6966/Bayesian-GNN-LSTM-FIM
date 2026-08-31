"""
pdp_analysis.py  —  Partial Dependence Plots for GAT_RNN_V2 (gnn-lstm-v2).

For each target feature, marginalises out every other predictor by fixing the
feature to a grid value and averaging predictions across all test stations.
The full windowed pipeline runs for every grid point.

Feature layout in normalized X (in_dim=164, after forage prepend + Ybar append):
  forage           [0:2]    Pinfish adult (0), Pinfish recruit (1)
  env              [2:8]    Temperature(2), Salinity(3), DissolvedO2(4),
                            Secchi_depth(5), Secchi_on_bottom_bin(6), Tide_enc(7)
  restoration      [8:16]
  habitat_structured[16:23]
  habitat_open     [23:30]
  shoreline        [30:37]
  water_effort     [37:40]  CloudCover(37), StartDepth(38), Effort(39)
  bycatch          [40:54]
  na_indicators    [54:60]
  community        [60:152]
  Ybar             [152:164] previous-year target-species averages

Outputs (analysis/figures/):
  pdp_env_water.png         4-panel: Temperature, Salinity, DO, StartDepth
  pdp_temperature_all.png   Temperature PDP all species (one line each)
  pdp_salinity_all.png      Salinity PDP all species
  pdp_forage.png            Pinfish PDP (adult + recruit) for all species
  pdp_prey_community.png    Prey nekton community group PDP (0–2× scaling)
  pdp_nonprey_community.png Non-prey nekton community group PDP (0–2× scaling)
  pdp_community.png         2-panel: prey + non-prey side by side
  pdp_2d_snook.png          2-D PDP Temperature × Salinity for Snook (A)
  pdp_2d_seatrout.png       2-D PDP Temperature × Salinity for Spotted Seatrout (A)
  pdp_2d_*_x_prey_community.png   2-D PDP: env feature × prey community scaling
  pdp_2d_*_x_nonprey_community.png 2-D PDP: env feature × non-prey community scaling
  pdp_values.csv            Raw 1-D PDP values (long format)

Run from the gnn-lstm-v2/ directory:
    python analysis/pdp_analysis.py
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
import dgl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GNN_DIR    = os.path.dirname(_SCRIPT_DIR)
for _p in [_GNN_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import GAT_RNN_V2
from utils import get_X_Y_2D

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DATA_NPZ  = os.path.join(_GNN_DIR, '../data/FIM_restoration_0412G_stations_monthly.npz')
ADJ_PKL   = os.path.join(_GNN_DIR, '../map/FIM_restoration_0412_v2_adj.pkl')
FID_PKL   = os.path.join(_GNN_DIR, '../map/FIM_restoration_0412_v2_fid_dict.pkl')
SPP_PKL   = os.path.join(_GNN_DIR, '../data/FIM_restoration_0412G_species_names.pkl')
DIST_PKL  = os.path.join(_GNN_DIR, '../map/FIM_restoration_0412_v2_dist_weights.pkl')
META_PKL  = os.path.join(_GNN_DIR, '../data/FIM_restoration_0412G_station_metadata.pkl')
RUN_ID      = os.environ.get('PDP_RUN_ID', 'run_g')
_MAXEPOCH   = '35' if RUN_ID == 'run_h' else ('30' if RUN_ID in ('run_k','run_k2') else '50')
_DS_MODEL   = ('FIM_restoration_0412G_K_stations_monthly' if RUN_ID == 'run_k'
               else 'FIM_restoration_0412G_K2_stations_monthly' if RUN_ID == 'run_k2'
               else 'FIM_restoration_0412G_stations_monthly')
CKPT_DIR  = os.path.join(
    _GNN_DIR,
    f'model/{_DS_MODEL}/2024/'
    f'gat-rnn-v2-windowed_bs-128_lr-0.001_maxepoch-{_MAXEPOCH}_testyear-2024_win-3_nwin-10_seed-0'
)
FIGURES_DIR = os.path.join(_SCRIPT_DIR, 'figures', RUN_ID, 'pdp')

TEST_YEAR         = 2024
WIN_SIZE          = 3
N_WINDOWS         = 10
N_GRID            = 20    # grid points per feature for 1-D PDP
N_GRID_2D         = 12   # grid points per axis for 2-D PDP (12×12 = 144 model evals)
RESTORATION_ONLY  = os.environ.get('PDP_RESTORATION_ONLY', '0') == '1'
# Set PDP_RESTORATION_ONLY=1 to skip env/forage/2-D blocks and only run restoration PDPs


def _save_pngpdf(fig, base_path):
    """Save a figure as both .png and .pdf."""
    base_path = os.path.splitext(base_path)[0]
    for ext in ('png', 'pdf'):
        fig.savefig(f'{base_path}.{ext}', dpi=150, bbox_inches='tight')

# Named feature indices in the 164-dim normalized X
FEATURES = {
    'Pinfish_A':   dict(idx=0,  unit='log1p',  label='Pinfish Adult (log1p)'),
    'Pinfish_R':   dict(idx=1,  unit='log1p',  label='Pinfish Recruit (log1p)'),
    'Temperature': dict(idx=2,  unit='°C',     label='Temperature (°C)'),
    'Salinity':    dict(idx=3,  unit='PSU',     label='Salinity (PSU)'),
    'DissolvedO2': dict(idx=4,  unit='mg/L',   label='Dissolved O₂ (mg/L)'),
    'Secchi':      dict(idx=5,  unit='m',       label='Secchi Depth (m)'),
    'StartDepth':  dict(idx=38, unit='m',       label='Water Depth (m)'),
    'Effort':      dict(idx=39, unit='hauls',   label='Sampling Effort (hauls)'),
}

# Restoration feature indices (cols 8–15 in the full X, after 2 forage + 6 env)
# Order matches preprocess_restoration_0412.py: restoration_cols list
RESTORATION_FEATURES = {
    'n_projects':             dict(idx=8,  unit='count', label='# Restoration Projects'),
    'Acres_Restored':         dict(idx=9,  unit='acres', label='Acres Restored'),
    'targeted_habitat_num':   dict(idx=10, unit='count', label='# Targeted Habitat Types'),
    'Restoration_technique':  dict(idx=11, unit='count', label='# Restoration Techniques'),
    'primary_habitat':        dict(idx=12, unit='code',  label='Primary Habitat (code)'),
    'secondary_habitat':      dict(idx=13, unit='code',  label='Secondary Habitat (code)'),
    'Technique_first':        dict(idx=14, unit='code',  label='Primary Technique (code)'),
    'Technique_second':       dict(idx=15, unit='code',  label='Secondary Technique (code)'),
}

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

SPECIES_NAMES = list(COMMON_NAMES.keys())   # the 12 target species (no forage)

SP_COLORS = [
    '#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
    '#a65628','#f781bf','#999999','#1b9e77','#d95f02',
    '#7570b3','#e7298a',
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[pdp] device = {device}')


# ─────────────────────────────────────────────────────────────────────────────
# Args namespace (mirrors shap_analysis.py / main.py)
# ─────────────────────────────────────────────────────────────────────────────

class FIMArgs:
    dataset             = 'FIM_restoration_0412_stations_monthly'
    data_dir            = DATA_NPZ
    us_adj_file         = ADJ_PKL
    crop_id_to_fid      = FID_PKL
    time_intervals      = 1
    progress_indices    = []
    num_weather_vars    = 0
    num_management_vars = 0
    num_soil_vars       = 0
    num_extra_vars      = 0
    soil_depths         = 6
    no_soil             = True
    no_management       = True
    no_log_transform    = False
    encoder_type        = 'mlp'
    aggregator_type     = 'mean'
    n_layers            = 2
    dropout             = 0.5
    z_dim               = 64
    edge_feat_dim       = 3
    use_film            = True
    use_bayes           = True
    bayes_prior_std     = 1.0
    clip_grad           = 10.0
    seed                = 0
    c1                  = 1.0
    c2                  = 1.0
    check_freq          = 50
    beta_kl             = 0.0
    batch_size          = 128
    excluded_station_ids = set()
    train_start_year    = None
    other_species_cols  = []
    month_weight_path   = ''
    bce_weight          = 0.0
    dist_weights        = None
    output_names        = SPECIES_NAMES
    output_idx          = []
    feature_groups      = None
    group_names         = None
    n_env_feats         = 0
    individual_indices  = []
    group_slices        = []
    n_species_for_attn  = len(SPECIES_NAMES)
    means               = None
    stds                = None
    count_means         = None
    count_stds          = None
    test_year           = TEST_YEAR
    win_size            = WIN_SIZE
    n_windows           = N_WINDOWS
    length              = 1
    forage_species      = 'Lagodon rhomboides_A,Lagodon rhomboides_R'
    transform           = 'log1p' if RUN_ID in ('run_h', 'run_k', 'run_k2') else 'root4'


# ─────────────────────────────────────────────────────────────────────────────
# Species / feature metadata (same as shap_analysis.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_species_info(spp_pkl_path, args):
    with open(spp_pkl_path, 'rb') as f:
        spp_data = pickle.load(f)

    if isinstance(spp_data, dict):
        target_spp        = spp_data['target_species']
        community_species = spp_data.get('feature_species', [])
        n_env             = int(spp_data.get('n_env_features', 0))
    else:
        target_spp        = [s for s in spp_data if s.endswith(('_A', '_R', '_SA', '_J'))]
        community_species = [s for s in spp_data if s not in set(target_spp)]
        n_env             = 0

    forage_set   = {s.strip() for s in args.forage_species.split(',') if s.strip()}
    forage_in_t  = [s for s in target_spp if s in forage_set]
    target_spp_f = [s for s in target_spp if s not in forage_set]
    all_cols     = list(range(2, 2 + len(target_spp)))
    forage_cols  = [2 + i for i, s in enumerate(target_spp) if s in forage_set]
    target_cols  = [c for c in all_cols if c not in forage_cols]
    target_spp   = target_spp_f

    args.output_names       = target_spp
    args.output_idx         = target_cols
    args.other_species_cols = forage_cols
    args.n_env_feats        = n_env

    n_forage    = len(forage_in_t)
    n_community = len(community_species)
    o               = n_forage
    community_start = o + n_env
    community_end   = community_start + n_community
    mid_habitat     = o + 14 + 7
    n_prey    = int(spp_data.get('n_prey_community',    n_community // 2))
    n_nonprey = int(spp_data.get('n_nonprey_community', n_community - n_prey))
    prey_end  = community_start + n_prey

    args.individual_indices = []
    args.group_slices = [
        (o+0,          o+6,          'env'),
        (o+6,          o+14,         'restoration'),
        (o+14,         mid_habitat,  'habitat_structured'),
        (mid_habitat,  o+28,         'habitat_open'),
        (o+28,         o+35,         'shoreline'),
        (o+35,         o+38,         'water_effort'),
        (o+38,         o+52,         'bycatch'),
        (o+52,         o+58,         'na_indicators'),
        (community_start, prey_end,  'prey_community'),
        (prey_end,  community_end,   'nonprey_community'),
    ]
    if n_forage > 0:
        args.group_slices.insert(0, (0, n_forage, 'forage'))

    return args


# ─────────────────────────────────────────────────────────────────────────────
# Graph & model helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_and_loader(adj, batch_size=128):
    sp_adj = sp.coo_matrix(adj)
    g      = dgl.from_scipy(sp_adj).to(device)
    g      = dgl.add_self_loop(g)
    N      = g.num_nodes()
    sampler    = dgl.dataloading.MultiLayerNeighborSampler([10, 10])
    nodeloader = dgl.dataloading.DataLoader(
        g, torch.arange(N).to(device), sampler,
        batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=0, device=device)
    return g, nodeloader


def find_latest_checkpoint(ckpt_dir):
    files  = [f for f in os.listdir(ckpt_dir) if f.startswith('model-')]
    if not files:
        raise FileNotFoundError(f'No model-* files in {ckpt_dir}')
    latest = sorted(files, key=lambda x: int(x.split('-')[1]))[-1]
    path   = os.path.join(ckpt_dir, latest)
    print(f'[pdp] Using checkpoint: {path}')
    return path


def load_model(ckpt_path, in_dim, out_dim, args):
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model = GAT_RNN_V2(args, in_dim, out_dim).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[pdp] Model loaded — {n_params:,} parameters')
    return model


def attach_dist_weights(blocks, dist_weights, edge_feat_dim):
    if dist_weights is None:
        return
    default = np.ones(edge_feat_dim, dtype=np.float32)
    for blk in blocks:
        es, ed  = blk.edges()
        src_ids = blk.srcdata[dgl.NID].cpu().numpy()
        dst_ids = blk.dstdata[dgl.NID].cpu().numpy()
        feats   = []
        for s, d in zip(es.cpu().numpy(), ed.cpu().numpy()):
            val = dist_weights.get((int(src_ids[s]), int(dst_ids[d])), default)
            val = np.asarray(val, dtype=np.float32).ravel()[:edge_feat_dim]
            if val.shape[0] < edge_feat_dim:
                val = np.pad(val, (0, edge_feat_dim - val.shape[0]))
            feats.append(val)
        blk.edata['dist_w'] = torch.tensor(
            np.stack(feats), dtype=torch.float32, device=blk.device)


# ─────────────────────────────────────────────────────────────────────────────
# Windowed data builder (same as train.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_merged_windows(X_dict, Y_dict, county_set, T,
                         win_size=WIN_SIZE, n_windows=N_WINDOWS):
    _sample_station = county_set[0]
    _sample_year    = next(iter(X_dict[_sample_station]))
    n_feat    = X_dict[_sample_station][_sample_year].shape[1]
    n_species = Y_dict[_sample_station][_sample_year].shape[1]
    n_counties = len(county_set)
    window_XY = {}
    for w in range(1, n_windows + 1):
        year_end   = T - (n_windows - w)
        year_start = year_end - win_size + 1
        window_years = list(range(year_start, year_end + 1))
        X_sum  = np.zeros((n_counties, 12, n_feat),    dtype=np.float64)
        Y_sum  = np.zeros((n_counties, 12, n_species), dtype=np.float64)
        counts = np.zeros((n_counties, 1,  1),          dtype=np.float64)
        for ci, station in enumerate(county_set):
            s_xdict = X_dict.get(station, {})
            s_ydict = Y_dict.get(station, {})
            n_valid = 0
            for yr in window_years:
                if yr in s_xdict:
                    X_sum[ci] += s_xdict[yr]
                    Y_sum[ci] += s_ydict[yr]
                    n_valid   += 1
            counts[ci, 0, 0] = n_valid
        valid_mask = counts > 0
        X_merged = np.where(valid_mask, X_sum / np.where(counts > 0, counts, 1), 0.0)
        Y_merged = np.where(valid_mask, Y_sum / np.where(counts > 0, counts, 1), 0.0)
        window_XY[w] = (
            X_merged.astype(np.float32),
            Y_merged.astype(np.float32),
            list(county_set),
        )
    return window_XY


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, nodeloader, windowed_XY_T, county_set,
                  args, dist_weights, x_perturb_fn=None):
    """
    Full inference pass.  x_perturb_fn(X_merged_np) → X_merged_np replaces
    feature values before stacking (used for PDP grid sweeps).
    Returns pred_count [N, 12, n_out], true_count [N, 12, n_out].
    """
    model.eval()
    n_windows = len(windowed_XY_T)
    N         = len(county_set)
    n_out     = windowed_XY_T[1][1].shape[-1]
    preds     = torch.zeros(N, 12, n_out)
    trues     = torch.zeros(N, 12, n_out)

    X_wins, Y_wins = [], []
    for w in range(1, n_windows + 1):
        X_w, Y_w, _ = windowed_XY_T[w]
        if x_perturb_fn is not None:
            X_w = x_perturb_fn(X_w)
        X_wins.append(X_w)
        Y_wins.append(Y_w)

    X_all_np = np.stack(X_wins, axis=1)   # [N, n_windows, 12, n_feat]
    Y_all_np = np.stack(Y_wins, axis=1)   # [N, n_windows, 12, n_out]

    for (in_nodes, out_nodes, blocks) in nodeloader:
        in_idx  = in_nodes.cpu().numpy()
        out_idx = out_nodes.cpu().numpy()
        X_src = torch.tensor(X_all_np[in_idx],  dtype=torch.float32, device=device)
        Y_dst = torch.tensor(Y_all_np[out_idx],  dtype=torch.float32, device=device)
        Y_std  = (Y_dst - args.means) / args.stds
        blocks = [b.int().to(device) for b in blocks]
        attach_dist_weights(blocks, dist_weights, args.edge_feat_dim)
        pred_std = model(blocks, X_src, Y_std[:, :-1])
        pred     = pred_std * args.stds + args.means
        p_last   = torch.clamp(torch.expm1(pred[:, -1]), min=0.0)
        t_last   = torch.clamp(torch.expm1(Y_dst[:, -1]), min=0.0)
        preds[out_nodes.cpu()] = p_last.cpu()
        trues[out_nodes.cpu()] = t_last.cpu()

    return preds.numpy(), trues.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# X stats recovery for axis back-transform
# ─────────────────────────────────────────────────────────────────────────────

def recover_X_stats(data, args):
    """
    Re-compute X mean / std over known training years from raw data,
    mirroring what get_X_Y_2D does — so we can back-transform grid values
    to physical units for axis labels.
    Returns X_mean [in_dim], X_std [in_dim].
    """
    years_all = data[:, 1].astype(int)
    known     = years_all < (args.test_year - 1)

    forage_cols = args.other_species_cols
    n_out_cols  = len(args.output_idx)
    if forage_cols:
        feature_start = 2 + n_out_cols + len(forage_cols)
    else:
        feature_start = max(8, 2 + n_out_cols)

    # Build raw X (same construction as data_utils.py)
    X_raw = data[:, feature_start:]
    if forage_cols:
        X_raw = np.concatenate([data[:, forage_cols], X_raw], axis=1)

    # Ybar columns use the same mean/std as Y itself; we skip back-transform there
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        X_mean = np.nanmean(X_raw[known], axis=0)
        X_std  = np.nanstd( X_raw[known], axis=0)
    X_std[(X_std < 1e-6) | np.isnan(X_std)] = 1.0
    X_mean[np.isnan(X_mean)] = 0.0
    return X_mean, X_std


# ─────────────────────────────────────────────────────────────────────────────
# 1-D PDP
# ─────────────────────────────────────────────────────────────────────────────

def compute_1d_pdp(model, nodeloader, windowed_XY_T, county_set,
                   args, dist_weights, feat_idx, n_grid,
                   X_mean_raw, X_std_raw):
    """
    Returns:
      grid_orig  [n_grid]           values on physical scale (for axis labels)
      pdp_mat    [n_grid, n_species] mean predicted count per species
    """
    # Build grid in normalized space from the actual data range
    X_wins = [windowed_XY_T[w][0] for w in range(1, len(windowed_XY_T) + 1)]
    X_all  = np.stack(X_wins, axis=1)   # [N, n_windows, 12, n_feat]
    feat_vals_std = X_all[..., feat_idx].ravel()
    lo = float(np.percentile(feat_vals_std, 5))
    hi = float(np.percentile(feat_vals_std, 95))
    grid_std  = np.linspace(lo, hi, n_grid)

    # Back-transform for axis labels
    n_raw = len(X_mean_raw)
    if feat_idx < n_raw:
        grid_orig = grid_std * X_std_raw[feat_idx] + X_mean_raw[feat_idx]
    else:
        grid_orig = grid_std   # Ybar columns — stay normalized

    pdp_mat = np.zeros((n_grid, len(args.output_names)))

    for i, v in enumerate(grid_std):
        def perturb(X_merged_np, _idx=feat_idx, _v=v):
            X_out = X_merged_np.copy()
            X_out[..., _idx] = _v
            return X_out

        pred_c, _ = run_inference(
            model, nodeloader, windowed_XY_T, county_set,
            args, dist_weights, x_perturb_fn=perturb
        )
        # Average over stations and months → [n_species]
        pdp_mat[i] = pred_c.mean(axis=(0, 1))

        if (i + 1) % 5 == 0:
            print(f'    grid {i+1}/{n_grid}', flush=True)

    return grid_orig, pdp_mat


# ─────────────────────────────────────────────────────────────────────────────
# Bay-level PDP (population total per bay)
# ─────────────────────────────────────────────────────────────────────────────

def compute_1d_pdp_by_bay(model, nodeloader, windowed_XY_T, county_set,
                          args, dist_weights, feat_idx, n_grid,
                          X_mean_raw, X_std_raw, station_bay_map):
    """
    Like compute_1d_pdp but returns per-bay TOTAL population (sum over stations
    and months) instead of the grand mean.

    Parameters
    ----------
    station_bay_map : dict  {station_id (int) → bay_label (str)}
        Maps each position in county_set to its bay.

    Returns
    -------
    grid_orig  : np.ndarray [n_grid]           physical-scale grid values
    bay_labels : list[str]                     sorted bay names
    pdp_bay    : np.ndarray [n_grid, n_bays, n_species]
                 total predicted count per (grid_value, bay, species)
    """
    X_wins = [windowed_XY_T[w][0] for w in range(1, len(windowed_XY_T) + 1)]
    X_all  = np.stack(X_wins, axis=1)
    feat_vals_std = X_all[..., feat_idx].ravel()
    lo = float(np.percentile(feat_vals_std, 5))
    hi = float(np.percentile(feat_vals_std, 95))
    grid_std  = np.linspace(lo, hi, n_grid)

    n_raw = len(X_mean_raw)
    if feat_idx < n_raw:
        grid_orig = grid_std * X_std_raw[feat_idx] + X_mean_raw[feat_idx]
    else:
        grid_orig = grid_std

    # Build bay index arrays  (county_set order matches run_inference output)
    bay_labels = sorted(set(station_bay_map.values()))
    bay_to_idx = {b: i for i, b in enumerate(bay_labels)}
    station_bay_idx = np.array(
        [bay_to_idx[station_bay_map.get(s, bay_labels[0])] for s in county_set],
        dtype=int
    )   # [N_stations]
    n_bays    = len(bay_labels)
    n_species = len(args.output_names)

    pdp_bay = np.zeros((n_grid, n_bays, n_species))

    for i, v in enumerate(grid_std):
        def perturb(X_np, _idx=feat_idx, _v=v):
            X_out = X_np.copy()
            X_out[..., _idx] = _v
            return X_out

        pred_c, _ = run_inference(
            model, nodeloader, windowed_XY_T, county_set,
            args, dist_weights, x_perturb_fn=perturb
        )
        # pred_c shape: [N_stations, 12_months, n_species]
        # sum over months → [N_stations, n_species]
        station_totals = pred_c.sum(axis=1)
        # accumulate into bays
        for b_idx in range(n_bays):
            mask = (station_bay_idx == b_idx)
            pdp_bay[i, b_idx, :] = station_totals[mask].sum(axis=0)

        if (i + 1) % 5 == 0:
            print(f'    grid {i+1}/{n_grid}', flush=True)

    return grid_orig, bay_labels, pdp_bay


def plot_restoration_by_species(pdp_bay_results, feat_keys, output_names,
                                 bay_labels, test_year, out_dir):
    """
    For each restoration predictor: one figure with one subplot per species (3×4).
    Grouped bars per grid value, one bar per bay.

    pdp_bay_results : dict  fname → (grid_orig, bay_labels, pdp_bay [n_grid, n_bays, n_sp])
    Saves to out_dir/pdp_restoration_pop_{fname}.png
    """
    BAY_COLORS  = {'AP': '#1f77b4', 'TB': '#ff7f0e',
                   'CH': '#2ca02c', 'CK': '#d62728'}
    BAY_NAMES   = {'AP': 'Apalachicola Bay', 'TB': 'Tampa Bay',
                   'CH': 'Charlotte Harbor', 'CK': 'Cedar Key'}

    N_BARS  = 8     # subsample grid to this many x-positions for readability
    n_sp    = len(output_names)
    n_cols  = 3
    n_rows  = int(np.ceil(n_sp / n_cols))

    for fname in feat_keys:
        finfo = RESTORATION_FEATURES[fname]
        grid_orig, bays, pdp_bay = pdp_bay_results[fname]

        # Subsample evenly to N_BARS positions
        n_grid  = len(grid_orig)
        idx_sub = np.round(np.linspace(0, n_grid - 1, min(N_BARS, n_grid))).astype(int)
        grid_sub  = grid_orig[idx_sub]          # [N_BARS]
        pdp_sub   = pdp_bay[idx_sub]            # [N_BARS, n_bays, n_sp]

        n_bays   = len(bays)
        bar_w    = 0.8 / n_bays                 # width of each individual bar
        x_base   = np.arange(len(idx_sub))      # integer positions for groups

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5.5 * n_cols, 4 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        for si, sp in enumerate(output_names):
            ax = axes_flat[si]
            cn = COMMON_NAMES.get(sp, sp)

            for bi, bay in enumerate(bays):
                color  = BAY_COLORS.get(bay, f'C{bi}')
                label  = BAY_NAMES.get(bay, bay)
                offset = (bi - (n_bays - 1) / 2) * bar_w
                ax.bar(x_base + offset, pdp_sub[:, bi, si],
                       width=bar_w * 0.92, color=color, label=label,
                       alpha=0.85, edgecolor='white', linewidth=0.3)

            # x-tick labels: formatted grid values
            tick_labels = [f'{v:.2g}' for v in grid_sub]
            ax.set_xticks(x_base)
            ax.set_xticklabels(tick_labels, rotation=35, ha='right', fontsize=7)
            ax.set_title(cn, fontsize=10, fontweight='bold')
            ax.set_xlabel(finfo['label'], fontsize=8)
            ax.set_ylabel('Total predicted population\n(fish)', fontsize=7.5)
            ax.tick_params(axis='y', labelsize=8)
            if si == 0:
                ax.legend(fontsize=8, loc='upper left')

        for j in range(n_sp, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            f'Effect on Bay Population — {finfo["label"]}  (test year {test_year})',
            fontsize=12, fontweight='bold', y=1.01
        )
        fig.tight_layout()
        out_path = os.path.join(out_dir, f'pdp_restoration_pop_{fname}')
        _save_pngpdf(fig, out_path)
        plt.close(fig)
        print(f'  Saved → {out_path}.png')


# ─────────────────────────────────────────────────────────────────────────────
# 2-D PDP
# ─────────────────────────────────────────────────────────────────────────────

def compute_group_pdp(model, nodeloader, windowed_XY_T, county_set,
                      args, dist_weights, group_cols, n_grid):
    """
    Group-level PDP: scale ALL columns in `group_cols` by a common multiplier.

    The grid ranges from 0× (all columns zeroed) to 2× (all columns doubled)
    relative to their observed values, so the x-axis is interpretable as
    "fraction of observed community abundance".

    Returns:
      grid_scale  [n_grid]           multiplier values (0 → 2)
      pdp_mat     [n_grid, n_species] mean predicted count per species
    """
    grid_scale = np.linspace(0.0, 2.0, n_grid)
    group_cols = np.asarray(group_cols, dtype=int)
    pdp_mat = np.zeros((n_grid, len(args.output_names)))

    for i, scale in enumerate(grid_scale):
        def perturb(X_merged_np, _cols=group_cols, _s=scale):
            X_out = X_merged_np.copy()
            X_out[..., _cols] = X_out[..., _cols] * _s
            return X_out

        pred_c, _ = run_inference(
            model, nodeloader, windowed_XY_T, county_set,
            args, dist_weights, x_perturb_fn=perturb
        )
        pdp_mat[i] = pred_c.mean(axis=(0, 1))

        if (i + 1) % 5 == 0:
            print(f'    grid {i+1}/{n_grid}', flush=True)

    return grid_scale, pdp_mat


def compute_2d_pdp(model, nodeloader, windowed_XY_T, county_set,
                   args, dist_weights, feat_idx_1, feat_idx_2,
                   n_grid, X_mean_raw, X_std_raw, sp_idx):
    """
    Vary two features jointly for a single species.
    Returns g1_orig [n_grid], g2_orig [n_grid], pdp2d [n_grid, n_grid].
    """
    X_wins = [windowed_XY_T[w][0] for w in range(1, len(windowed_XY_T) + 1)]
    X_all  = np.stack(X_wins, axis=1)

    def grid_for(idx):
        vals = X_all[..., idx].ravel()
        lo   = float(np.percentile(vals, 5))
        hi   = float(np.percentile(vals, 95))
        g    = np.linspace(lo, hi, n_grid)
        n_r  = len(X_mean_raw)
        orig = g * X_std_raw[idx] + X_mean_raw[idx] if idx < n_r else g
        return g, orig

    g1, g1_orig = grid_for(feat_idx_1)
    g2, g2_orig = grid_for(feat_idx_2)
    pdp2d = np.zeros((n_grid, n_grid))

    for i, v1 in enumerate(g1):
        for j, v2 in enumerate(g2):
            def perturb(X_np, _i1=feat_idx_1, _v1=v1, _i2=feat_idx_2, _v2=v2):
                X_out = X_np.copy()
                X_out[..., _i1] = _v1
                X_out[..., _i2] = _v2
                return X_out

            pred_c, _ = run_inference(
                model, nodeloader, windowed_XY_T, county_set,
                args, dist_weights, x_perturb_fn=perturb
            )
            # Mean over stations and months for this species
            pdp2d[i, j] = pred_c[..., sp_idx].mean()

        print(f'  [2D] row {i+1}/{n_grid}', flush=True)

    return g1_orig, g2_orig, pdp2d


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_1d_all_species(grid, pdp_mat, feat_label, output_names,
                        ref_lines, title, out_path):
    """Multi-line plot: one line per species."""
    n_sp = pdp_mat.shape[1]
    cmap = cm.get_cmap('tab20', n_sp)
    fig, ax = plt.subplots(figsize=(11, 5))
    for s in range(n_sp):
        sp  = output_names[s]
        cn  = COMMON_NAMES.get(sp, sp)
        ax.plot(grid, pdp_mat[:, s], color=SP_COLORS[s % len(SP_COLORS)],
                lw=1.4, label=cn, alpha=0.85)
    for xval, col, lbl in ref_lines:
        ax.axvline(xval, color=col, lw=0.9, ls='--', alpha=0.55, label=lbl)
    ax.set_xlabel(feat_label, fontsize=11)
    ax.set_ylabel('Mean predicted count', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=7, ncol=3, bbox_to_anchor=(1.01, 1), loc='upper left')
    fig.tight_layout()
    _save_pngpdf(fig, out_path)
    plt.close(fig)
    print(f'  Saved → {out_path}')


def plot_4panel(pdp_results, feat_keys, output_names, out_path):
    """2×2 panel: each cell = one feature, all species (thin lines)."""
    n = len(feat_keys)
    n_cols = 2
    n_rows = (n + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 4 * n_rows), squeeze=False)
    cmap = cm.get_cmap('tab20', len(output_names))
    for fi, fname in enumerate(feat_keys):
        ax  = axes[fi // n_cols][fi % n_cols]
        grid, pdp_mat = pdp_results[fname]
        finfo = FEATURES[fname]
        for s in range(pdp_mat.shape[1]):
            sp = output_names[s]
            cn = COMMON_NAMES.get(sp, sp)
            ax.plot(grid, pdp_mat[:, s], color=SP_COLORS[s % len(SP_COLORS)],
                    lw=1.1, label=cn, alpha=0.75)
        ax.set_title(finfo['label'], fontsize=10, fontweight='bold')
        ax.set_xlabel(finfo['label'], fontsize=8)
        ax.set_ylabel('Mean predicted count', fontsize=8)
        ax.tick_params(labelsize=8)
        if fi == 0:
            ax.legend(fontsize=5, ncol=2, loc='upper left')
    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)
    fig.suptitle('Partial Dependence Plots — Environmental & Effort Features (2024)',
                 fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()
    _save_pngpdf(fig, out_path)
    plt.close(fig)
    print(f'  Saved → {out_path}')


def plot_restoration_panel(pdp_results, feat_keys, output_names, test_year, out_path):
    """
    2×4 panel: each subplot = one restoration predictor, all 12 species as lines.
    Sorted so quantitative features (n_projects, Acres_Restored, targeted_habitat_num,
    Restoration_technique) appear first (row 1), categorical codes last (row 2).
    """
    n = len(feat_keys)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 4 * n_rows), squeeze=False)

    for fi, fname in enumerate(feat_keys):
        ax   = axes[fi // n_cols][fi % n_cols]
        grid, pdp_mat = pdp_results[fname]
        finfo = RESTORATION_FEATURES[fname]
        for s in range(pdp_mat.shape[1]):
            sp = output_names[s]
            cn = COMMON_NAMES.get(sp, sp)
            ax.plot(grid, pdp_mat[:, s],
                    color=SP_COLORS[s % len(SP_COLORS)],
                    lw=1.3, label=cn, alpha=0.85)
        ax.set_title(finfo['label'], fontsize=10, fontweight='bold')
        ax.set_xlabel(finfo['label'], fontsize=8)
        ax.set_ylabel('Mean predicted count', fontsize=8)
        ax.tick_params(labelsize=7)
        if fi == 0:
            ax.legend(fontsize=5.5, ncol=2, loc='upper right')

    # Hide unused axes
    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    fig.suptitle(
        f'Partial Dependence on Restoration Predictors  (test year {test_year})',
        fontsize=13, fontweight='bold', y=1.01
    )
    fig.tight_layout()
    _save_pngpdf(fig, out_path)
    plt.close(fig)
    print(f'  Saved → {out_path}')


def plot_restoration_top3(pdp_results, output_names, test_year, out_path):
    """
    Focused 1×3 panel for the three most interpretable quantitative restoration
    predictors: n_projects, Acres_Restored, targeted_habitat_num.
    Each subplot shows all 12 species as coloured lines with a legend.
    """
    feat_keys = ['n_projects', 'Acres_Restored', 'targeted_habitat_num']
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=False)
    for ax, fname in zip(axes, feat_keys):
        grid, pdp_mat = pdp_results[fname]
        finfo = RESTORATION_FEATURES[fname]
        for s in range(pdp_mat.shape[1]):
            sp = output_names[s]
            cn = COMMON_NAMES.get(sp, sp)
            ax.plot(grid, pdp_mat[:, s],
                    color=SP_COLORS[s % len(SP_COLORS)],
                    lw=1.5, label=cn, alpha=0.9)
        ax.set_xlabel(finfo['label'], fontsize=11)
        ax.set_ylabel('Mean predicted count (fish/obs)', fontsize=10)
        ax.set_title(finfo['label'], fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=9)
    axes[-1].legend(fontsize=7, ncol=1, bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.suptitle(
        f'Partial Dependence on Key Restoration Predictors  (test year {test_year})',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    _save_pngpdf(fig, out_path)
    plt.close(fig)
    print(f'  Saved → {out_path}')


def compute_2d_pdp_feat_x_group(model, nodeloader, windowed_XY_T, county_set,
                                args, dist_weights, feat_idx, group_cols,
                                n_grid, X_mean_raw, X_std_raw, sp_idx):
    """
    2-D PDP: axis 1 = single feature (physical units), axis 2 = group scaling (0–2×).

    Returns:
      feat_grid_orig [n_grid]  physical-scale values for the single feature
      scale_grid     [n_grid]  multiplier values for the group (0 → 2)
      pdp2d          [n_grid, n_grid]  mean predicted count for sp_idx
    """
    X_wins = [windowed_XY_T[w][0] for w in range(1, len(windowed_XY_T) + 1)]
    X_all  = np.stack(X_wins, axis=1)
    group_cols = np.asarray(group_cols, dtype=int)

    vals = X_all[..., feat_idx].ravel()
    lo   = float(np.percentile(vals, 5))
    hi   = float(np.percentile(vals, 95))
    feat_grid_std = np.linspace(lo, hi, n_grid)
    n_r = len(X_mean_raw)
    feat_grid_orig = (feat_grid_std * X_std_raw[feat_idx] + X_mean_raw[feat_idx]
                      if feat_idx < n_r else feat_grid_std)

    scale_grid = np.linspace(0.0, 2.0, n_grid)
    pdp2d = np.zeros((n_grid, n_grid))

    for i, fv in enumerate(feat_grid_std):
        for j, sv in enumerate(scale_grid):
            def perturb(X_np, _fi=feat_idx, _fv=fv, _gc=group_cols, _sv=sv):
                X_out = X_np.copy()
                X_out[..., _fi] = _fv
                X_out[..., _gc] = X_out[..., _gc] * _sv
                return X_out

            pred_c, _ = run_inference(
                model, nodeloader, windowed_XY_T, county_set,
                args, dist_weights, x_perturb_fn=perturb
            )
            pdp2d[i, j] = pred_c[..., sp_idx].mean()

        print(f'  [2D-group] row {i+1}/{n_grid}', flush=True)

    return feat_grid_orig, scale_grid, pdp2d


def plot_2d_pdp(g1_orig, g2_orig, pdp2d, feat1_label, feat2_label,
                sp_label, out_path):
    """2-D PDP heatmap with contour lines."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.pcolormesh(g2_orig, g1_orig, pdp2d, cmap='YlOrRd', shading='auto')
    ax.contour(g2_orig, g1_orig, pdp2d, levels=5,
               colors='white', linewidths=0.6, alpha=0.7)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Mean predicted count', fontsize=9)
    ax.set_xlabel(feat2_label, fontsize=10)
    ax.set_ylabel(feat1_label, fontsize=10)
    ax.set_title(f'2-D PDP: {sp_label}\n({feat1_label} × {feat2_label})',
                 fontsize=11)
    fig.tight_layout()
    _save_pngpdf(fig, out_path)
    plt.close(fig)
    print(f'  Saved → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print('=' * 64)
    print('GNN-RNN v2 FIM — Partial Dependence Plot Analysis')
    print(f'Test year : {TEST_YEAR}   n_grid={N_GRID}   n_grid_2D={N_GRID_2D}')
    print('=' * 64)

    args = FIMArgs()

    # ── Species metadata ──────────────────────────────────────────────────────
    print('\n[pdp] Loading species metadata …')
    args = load_species_info(SPP_PKL, args)
    print(f'[pdp] {len(args.output_names)} target species')

    # ── Raw data (for X stats recovery) ──────────────────────────────────────
    raw_data = np.load(DATA_NPZ)['data']
    X_mean_raw, X_std_raw = recover_X_stats(raw_data, args)

    # ── Load + normalize data ─────────────────────────────────────────────────
    print('\n[pdp] Loading data …')
    (X_dict, Y_dict, _, adj, _,
     min_year, max_year, county_set, _, _) = \
        get_X_Y_2D(raw_data, args, device)
    print(f'[pdp] Stations: {len(county_set)}, years: {min_year}–{max_year}')

    # ── Graph ─────────────────────────────────────────────────────────────────
    print('\n[pdp] Building graph …')
    g, nodeloader = build_graph_and_loader(adj, batch_size=128)
    print(f'[pdp] Graph: {g.num_nodes()} nodes, {g.num_edges()} edges')

    # ── Distance weights ──────────────────────────────────────────────────────
    dist_weights = None
    if os.path.exists(DIST_PKL):
        with open(DIST_PKL, 'rb') as f:
            dist_weights = pickle.load(f)
        _s = next(iter(dist_weights.values()))
        args.edge_feat_dim = int(np.asarray(_s).ravel().shape[0])
        print(f'[pdp] Edge feat dim: {args.edge_feat_dim}')

    # ── Model ─────────────────────────────────────────────────────────────────
    print('\n[pdp] Loading model …')
    _s0      = county_set[0]
    _y0      = next(iter(X_dict[_s0]))
    in_dim   = X_dict[_s0][_y0].shape[-1]
    out_dim  = len(args.output_names)
    print(f'[pdp] in_dim={in_dim}, out_dim={out_dim}')
    ckpt_path = find_latest_checkpoint(CKPT_DIR)
    model     = load_model(ckpt_path, in_dim, out_dim, args)

    # ── Build windowed data for test year ──────────────────────────────────────
    print(f'\n[pdp] Building windowed data for test year {TEST_YEAR} …')
    windowed_XY_T = build_merged_windows(
        X_dict, Y_dict, county_set, TEST_YEAR,
        win_size=WIN_SIZE, n_windows=N_WINDOWS
    )
    sample_X = windowed_XY_T[1][0]
    print(f'[pdp] {N_WINDOWS} windows × {sample_X.shape[0]} stations '
          f'× 12 months × {sample_X.shape[-1]} features')

    # ─────────────────────────────────────────────────────────────────────────
    # 1-D PDPs  (skipped when RESTORATION_ONLY=1)
    # ─────────────────────────────────────────────────────────────────────────
    target_feats_1d = ['Temperature', 'Salinity', 'DissolvedO2', 'StartDepth']
    pdp_results = {}
    records     = []

    if not RESTORATION_ONLY:
        for fname in target_feats_1d:
            finfo = FEATURES[fname]
            print(f'\n[pdp] 1-D PDP: {fname} (idx={finfo["idx"]}) …')
            grid_orig, pdp_mat = compute_1d_pdp(
                model, nodeloader, windowed_XY_T, county_set,
                args, dist_weights, finfo['idx'], N_GRID,
                X_mean_raw, X_std_raw
            )
            pdp_results[fname] = (grid_orig, pdp_mat)
            for gi, gv in enumerate(grid_orig):
                for si, sp_name in enumerate(args.output_names):
                    records.append(dict(
                        feature=fname, grid_value=round(float(gv), 4),
                        species=COMMON_NAMES.get(sp_name, sp_name),
                        mean_count=round(float(pdp_mat[gi, si]), 6)
                    ))

        # Forage species PDP
        for fname in ['Pinfish_A', 'Pinfish_R']:
            finfo = FEATURES[fname]
            print(f'\n[pdp] 1-D PDP: {fname} (idx={finfo["idx"]}) …')
            grid_orig, pdp_mat = compute_1d_pdp(
                model, nodeloader, windowed_XY_T, county_set,
                args, dist_weights, finfo['idx'], N_GRID,
                X_mean_raw, X_std_raw
            )
            pdp_results[fname] = (grid_orig, pdp_mat)
            for gi, gv in enumerate(grid_orig):
                for si, sp_name in enumerate(args.output_names):
                    records.append(dict(
                        feature=fname, grid_value=round(float(gv), 4),
                        species=COMMON_NAMES.get(sp_name, sp_name),
                        mean_count=round(float(pdp_mat[gi, si]), 6)
                    ))

        # Save CSV
        csv_path = os.path.join(FIGURES_DIR, 'pdp_values.csv')
        pd.DataFrame(records).to_csv(csv_path, index=False)
        print(f'\n  Saved → {csv_path}')
    else:
        print('[pdp] RESTORATION_ONLY=1 → skipping env/forage 1-D PDPs')

    # ── Plot 1: 4-panel environmental features ────────────────────────────────
    if not RESTORATION_ONLY:
        print('\n[pdp] Plotting 4-panel env/water figure …')
        plot_4panel(pdp_results, target_feats_1d, args.output_names,
                    os.path.join(FIGURES_DIR, 'pdp_env_water.png'))

    if not RESTORATION_ONLY:
        # ── Plot 2: Temperature — all species ─────────────────────────────────────
        print('\n[pdp] Plotting Temperature PDP (all species) …')
        plot_1d_all_species(
            *pdp_results['Temperature'], 'Temperature (°C)',
            args.output_names,
            ref_lines=[(20, 'steelblue', '20°C (cool threshold)'),
                       (28, 'tomato',    '28°C (thermal stress)')],
            title=f'Partial Dependence on Temperature  (test year {TEST_YEAR})',
            out_path=os.path.join(FIGURES_DIR, 'pdp_temperature_all.png')
        )

    if not RESTORATION_ONLY:
        # ── Plot 3: Salinity — all species ────────────────────────────────────────
        print('\n[pdp] Plotting Salinity PDP (all species) …')
        plot_1d_all_species(
            *pdp_results['Salinity'], 'Salinity (PSU)',
            args.output_names,
            ref_lines=[(0.5, 'steelblue', '0.5 PSU (fresh)'),
                       (15,  'goldenrod', '15 PSU (mesohaline)'),
                       (30,  'tomato',    '30 PSU (polyhaline)')],
            title=f'Partial Dependence on Salinity  (test year {TEST_YEAR})',
            out_path=os.path.join(FIGURES_DIR, 'pdp_salinity_all.png')
        )

        # ── Plot 4: Forage (Pinfish) ──────────────────────────────────────────────
        print('\n[pdp] Plotting Pinfish (forage) PDP …')
        n_sp = len(args.output_names)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, fname, ftitle in zip(axes, ['Pinfish_A', 'Pinfish_R'],
                                     ['Pinfish Adult', 'Pinfish Recruit']):
            grid, pdp_mat = pdp_results[fname]
            for s in range(n_sp):
                sp = args.output_names[s]
                cn = COMMON_NAMES.get(sp, sp)
                ax.plot(grid, pdp_mat[:, s], color=SP_COLORS[s % len(SP_COLORS)],
                        lw=1.4, label=cn, alpha=0.85)
            ax.set_xlabel(FEATURES[fname]['label'], fontsize=10)
            ax.set_ylabel('Mean predicted count', fontsize=10)
            ax.set_title(f'Partial Dependence on {ftitle}', fontsize=11)
            ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1), loc='upper left')
        fig.suptitle(f'Forage (Pinfish) Effect on Target Species  (test year {TEST_YEAR})',
                     fontsize=12, fontweight='bold', y=1.01)
        fig.tight_layout()
        forage_path = os.path.join(FIGURES_DIR, 'pdp_forage.png')
        _save_pngpdf(fig, forage_path)
        plt.close(fig)
        print(f'  Saved → {forage_path}')

        # ─────────────────────────────────────────────────────────────────────────
        # Community PDPs: prey and non-prey nekton community
        # ─────────────────────────────────────────────────────────────────────────
        prey_slice    = next(((s, e) for s, e, n in args.group_slices
                              if n == 'prey_community'), None)
        nonprey_slice = next(((s, e) for s, e, n in args.group_slices
                              if n == 'nonprey_community'), None)

        community_groups = []
        if prey_slice:
            community_groups.append(('prey_community', list(range(*prey_slice)),
                                     'Prey Nekton Community'))
        if nonprey_slice:
            community_groups.append(('nonprey_community', list(range(*nonprey_slice)),
                                     'Non-prey Nekton Community'))

        for group_key, group_cols, group_label in community_groups:
            print(f'\n[pdp] Group PDP: {group_label} '
                  f'(cols {group_cols[0]}–{group_cols[-1]}, {len(group_cols)} species) …')
            grid_scale, pdp_mat = compute_group_pdp(
                model, nodeloader, windowed_XY_T, county_set,
                args, dist_weights, group_cols, N_GRID
            )
            pdp_results[group_key] = (grid_scale, pdp_mat)

            for gi, gv in enumerate(grid_scale):
                for si, sp_name in enumerate(args.output_names):
                    records.append(dict(
                        feature=group_key, grid_value=round(float(gv), 4),
                        species=COMMON_NAMES.get(sp_name, sp_name),
                        mean_count=round(float(pdp_mat[gi, si]), 6)
                    ))

            plot_1d_all_species(
                grid_scale, pdp_mat,
                f'{group_label} Abundance (× baseline)',
                args.output_names,
                ref_lines=[(1.0, 'grey', 'Baseline (1×)')],
                title=f'Partial Dependence on {group_label}  (test year {TEST_YEAR})',
                out_path=os.path.join(FIGURES_DIR, f'pdp_{group_key}.png')
            )

        # Re-save CSV with community PDPs included
        csv_path = os.path.join(FIGURES_DIR, 'pdp_values.csv')
        pd.DataFrame(records).to_csv(csv_path, index=False)
        print(f'\n  Updated → {csv_path}')

        # ── Plot: 2-panel community summary (prey + non-prey side by side) ───
        if prey_slice and nonprey_slice:
            print('\n[pdp] Plotting community 2-panel summary …')
            n_sp = len(args.output_names)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, gkey, gtitle in zip(axes,
                    ['prey_community', 'nonprey_community'],
                    ['Prey Nekton Community', 'Non-prey Nekton Community']):
                grid, pdp_mat = pdp_results[gkey]
                for s in range(n_sp):
                    sp = args.output_names[s]
                    cn = COMMON_NAMES.get(sp, sp)
                    ax.plot(grid, pdp_mat[:, s],
                            color=SP_COLORS[s % len(SP_COLORS)],
                            lw=1.4, label=cn, alpha=0.85)
                ax.axvline(1.0, color='grey', lw=0.9, ls='--', alpha=0.55)
                ax.set_xlabel(f'{gtitle} Abundance (× baseline)', fontsize=10)
                ax.set_ylabel('Mean predicted count', fontsize=10)
                ax.set_title(f'Partial Dependence on {gtitle}', fontsize=11)
                ax.legend(fontsize=7, ncol=2,
                          bbox_to_anchor=(1.01, 1), loc='upper left')
            fig.suptitle(f'Community Effect on Target Species  '
                         f'(test year {TEST_YEAR})',
                         fontsize=12, fontweight='bold', y=1.01)
            fig.tight_layout()
            comm_path = os.path.join(FIGURES_DIR, 'pdp_community.png')
            _save_pngpdf(fig, comm_path)
            plt.close(fig)
            print(f'  Saved → {comm_path}')

        # ─────────────────────────────────────────────────────────────────────────
        # 2-D PDPs: Temperature × Salinity for focal species
        # ─────────────────────────────────────────────────────────────────────────
        drum_sa_idx  = next((i for i, s in enumerate(args.output_names)
                             if 'ocellatus_SA' in s), None)
        drum_r_idx   = next((i for i, s in enumerate(args.output_names)
                             if 'ocellatus_R' in s), None)
        trout_a_idx  = next((i for i, s in enumerate(args.output_names)
                             if 'nebulosus_A' in s), None)
        trout_r_idx  = next((i for i, s in enumerate(args.output_names)
                             if 'nebulosus_R' in s), None)

        for sp_idx, sp_tag, sp_label in [
                (drum_sa_idx, 'reddrum_SA',  'Red Drum (Sub-adult)'),
                (drum_r_idx,  'reddrum_R',   'Red Drum (Recruit)'),
                (trout_a_idx, 'seatrout_A',  'Spotted Seatrout (Adult)'),
                (trout_r_idx, 'seatrout_R',  'Spotted Seatrout (Recruit)'),
        ]:
            if sp_idx is None:
                continue
            print(f'\n[pdp] 2-D PDP Temperature × Salinity for {sp_label} …')
            g1_orig, g2_orig, pdp2d = compute_2d_pdp(
                model, nodeloader, windowed_XY_T, county_set,
                args, dist_weights,
                feat_idx_1=FEATURES['Temperature']['idx'],
                feat_idx_2=FEATURES['Salinity']['idx'],
                n_grid=N_GRID_2D,
                X_mean_raw=X_mean_raw, X_std_raw=X_std_raw,
                sp_idx=sp_idx
            )
            plot_2d_pdp(g1_orig, g2_orig, pdp2d,
                        'Temperature (°C)', 'Salinity (PSU)', sp_label,
                        os.path.join(FIGURES_DIR, f'pdp_2d_{sp_tag}.png'))

        # ─────────────────────────────────────────────────────────────────────────
        # 2-D PDPs: feature × community group for focal species
        # ─────────────────────────────────────────────────────────────────────────
        prey_cols    = list(range(*prey_slice)) if prey_slice else []
        nonprey_cols = list(range(*nonprey_slice)) if nonprey_slice else []

        pdp_2d_combos = [
            # Red Drum (Sub-adult)
            (drum_sa_idx,  'reddrum_SA', 'Red Drum (Sub-adult)',
             'Temperature', prey_cols, 'Prey Community'),
            (drum_sa_idx,  'reddrum_SA', 'Red Drum (Sub-adult)',
             'Salinity', prey_cols, 'Prey Community'),
            (drum_sa_idx,  'reddrum_SA', 'Red Drum (Sub-adult)',
             'Temperature', nonprey_cols, 'Non-prey Community'),
            # Red Drum (Recruit)
            (drum_r_idx,   'reddrum_R', 'Red Drum (Recruit)',
             'Temperature', prey_cols, 'Prey Community'),
            (drum_r_idx,   'reddrum_R', 'Red Drum (Recruit)',
             'Salinity', prey_cols, 'Prey Community'),
            (drum_r_idx,   'reddrum_R', 'Red Drum (Recruit)',
             'Temperature', nonprey_cols, 'Non-prey Community'),
            # Spotted Seatrout (Recruit)
            (trout_r_idx,  'seatrout_R', 'Spotted Seatrout (Recruit)',
             'Temperature', prey_cols, 'Prey Community'),
            (trout_r_idx,  'seatrout_R', 'Spotted Seatrout (Recruit)',
             'Salinity', prey_cols, 'Prey Community'),
            (trout_r_idx,  'seatrout_R', 'Spotted Seatrout (Recruit)',
             'Salinity', nonprey_cols, 'Non-prey Community'),
            # Spotted Seatrout (Adult)
            (trout_a_idx,  'seatrout_A', 'Spotted Seatrout (Adult)',
             'Temperature', prey_cols, 'Prey Community'),
            (trout_a_idx,  'seatrout_A', 'Spotted Seatrout (Adult)',
             'Salinity', prey_cols, 'Prey Community'),
            (trout_a_idx,  'seatrout_A', 'Spotted Seatrout (Adult)',
             'DissolvedO2', prey_cols, 'Prey Community'),
        ]

        for sp_idx, sp_tag, sp_label, feat_key, g_cols, g_label in pdp_2d_combos:
            if sp_idx is None or not g_cols:
                continue
            finfo = FEATURES[feat_key]
            g_tag = g_label.lower().replace(' ', '_').replace('-', '')
            out_name = f'pdp_2d_{sp_tag}_{feat_key.lower()}_x_{g_tag}'
            print(f'\n[pdp] 2-D PDP: {sp_label} — {finfo["label"]} × {g_label} …')

            feat_grid, scale_grid, pdp2d = compute_2d_pdp_feat_x_group(
                model, nodeloader, windowed_XY_T, county_set,
                args, dist_weights,
                feat_idx=finfo['idx'], group_cols=g_cols,
                n_grid=N_GRID_2D,
                X_mean_raw=X_mean_raw, X_std_raw=X_std_raw,
                sp_idx=sp_idx
            )
            plot_2d_pdp(feat_grid, scale_grid, pdp2d,
                        finfo['label'],
                        f'{g_label} (× baseline)',
                        sp_label,
                        os.path.join(FIGURES_DIR, f'{out_name}.png'))

    # ── Final CSV save ────────────────────────────────────────────────────────
    csv_path = os.path.join(FIGURES_DIR, 'pdp_values.csv')
    if records:
        pd.DataFrame(records).to_csv(csv_path, index=False)
        print(f'\n  Saved → {csv_path}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 64)
    print(f'PDP ANALYSIS COMPLETE  (test year {TEST_YEAR})')
    print('=' * 64)
    if os.path.exists(csv_path):
        df_csv = pd.read_csv(csv_path)
        for fname in target_feats_1d + ['prey_community', 'nonprey_community']:
            sub = df_csv[df_csv.feature == fname]
            if sub.empty:
                continue
            ranges = sub.groupby('species')['mean_count'].agg(lambda x: x.max() - x.min())
            top3   = ranges.nlargest(3)
            print(f'\n  {fname} — species with widest response range:')
            for sp_name, rng in top3.items():
                print(f'    {sp_name:<40s}  Δcount = {rng:.4f}')

    print(f'\nAll outputs saved to: {os.path.abspath(FIGURES_DIR)}/')


if __name__ == '__main__':
    main()
