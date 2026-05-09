"""
shap_analysis.py — Permutation-based feature group importance for GAT_RNN_V2.

Approach: model-agnostic permutation importance at the group level.
  1. Load trained GAT_RNN_V2 checkpoint (2024 fold, MLP encoder)
  2. Load test data for 2024 and build year_XY tensors
  3. Run baseline inference → compute per-species MSE
  4. For each of the 10 feature groups: shuffle that group's columns across
     stations → re-run inference → importance = (new_MSE - baseline_MSE) / baseline_MSE
  5. Repeat 5 times, average importance
  6. Produce per-species heatmap and mean-importance bar chart

Output files:
  analysis/figures/shap_group_heatmap.png
  analysis/figures/shap_group_importance.png
  analysis/figures/shap_group_importance.csv

Run from the gnn-lstm-v2/ directory:
  python analysis/shap_analysis.py

Feature layout in X (152 features, 0-indexed after forage prepend):
  forage           [0 :2 ]  2 features  — Pinfish adult + recruit counts
  env              [2 :8 ]  6 features  — water quality
  restoration      [8 :16]  8 features  — restoration site metrics
  habitat_structured[16:23]  7 features  — seagrass, oyster, mangrove
  habitat_open     [23:30]  7 features  — unvegetated / open water
  shoreline        [30:37]  7 features  — shoreline type / length
  water_effort     [37:40]  3 features  — sampling effort
  bycatch          [40:54] 14 features  — bycatch species
  na_indicators    [54:60]  6 features  — missing-data flags
  community_A      [60:106] 46 features  — background community, first half
  community_B      [106:152] 46 features — background community, second half
"""

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
import dgl

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Path setup ────────────────────────────────────────────────────────────────
# Always resolve relative to this script so we can be called from anywhere.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GNN_DIR    = os.path.dirname(_SCRIPT_DIR)              # gnn-lstm-v2/
_REPO_DIR   = os.path.dirname(_GNN_DIR)                 # GNN-RNN-main/
for _p in [_GNN_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import GAT_RNN_V2
from utils import get_X_Y_2D

# ── Run-specific config (must come before FIGURES_DIR) ────────────────────────
RUN_ID    = os.environ.get('SHAP_RUN_ID', 'run_h')
_DS       = '0412G' if RUN_ID in ('run_g','run_h') else '0412'
_MAXEPOCH = '35' if RUN_ID == 'run_h' else ('50' if RUN_ID == 'run_g' else '30')

# ── Output directory ──────────────────────────────────────────────────────────
FIGURES_DIR = os.path.join(_SCRIPT_DIR, 'figures', RUN_ID, 'shap')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[shap] device = {device}')

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TEST_YEAR  = 2024
WIN_SIZE   = 3    # years per window (averaged before feeding to model)
N_WINDOWS  = 10   # outer LSTM windows (= sequence length seen by model)

# Paths (relative to gnn-lstm-v2/)
DATA_NPZ  = os.path.join(_GNN_DIR, f'../data/FIM_restoration_{_DS}_stations_monthly.npz')
ADJ_PKL   = os.path.join(_GNN_DIR, '../map/FIM_restoration_0412_v2_adj.pkl')
FID_PKL   = os.path.join(_GNN_DIR, '../map/FIM_restoration_0412_v2_fid_dict.pkl')
SPP_PKL   = os.path.join(_GNN_DIR, f'../data/FIM_restoration_{_DS}_species_names.pkl')
CKPT_DIR  = os.path.join(
    _GNN_DIR,
    f'model/FIM_restoration_{_DS}_stations_monthly/2024/'
    f'gat-rnn-v2-windowed_bs-128_lr-0.001_maxepoch-{_MAXEPOCH}_testyear-2024_win-3_nwin-10_seed-0'
)
DIST_PKL  = os.path.join(_GNN_DIR, '../map/FIM_restoration_0412_v2_dist_weights.pkl')

N_PERM = 5   # permutation repeats

# Feature groups are now built dynamically from args.group_slices in
# load_species_info() — using the *correct* prey_community / nonprey_community
# split from the species pkl, with `forage` merged into prey_community
# (forage species are part of the prey assemblage, not a separate concept).
# See `build_feature_groups_from_args()` below.

# ── Species metadata ──────────────────────────────────────────────────────────
SPECIES_NAMES = [
    'Archosargus probatocephalus_A',
    'Archosargus probatocephalus_R',
    'Callinectes sapidus_R',
    'Centropomus undecimalis_A',
    'Centropomus undecimalis_SA',
    'Cynoscion nebulosus_A',
    'Cynoscion nebulosus_R',
    'Lutjanus griseus_R',
    'Lutjanus griseus_SA',
    'Mycteroperca microlepis_SA',
    'Sciaenops ocellatus_R',
    'Sciaenops ocellatus_SA',
]

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


# ─────────────────────────────────────────────────────────────────────────────
# Minimal args namespace for get_X_Y_2D / GAT_RNN_V2
# ─────────────────────────────────────────────────────────────────────────────

class FIMArgs:
    """Attribute namespace that mimics argparse output expected by utils / model."""
    dataset             = f'FIM_restoration_{_DS}_stations_monthly'
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
    encoder_type        = 'mlp'          # checkpoint uses MLP encoder
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
    c1                  = 1.0    # auxiliary loss weight (not used in eval)
    c2                  = 1.0    # primary loss weight   (not used in eval)
    check_freq          = 50     # logging frequency     (not used in eval)
    beta_kl             = 0.0    # KL weight             (not used in eval)
    batch_size          = 128
    excluded_station_ids = set()
    train_start_year    = None
    other_species_cols  = []
    month_weight_path   = ''
    bce_weight          = 0.0
    dist_weights        = None
    # Filled after loading pkl / get_X_Y_2D:
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
    # Windowed training params: the model sees n_windows averaged windows
    win_size            = WIN_SIZE
    n_windows           = N_WINDOWS
    # length is used by get_X_Y_2D for filling in missing years; set to 1
    # since we will call get_X_Y_2D only for the per-year dicts, not sequences
    length              = 1
    forage_species      = 'Lagodon rhomboides_A,Lagodon rhomboides_R'


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(ckpt_dir):
    """Return path to the highest-epoch model-N file in ckpt_dir."""
    files = [f for f in os.listdir(ckpt_dir) if f.startswith('model-')]
    if not files:
        raise FileNotFoundError(f'No model-* files found in {ckpt_dir}')
    latest = sorted(files, key=lambda x: int(x.split('-')[1]))[-1]
    path = os.path.join(ckpt_dir, latest)
    print(f'[shap] Using checkpoint: {path}')
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Species / feature metadata
# ─────────────────────────────────────────────────────────────────────────────

def load_species_info(spp_pkl_path, args):
    """
    Load species pkl and fill args.output_names, output_idx, other_species_cols,
    n_env_feats, individual_indices, group_slices.

    The pkl is expected to be a dict with keys:
      target_species, feature_species, n_env_features, (optionally) feature_groups
    We use forage_species = '' so all targets remain in the output.
    """
    with open(spp_pkl_path, 'rb') as f:
        spp_data = pickle.load(f)

    if isinstance(spp_data, dict):
        target_spp        = spp_data['target_species']
        community_species = spp_data.get('feature_species', [])
        n_env             = int(spp_data.get('n_env_features', 0))
    else:
        # Legacy list format — treat everything ending in _A/_R/_SA/_J as target
        target_spp        = [s for s in spp_data if s.endswith(('_A', '_R', '_SA', '_J'))]
        community_species = [s for s in spp_data if s not in set(target_spp)]
        n_env             = 0

    # Exclude forage species from targets (same as training)
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

    # Build group_slices exactly as main.py does
    n_forage    = len(forage_in_t)
    n_community = len(community_species)
    n_eco_total = n_env

    o               = n_forage
    community_start = o + n_eco_total
    community_end   = community_start + n_community
    mid_habitat     = o + 14 + 7
    # Gear-based community split (from pkl if available, else half)
    n_prey    = int(spp_data.get('n_prey_community',    n_community // 2))
    n_nonprey = int(spp_data.get('n_nonprey_community', n_community - n_prey))
    prey_end  = community_start + n_prey

    args.individual_indices = []   # all-group design: no individual tokens
    args.group_slices = [
        (o+0,           o+6,           'env'),
        (o+6,           o+14,          'restoration'),
        (o+14,          mid_habitat,   'habitat_structured'),
        (mid_habitat,   o+28,          'habitat_open'),
        (o+28,          o+35,          'shoreline'),
        (o+35,          o+38,          'water_effort'),
        (o+38,          o+52,          'bycatch'),
        (o+52,          o+58,          'na_indicators'),
        (community_start, prey_end,    'prey_community'),
        (prey_end,  community_end,     'nonprey_community'),
    ]
    # forage token at front only when forage species are present
    if n_forage > 0:
        args.group_slices.insert(0, (0, n_forage, 'forage'))

    print(f'[shap] Loaded {len(args.output_names)} target species')
    print(f'[shap] n_env_feats={n_env}, n_community={n_community}')
    print(f'[shap] group_slices ({len(args.group_slices)}): '
          + ', '.join(f'{n}[{e-s}]' for s, e, n in args.group_slices))
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(args):
    """
    Load NPZ and build X_dict / Y_dict via get_X_Y_2D.
    Returns (X_dict, Y_dict, adj, county_set, county_avg, year_avg_Y,
             min_year, max_year).
    """
    raw  = np.load(args.data_dir)
    data = raw['data']
    print(f'[shap] Raw data shape: {data.shape}')

    out = get_X_Y_2D(data, args, device)
    (X_dict, Y_dict, avail_dict, adj, order_map,
     min_year, max_year, county_set, county_avg, year_avg_Y) = out

    return X_dict, Y_dict, adj, county_set, county_avg, year_avg_Y, min_year, max_year


def compute_normalisation(X_dict, Y_dict, county_set, test_year, args):
    """
    Compute per-species mean/std on training years (all years < test_year - 1)
    and store in args.means / args.stds / args.count_means / args.count_stds.

    Y_dict values are [12, n_out] arrays (one per station-year).
    """
    all_Y = []
    for station in county_set:
        for year, y_arr in Y_dict[station].items():
            if year < test_year - 1:
                all_Y.append(y_arr)   # [12, n_out]

    if not all_Y:
        raise ValueError('No training data found for normalisation.')

    all_Y = torch.tensor(np.stack(all_Y, axis=0), dtype=torch.float32)  # [N, 12, n_out]
    all_Y_flat = all_Y.reshape(-1, all_Y.shape[-1])                     # [N*12, n_out]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        means = torch.tensor(
            np.nanmean(all_Y_flat.numpy(), axis=0, keepdims=True), dtype=torch.float32
        ).to(device)
        stds = torch.tensor(
            np.nanstd(all_Y_flat.numpy(), axis=0, keepdims=True), dtype=torch.float32
        ).to(device)

    stds[stds < 1e-6] = 1.0

    all_Y_cnt = torch.expm1(all_Y_flat.clamp(min=0))
    cnt_means = all_Y_cnt.nanmean(0, keepdim=True).to(device)
    cnt_stds  = torch.std(all_Y_cnt, dim=0, keepdim=True).to(device)
    cnt_stds[cnt_stds < 1e-6] = 1.0

    args.means       = means
    args.stds        = stds
    args.count_means = cnt_means
    args.count_stds  = cnt_stds
    n_train = sum(1 for s in county_set for y in Y_dict[s] if y < test_year - 1)
    print(f'[shap] Normalisation computed from {n_train} station-year entries')
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Windowed data builder (self-contained copy from train.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_merged_windows(X_dict, Y_dict, county_set,
                         T, win_size=3, n_windows=10):
    """
    Build n_windows sliding-window averages for prediction year T.

    For window w (1-indexed, 1 … n_windows):
        year_end   = T - (n_windows - w)
        year_start = year_end - win_size + 1

    Each window's X and Y are averaged over the win_size years that have data.

    Parameters
    ----------
    X_dict     : dict  station_id → {year: np.ndarray [12, n_features]}
    Y_dict     : dict  station_id → {year: np.ndarray [12, n_species]}
    county_set : list  ordered station ids
    T          : int   prediction year
    win_size   : int   years per window (default 3)
    n_windows  : int   total windows    (default 10)

    Returns
    -------
    window_XY : dict
        window_idx (1 … n_windows) →
            (X_merged np.ndarray [N, 12, n_feat],
             Y_merged np.ndarray [N, 12, n_sp],
             counties list[int])
    """
    _s0     = county_set[0]
    _y0     = next(iter(X_dict[_s0]))
    n_feat  = X_dict[_s0][_y0].shape[1]
    n_sp    = Y_dict[_s0][_y0].shape[1]
    N       = len(county_set)

    window_XY = {}
    for w in range(1, n_windows + 1):
        year_end     = T - (n_windows - w)
        year_start   = year_end - win_size + 1
        window_years = list(range(year_start, year_end + 1))

        X_sum  = np.zeros((N, 12, n_feat),  dtype=np.float64)
        Y_sum  = np.zeros((N, 12, n_sp),    dtype=np.float64)
        counts = np.zeros((N, 1, 1),         dtype=np.float64)

        for ci, station in enumerate(county_set):
            sx = X_dict.get(station, {})
            sy = Y_dict.get(station, {})
            n_valid = 0
            for yr in window_years:
                if yr in sx:
                    X_sum[ci]  += sx[yr]
                    Y_sum[ci]  += sy[yr]
                    n_valid    += 1
            counts[ci, 0, 0] = n_valid

        valid    = counts > 0
        X_merged = np.where(valid, X_sum / np.where(counts > 0, counts, 1), 0.0)
        Y_merged = np.where(valid, Y_sum / np.where(counts > 0, counts, 1), 0.0)

        window_XY[w] = (
            X_merged.astype(np.float32),
            Y_merged.astype(np.float32),
            list(county_set),
        )
    return window_XY


# ─────────────────────────────────────────────────────────────────────────────
# Graph / DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_and_loader(adj, batch_size=256):
    """Build DGL graph and neighbor-sampling DataLoader."""
    sp_adj = sp.coo_matrix(adj)
    g = dgl.from_scipy(sp_adj).to(device)
    g = dgl.add_self_loop(g)
    N = adj.shape[0]
    sampler    = dgl.dataloading.MultiLayerNeighborSampler([10, 10])
    nodeloader = dgl.dataloading.DataLoader(
        g, torch.arange(N).to(device), sampler,
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=0, device=device,
    )
    return g, nodeloader


def attach_dist_weights(blocks, dist_weights, edge_feat_dim=3):
    """
    Attach [dist_w, hab_sim, env_sim] edge features to DGL blocks.
    Falls back to ones if no weights are found for an edge.
    """
    if dist_weights is None:
        return
    default_feat = np.ones(edge_feat_dim, dtype=np.float32)
    for blk in blocks:
        edges_src, edges_dst = blk.edges()
        src_ids = blk.srcdata[dgl.NID].cpu().numpy()
        dst_ids = blk.dstdata[dgl.NID].cpu().numpy()
        feats = []
        for s, d in zip(edges_src.cpu().numpy(), edges_dst.cpu().numpy()):
            global_s = int(src_ids[s])
            global_d = int(dst_ids[d])
            val = dist_weights.get((global_s, global_d), default_feat)
            val = np.asarray(val, dtype=np.float32)
            if val.ndim == 0:
                val = np.array([float(val)], dtype=np.float32)
            feats.append(val)
        blk.edata['dist_w'] = torch.tensor(
            np.stack(feats, axis=0), dtype=torch.float32, device=blk.device
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path, in_dim, out_dim, args):
    """Instantiate GAT_RNN_V2 and load checkpoint weights."""
    model = GAT_RNN_V2(args, in_dim, out_dim).to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Support both raw state dict and {'model': state_dict, ...} formats
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[shap] Model loaded — {n_params:,} parameters')
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, nodeloader, windowed_XY_T, county_set,
                  args, dist_weights, x_perturb_fn=None):
    """
    Full inference pass for GAT_RNN_V2 using the windowed data representation.

    windowed_XY_T : dict  window_idx (1..n_windows) → (X_merged, Y_merged, counties)
                    as returned by build_merged_windows for a single prediction year T.
                    X_merged : np.ndarray [N_stations, 12, n_feat]
                    Y_merged : np.ndarray [N_stations, 12, n_out]

    x_perturb_fn  : optional callable(X_src_np: np.ndarray) → np.ndarray
                    Applied to each X_merged array before stacking. Used for
                    permutation importance (perturbs feature columns).

    Returns:
      pred_count [N, 12, n_out]  — expm1-transformed predictions (count scale)
      true_count [N, 12, n_out]  — expm1-transformed labels      (count scale)
    """
    model.eval()
    n_windows = len(windowed_XY_T)
    N         = len(county_set)
    n_out     = windowed_XY_T[1][1].shape[-1]   # from Y_merged of window 1
    preds     = torch.zeros(N, 12, n_out)
    trues     = torch.zeros(N, 12, n_out)

    # Build full [N, n_windows, 12, n_feat/n_out] arrays once
    X_wins, Y_wins = [], []
    for w in range(1, n_windows + 1):
        X_w, Y_w, _ = windowed_XY_T[w]
        if x_perturb_fn is not None:
            X_w = x_perturb_fn(X_w)    # perturb feature group for importance
        X_wins.append(X_w)
        Y_wins.append(Y_w)

    X_all_np = np.stack(X_wins, axis=1)  # [N, n_windows, 12, n_feat]
    Y_all_np = np.stack(Y_wins, axis=1)  # [N, n_windows, 12, n_out]

    with torch.no_grad():
        for (in_nodes, out_nodes, blocks) in nodeloader:
            in_idx  = in_nodes.cpu().numpy()
            out_idx = out_nodes.cpu().numpy()

            X_src = torch.tensor(X_all_np[in_idx],  dtype=torch.float32, device=device)
            Y_dst = torch.tensor(Y_all_np[out_idx],  dtype=torch.float32, device=device)

            # Normalise look-back labels; Y_std[:, :-1] = [n_dst, n_windows-1, 12, n_out]
            Y_std  = (Y_dst - args.means) / args.stds
            blocks = [b.int().to(device) for b in blocks]
            attach_dist_weights(blocks, dist_weights, args.edge_feat_dim)

            # model.forward: x [n_src, n_windows, 12, n_feat]
            #                y [n_dst, n_windows-1, 12, n_out]  (look-back)
            pred_std = model(blocks, X_src, Y_std[:, :-1])
            pred     = pred_std * args.stds + args.means   # [n_dst, n_windows, 12, n_out]

            p_last = torch.clamp(torch.expm1(pred[:, -1]), min=0.0)  # [batch, 12, n_out]
            t_last = torch.clamp(torch.expm1(Y_dst[:, -1]), min=0.0)

            preds[out_nodes.cpu()] = p_last.cpu()
            trues[out_nodes.cpu()] = t_last.cpu()

    return preds.numpy(), trues.numpy()


def mse_per_species(pred, true):
    """
    Compute per-species MSE averaged over stations and months.

    pred, true : [N, 12, n_out]
    Returns    : [n_out] MSE array.
    """
    n_out = pred.shape[-1]
    mse   = np.zeros(n_out)
    for j in range(n_out):
        p = pred[..., j].ravel()
        t = true[..., j].ravel()
        valid = ~(np.isnan(p) | np.isnan(t))
        if valid.sum() > 0:
            mse[j] = np.mean((p[valid] - t[valid]) ** 2)
    return mse


# ─────────────────────────────────────────────────────────────────────────────
# Feature-group construction
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_groups_from_args(args, in_dim):
    """Build SHAP feature groups from args.group_slices.

    args.group_slices is a list of (start, end, name) built in load_species_info()
    from the species pkl. The split distinguishes:
        prey_community     — species in the predator diet
        nonprey_community  — co-occurring species not in diet
        forage             — Pinfish A+R (prepended to X at [0:n_forage])

    Forage species ARE prey; we merge `forage` into `prey_community` so the
    importance attributed to that biological category is not artificially split.

    Returns: list of (name, np.ndarray of column indices), each clamped to in_dim.
    """
    # Map name → list of column indices
    by_name = {}
    for s, e, n in args.group_slices:
        if s >= in_dim: continue
        cols = list(range(s, min(e, in_dim)))
        by_name.setdefault(n, []).extend(cols)

    # Merge forage into prey_community
    if 'forage' in by_name and 'prey_community' in by_name:
        by_name['prey_community'] = sorted(set(by_name['prey_community'])
                                            | set(by_name['forage']))
        del by_name['forage']

    # Preserve a sensible display order
    order = ['env', 'restoration', 'habitat_structured', 'habitat_open',
             'shoreline', 'water_effort', 'bycatch', 'na_indicators',
             'prey_community', 'nonprey_community']
    out = []
    for n in order:
        if n in by_name:
            out.append((n, np.asarray(by_name[n], dtype=int)))
    # Append any remaining unanticipated groups in original order
    seen = {n for n, _ in out}
    for s, e, n in args.group_slices:
        if n not in seen and n in by_name:
            out.append((n, np.asarray(by_name[n], dtype=int)))
            seen.add(n)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Permutation importance
# ─────────────────────────────────────────────────────────────────────────────

def permutation_importance(model, nodeloader, windowed_XY_T, county_set,
                           args, dist_weights, feature_groups, n_repeats=5):
    """
    Permutation-based group importance.

    For each feature group g:
      1. Shuffle group g's feature columns across stations (for every window)
      2. Re-run full inference → compute MSE per species
      3. importance[g, s] = (mse_perm[s] - mse_baseline[s]) / (mse_baseline[s] + 1e-10)
    Repeat n_repeats times and average.

    Returns:
      importance [n_groups, n_species]  — relative MSE increase (higher = more important)
      baseline_mse [n_species]
    """
    N = len(county_set)

    print('[shap] Running baseline inference …')
    baseline_pred, true_vals = run_inference(
        model, nodeloader, windowed_XY_T, county_set, args, dist_weights
    )
    baseline_mse = mse_per_species(baseline_pred, true_vals)
    print(f'[shap] Baseline mean MSE across species: {baseline_mse.mean():.4f}')

    n_groups  = len(feature_groups)
    n_species = baseline_mse.shape[0]
    imp_sum     = np.zeros((n_groups, n_species))   # relative MSE increase
    delta_sum   = np.zeros((n_groups, n_species))   # absolute MSE increase (count-scale)

    for gi, (name, cols) in enumerate(feature_groups):
        cols = np.asarray(cols, dtype=int)
        group_imp   = np.zeros((n_repeats, n_species))
        group_delta = np.zeros((n_repeats, n_species))

        for rep in range(n_repeats):
            # Permutation index: shuffle stations randomly (same permutation applied
            # to all windows and months so the group signal is destroyed across stations)
            perm_idx = np.random.permutation(N)

            def perturb_fn(X_merged_np, _cols=cols, _perm=perm_idx):
                """Replace columns _cols with values from a permuted station."""
                X_out = X_merged_np.copy()
                X_out[:, :, _cols] = X_merged_np[_perm][:, :, _cols]
                return X_out

            perm_pred, _ = run_inference(
                model, nodeloader, windowed_XY_T, county_set,
                args, dist_weights, x_perturb_fn=perturb_fn
            )
            perm_mse = mse_per_species(perm_pred, true_vals)
            # Both absolute (count-scale) and relative increases
            group_delta[rep] = perm_mse - baseline_mse
            group_imp[rep]   = (perm_mse - baseline_mse) / (baseline_mse + 1e-10)

        imp_sum[gi]   = group_imp.mean(axis=0)
        delta_sum[gi] = group_delta.mean(axis=0)
        mean_imp      = float(np.nanmean(imp_sum[gi]))
        mean_drmse    = float(np.nanmean(np.sqrt(np.maximum(delta_sum[gi], 0))))
        print(f'  [perm] {name:24s}: rel ΔMSE={mean_imp:+.4f}  '
              f'abs ΔRMSE≈{mean_drmse:.3f} fish/obs')

    return imp_sum, baseline_mse, delta_sum


# ─────────────────────────────────────────────────────────────────────────────
# Population-contribution importance
# ─────────────────────────────────────────────────────────────────────────────

def population_contribution(model, nodeloader, windowed_XY_T, county_set,
                             args, dist_weights, feature_groups, n_repeats=5):
    """
    Measures how much each feature group influences the PREDICTED POPULATION.

    For each group g and repeat r:
      1. Permute group g's columns across stations
      2. Re-run inference → permuted predicted population total per species
      3. delta_pop[g, s, r] = baseline_pop[s] - permuted_pop[g, s, r]
         (positive = group inflates predictions; sign shows direction)

    Returns
    -------
    baseline_pop   : [n_species]                  total predicted population at baseline
    delta_pop_mean : [n_groups, n_species]         mean Δpopulation (signed, fish)
    delta_pop_abs  : [n_groups, n_species]         mean |Δpopulation| (unsigned, fish)
    """
    N = len(county_set)

    print('[shap-pop] Running baseline inference …')
    baseline_pred, _ = run_inference(
        model, nodeloader, windowed_XY_T, county_set, args, dist_weights
    )
    # Sum over stations and months → total predicted population per species
    baseline_pop = baseline_pred.sum(axis=(0, 1))   # [n_species]
    print(f'[shap-pop] Baseline total population (mean across species): '
          f'{baseline_pop.mean():.2f} fish')

    n_groups  = len(feature_groups)
    n_species = len(baseline_pop)
    delta_sum_signed = np.zeros((n_groups, n_species))
    delta_sum_abs    = np.zeros((n_groups, n_species))

    for gi, (name, cols) in enumerate(feature_groups):
        cols = np.asarray(cols, dtype=int)
        rep_signed = np.zeros((n_repeats, n_species))
        rep_abs    = np.zeros((n_repeats, n_species))

        for rep in range(n_repeats):
            perm_idx = np.random.permutation(N)

            def perturb_fn(X_np, _cols=cols, _perm=perm_idx):
                X_out = X_np.copy()
                X_out[:, :, _cols] = X_np[_perm][:, :, _cols]
                return X_out

            perm_pred, _ = run_inference(
                model, nodeloader, windowed_XY_T, county_set,
                args, dist_weights, x_perturb_fn=perturb_fn
            )
            perm_pop = perm_pred.sum(axis=(0, 1))   # [n_species]
            delta = baseline_pop - perm_pop          # positive = group raises predictions
            rep_signed[rep] = delta
            rep_abs[rep]    = np.abs(delta)

        delta_sum_signed[gi] = rep_signed.mean(axis=0)
        delta_sum_abs[gi]    = rep_abs.mean(axis=0)
        mean_abs = float(delta_sum_abs[gi].mean())
        print(f'  [pop] {name:24s}: mean |ΔPop| = {mean_abs:.2f} fish  '
              f'(signed mean = {float(delta_sum_signed[gi].mean()):+.2f})')

    return baseline_pop, delta_sum_signed, delta_sum_abs


def plot_population_bar(delta_pop_abs, group_names, out_path):
    """
    Horizontal bar chart: mean |ΔPredicted Population| across species per group.
    Sorted by importance (largest first).
    """
    mean_abs  = delta_pop_abs.mean(axis=1)
    order     = np.argsort(mean_abs)
    fig, ax   = plt.subplots(figsize=(8, max(4, len(group_names) * 0.55)))

    colors = plt.cm.YlOrRd(
        (mean_abs[order] - mean_abs[order].min()) /
        max(mean_abs[order].max() - mean_abs[order].min(), 1e-9) * 0.75 + 0.1
    )
    bars = ax.barh(range(len(group_names)),
                   mean_abs[order],
                   color=colors, edgecolor='grey', linewidth=0.5)
    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels([group_names[i] for i in order], fontsize=11)
    ax.set_xlabel('Mean |ΔPredicted Population|  (fish, summed over stations & months)',
                  fontsize=10)
    ax.set_title(
        'Feature Group Importance — Effect on Predicted Population\n'
        '(Permuting group → how much does total predicted catch change?)',
        fontsize=11, fontweight='bold'
    )
    for bar, val in zip(bars, mean_abs[order]):
        ax.text(val + mean_abs.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}', va='center', fontsize=9)
    ax.set_xlim(0, mean_abs.max() * 1.18)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[shap] Population importance bar chart → {out_path}')


def plot_population_heatmap(delta_pop_abs, group_names, species_names, out_path):
    """
    Heatmap: rows = groups, columns = species. Color = |ΔPopulation| in fish.
    """
    n_groups, n_species = delta_pop_abs.shape
    fig_w = max(10, n_species * 1.0)
    fig_h = max(4,  n_groups  * 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    mat  = delta_pop_abs.copy()
    vmax = float(np.nanpercentile(mat, 95))
    im   = ax.imshow(mat, cmap='YlOrRd', aspect='auto', vmin=0, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('|ΔPredicted Population| (fish)', fontsize=10)

    ax.set_xticks(range(n_species))
    ax.set_xticklabels(species_names, rotation=40, ha='right', fontsize=9)
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(group_names, fontsize=10)
    ax.set_title(
        'Feature Group Effect on Predicted Population per Species\n'
        '(Mean |Δtotal catch| when group is permuted)',
        fontsize=11, fontweight='bold', pad=12
    )
    for i in range(n_groups):
        for j in range(n_species):
            v = mat[i, j]
            tc = 'white' if v > 0.6 * vmax else 'black'
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=7, color=tc)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[shap] Population heatmap → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(importance, group_names, species_common_names, out_path):
    """
    Heatmap: rows = feature groups, columns = species.
    Color encodes relative MSE increase (red = high importance, white = low).
    """
    n_groups, n_species = importance.shape
    fig_w = max(10, n_species * 1.0)
    fig_h = max(4,  n_groups  * 0.6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    mat   = np.nan_to_num(importance)
    vmax  = float(np.nanpercentile(mat, 95))
    vmin  = 0.0
    im    = ax.imshow(mat, cmap='RdYlBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    cbar  = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Relative MSE increase', fontsize=10)

    ax.set_xticks(range(n_species))
    ax.set_xticklabels(species_common_names, rotation=40, ha='right', fontsize=9)
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(group_names, fontsize=10)
    ax.set_title(
        'Feature group importance per species\n'
        '(Permutation: relative MSE increase when group is shuffled)',
        fontsize=11, fontweight='bold', pad=12
    )

    # Annotate cells
    for i in range(n_groups):
        for j in range(n_species):
            v = mat[i, j]
            text_color = 'white' if v > 0.6 * vmax else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=7, color=text_color)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[shap] Heatmap saved → {out_path}')


def plot_bar_chart(importance, group_names, out_path):
    """
    Horizontal bar chart of mean importance across species, sorted by importance.
    """
    mean_imp  = importance.mean(axis=1)      # [n_groups]
    order     = np.argsort(mean_imp)         # ascending
    sorted_names = [group_names[i] for i in order]
    sorted_vals  = mean_imp[order]

    fig, ax = plt.subplots(figsize=(8, max(4, len(group_names) * 0.5)))

    colors = plt.cm.RdYlBu_r(
        (sorted_vals - sorted_vals.min()) /
        max(sorted_vals.max() - sorted_vals.min(), 1e-9)
        * 0.75 + 0.1
    )
    bars = ax.barh(range(len(group_names)), sorted_vals, color=colors)

    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.set_xlabel('Mean relative MSE increase (across species)', fontsize=11)
    ax.set_title(
        'Feature group importance — average across 12 species\n'
        '(Higher = model relies more on this group)',
        fontsize=12, fontweight='bold'
    )
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')

    for bar, val in zip(bars, sorted_vals):
        ax.text(val + max(sorted_vals) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.set_xlim(min(0, sorted_vals.min()) - max(sorted_vals) * 0.05,
                sorted_vals.max() * 1.2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[shap] Bar chart saved → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('=' * 64)
    print('GNN-RNN FIM — Permutation Group Importance Analysis')
    print(f'Test year : {TEST_YEAR}')
    print(f'Repeats   : {N_PERM}')
    print('=' * 64)

    # ── Build args ────────────────────────────────────────────────────────────
    args = FIMArgs()

    # ── Load species metadata ─────────────────────────────────────────────────
    print('\n[shap] Loading species metadata …')
    args = load_species_info(SPP_PKL, args)

    # ── Load data ─────────────────────────────────────────────────────────────
    print('\n[shap] Loading data …')
    (X_dict, Y_dict, adj, county_set, county_avg,
     year_avg_Y, min_year, max_year) = load_data(args)

    print(f'[shap] Stations: {len(county_set)}, years: {min_year}–{max_year}')

    # ── Normalisation from raw Y_dict (training years) ────────────────────────
    args = compute_normalisation(X_dict, Y_dict, county_set, TEST_YEAR, args)

    # ── Determine in_dim / out_dim from raw dicts ─────────────────────────────
    _s0  = county_set[0]
    _y0  = next(iter(X_dict[_s0]))
    in_dim  = X_dict[_s0][_y0].shape[-1]   # n_features (last dim of [12, n_feat])
    out_dim = len(args.output_names)
    print(f'[shap] in_dim={in_dim}, out_dim={out_dim}')

    # Build feature groups from args.group_slices (correct prey/nonprey split,
    # forage merged into prey_community).
    feature_groups = build_feature_groups_from_args(args, in_dim)
    group_names = [n for n, _ in feature_groups]
    print(f'[shap] Feature groups ({len(feature_groups)}): '
          + ', '.join(f'{n}[{len(c)}]' for n, c in feature_groups))

    # ── Build graph & DataLoader ──────────────────────────────────────────────
    print('\n[shap] Building graph …')
    g, nodeloader = build_graph_and_loader(adj, batch_size=128)
    print(f'[shap] Graph: {g.num_nodes()} nodes, {g.num_edges()} edges')

    # ── Load distance weights ─────────────────────────────────────────────────
    dist_weights = None
    if os.path.exists(DIST_PKL):
        with open(DIST_PKL, 'rb') as f:
            dist_weights = pickle.load(f)
        # Detect edge feature dimension
        _sample = next(iter(dist_weights.values()))
        _arr    = np.asarray(_sample)
        args.edge_feat_dim = int(_arr.shape[0]) if _arr.ndim >= 1 else 1
        print(f'[shap] Loaded {len(dist_weights):,} edge weights '
              f'(edge_feat_dim={args.edge_feat_dim})')
    else:
        args.edge_feat_dim = 3
        print(f'[shap] No dist_weights file at {DIST_PKL} — using uniform edge features')

    # ── Load model ────────────────────────────────────────────────────────────
    print('\n[shap] Loading model …')
    ckpt_path = find_latest_checkpoint(CKPT_DIR)
    model     = load_model(ckpt_path, in_dim, out_dim, args)

    # ── Build windowed data for test year ────────────────────────────────────
    print(f'\n[shap] Building windowed data for test year {TEST_YEAR} …')
    windowed_XY_T = build_merged_windows(
        X_dict, Y_dict, county_set,
        TEST_YEAR, win_size=WIN_SIZE, n_windows=N_WINDOWS
    )
    # windowed_XY_T: {1..10} → (X_merged [N,12,n_feat], Y_merged [N,12,n_out], counties)
    sample_X = windowed_XY_T[1][0]   # [N, 12, n_feat]
    print(f'[shap] Windowed data: {N_WINDOWS} windows × {sample_X.shape[0]} stations '
          f'× 12 months × {sample_X.shape[-1]} features')

    # ── Permutation importance ────────────────────────────────────────────────
    print(f'\n[shap] === Permutation importance (n_repeats={N_PERM}) ===')
    importance, baseline_mse, delta_mse_count = permutation_importance(
        model, nodeloader, windowed_XY_T, county_set,
        args, dist_weights, feature_groups, n_repeats=N_PERM
    )
    # importance:        [n_groups, n_species]   relative MSE increase (unitless)
    # delta_mse_count:   [n_groups, n_species]   absolute MSE increase in COUNT scale
    baseline_rmse_count = np.sqrt(np.maximum(baseline_mse, 0))   # per species, fish/obs
    delta_rmse_count    = np.sqrt(np.maximum(delta_mse_count, 0))

    # ── Species display names ─────────────────────────────────────────────────
    species_display = [
        COMMON_NAMES.get(s, s) for s in args.output_names
    ]

    # ── Save CSVs (relative + count-scale absolute) ───────────────────────────
    csv_path = os.path.join(FIGURES_DIR, 'shap_group_importance.csv')
    pd.DataFrame(importance, index=group_names, columns=species_display) \
        .rename_axis('feature_group').to_csv(csv_path)
    print(f'[shap] Relative ΔMSE  → {csv_path}')

    csv_dmse_path = os.path.join(FIGURES_DIR, 'shap_group_dMSE_count.csv')
    pd.DataFrame(delta_mse_count, index=group_names, columns=species_display) \
        .rename_axis('feature_group').to_csv(csv_dmse_path)
    print(f'[shap] Absolute ΔMSE (count²)  → {csv_dmse_path}')

    csv_drmse_path = os.path.join(FIGURES_DIR, 'shap_group_dRMSE_count.csv')
    pd.DataFrame(delta_rmse_count, index=group_names, columns=species_display) \
        .rename_axis('feature_group').to_csv(csv_drmse_path)
    print(f'[shap] Absolute ΔRMSE (fish/obs)  → {csv_drmse_path}')

    # baseline RMSE per species, count scale, for context
    base_path = os.path.join(FIGURES_DIR, 'shap_baseline_rmse_count.csv')
    pd.DataFrame({'species': species_display,
                   'baseline_MSE_count': baseline_mse,
                   'baseline_RMSE_count': baseline_rmse_count}) \
        .to_csv(base_path, index=False)
    print(f'[shap] Baseline RMSE per species (count) → {base_path}')

    # ── Plot 1: heatmap ───────────────────────────────────────────────────────
    heatmap_path = os.path.join(FIGURES_DIR, 'shap_group_heatmap.png')
    plot_heatmap(importance, group_names, species_display, heatmap_path)

    # ── Plot 2: mean importance bar chart ─────────────────────────────────────
    bar_path = os.path.join(FIGURES_DIR, 'shap_group_importance.png')
    plot_bar_chart(importance, group_names, bar_path)

    # ── Population contribution ───────────────────────────────────────────────
    print(f'\n[shap] === Population contribution (n_repeats={N_PERM}) ===')
    baseline_pop, delta_pop_signed, delta_pop_abs = population_contribution(
        model, nodeloader, windowed_XY_T, county_set,
        args, dist_weights, feature_groups, n_repeats=N_PERM
    )

    # Save CSVs
    pop_abs_path = os.path.join(FIGURES_DIR, 'shap_pop_importance_abs.csv')
    pd.DataFrame(delta_pop_abs, index=group_names, columns=species_display) \
        .rename_axis('feature_group').to_csv(pop_abs_path)
    pop_signed_path = os.path.join(FIGURES_DIR, 'shap_pop_importance_signed.csv')
    pd.DataFrame(delta_pop_signed, index=group_names, columns=species_display) \
        .rename_axis('feature_group').to_csv(pop_signed_path)
    pop_base_path = os.path.join(FIGURES_DIR, 'shap_pop_baseline.csv')
    pd.DataFrame({'species': species_display, 'baseline_total_pop': baseline_pop}) \
        .to_csv(pop_base_path, index=False)
    print(f'[shap] Population importance CSVs → {pop_abs_path}')

    # Plots
    pop_bar_path  = os.path.join(FIGURES_DIR, 'shap_pop_importance_bar.png')
    pop_heat_path = os.path.join(FIGURES_DIR, 'shap_pop_importance_heatmap.png')
    plot_population_bar(delta_pop_abs, group_names, pop_bar_path)
    plot_population_heatmap(delta_pop_abs, group_names, species_display, pop_heat_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    mean_imp = importance.mean(axis=1)
    ranked   = sorted(zip(group_names, mean_imp), key=lambda x: x[1], reverse=True)

    print('\n' + '=' * 64)
    print('PERMUTATION IMPORTANCE SUMMARY')
    print('=' * 64)
    print(f'  Test year       : {TEST_YEAR}')
    print(f'  Checkpoint      : {os.path.basename(ckpt_path)}')
    print(f'  in_dim={in_dim}, out_dim={out_dim}')
    print(f'  Baseline mean MSE  (count scale, mean across species): {baseline_mse.mean():.4f}')
    print(f'  Baseline mean RMSE (count scale, mean across species): {baseline_rmse_count.mean():.3f} fish/obs')
    print(f'\n  Per-species baseline RMSE (count scale, fish/observation):')
    for sp, r in zip(species_display, baseline_rmse_count):
        print(f'    {sp:<32s}  {r:6.3f}')
    print(f'\n  Feature group importance — RELATIVE ΔMSE (mean across species), ranked:')
    for name, val in ranked:
        bar = '█' * max(1, int(val * 30 / (max(v for _, v in ranked) + 1e-9)))
        print(f'    {name:<24s}  {val:+.4f}  {bar}')
    print(f'\n  Feature group importance — ABSOLUTE ΔRMSE (count scale, mean fish/obs), ranked:')
    drmse_mean = delta_rmse_count.mean(axis=1)
    drmse_ranked = sorted(zip(group_names, drmse_mean), key=lambda x: x[1], reverse=True)
    for name, val in drmse_ranked:
        bar = '█' * max(1, int(val * 30 / (max(v for _, v in drmse_ranked) + 1e-9)))
        print(f'    {name:<24s}  {val:+.4f} fish  {bar}')
    print(f'\n  Feature group importance — PREDICTED POPULATION EFFECT '
          f'(mean |ΔPop| in fish), ranked:')
    pop_mean   = delta_pop_abs.mean(axis=1)
    pop_ranked = sorted(zip(group_names, pop_mean), key=lambda x: x[1], reverse=True)
    for gname, val in pop_ranked:
        bbar = '█' * max(1, int(val * 30 / (max(v for _, v in pop_ranked) + 1e-9)))
        print(f'    {gname:<24s}  {val:6.1f} fish  {bbar}')
    print(f'\n  Outputs:')
    print(f'    {heatmap_path}')
    print(f'    {bar_path}')
    print(f'    {pop_bar_path}')
    print(f'    {pop_heat_path}')
    print(f'    {csv_path}')
    print('=' * 64)


if __name__ == '__main__':
    main()
