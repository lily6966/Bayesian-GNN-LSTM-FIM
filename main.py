import argparse
import os
import pickle
from train import train
test = None  # test.py not used in v2 training mode

# Index of the yield variable for each variable
OUTPUT_INDICES = {'corn': 2,
                  'upland_cotton': 3,
                  'sorghum': 4,
                  'soybeans': 5,
                  'spring_wheat': 6,
                  'winter_wheat': 7}

# Indices of the progress variables for each crop type in the X array.
PROGRESS_INDICES_DAILY = {'corn': list(range(8403-8, 13148-8)),
                          'upland_cotton': list(range(13148-8, 17893-8)),
                          'sorghum': list(range(17893-8, 22638-8)),
                          'soybeans': list(range(22638-8, 28113-8)),
                          'spring_wheat': list(range(32858-8, 37603-8)),
                          'winter_wheat': list(range(37603-8, 43443-8))}
PROGRESS_INDICES_WEEKLY = {'corn': list(range(1204-8, 1880-8)),
                          'upland_cotton': list(range(1880-8, 2556-8)),
                          'sorghum': list(range(2556-8, 3232-8)),
                          'soybeans': list(range(3232-8, 4012-8)),
                          'spring_wheat': list(range(4688-8, 5364-8)),
                          'winter_wheat': list(range(5364-8, 6196-8))}

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', "--dataset", default='soybean', type=str, help='dataset name')
parser.add_argument('-adj', "--us_adj_file", default='', type=str, help='adjacency file')
parser.add_argument('-fid_map', "--crop_id_to_fid", default='', type=str, help='crop id to fid file')
parser.add_argument('-cp', "--checkpoint_path", default='./ckpt', type=str, help='The path to a checkpoint from which to fine-tune.')

parser.add_argument('-dd', "--data_dir", default='data/soybean_data.npz', type=str, help='The data directory')

parser.add_argument('-seed', "--seed", default=0, type=int, help='seed')
parser.add_argument('-bs', "--batch_size", default=128, type=int, help='the number of data points in one minibatch')
parser.add_argument('-lr', "--learning_rate", default=1e-3, type=float, help='initial learning rate')
parser.add_argument('-epoch', "--max_epoch", default=30, type=int, help='max epoch to train')
parser.add_argument('-wd', "--weight_decay", default=1e-5, type=float, help='weight decay rate')
parser.add_argument('-lrdr', "--lr_decay_ratio", default=0.5, type=float, help='The decay ratio of learning rate')

parser.add_argument('-se', "--save_epoch", default=1, type=int, help='epochs to save the checkpoint of the model')
parser.add_argument('-max_keep', "--max_keep", default=3, type=int, help='maximum number of saved model')
parser.add_argument('-check_freq', "--check_freq", default=50, type=int, help='checking frequency')

parser.add_argument('-eta_min', "--eta_min", default=1e-5, type=float, help='minimum lr')
parser.add_argument('-gamma', "--gamma", default=0.5, type=float, help='StepLR decay')
parser.add_argument('-T0', "--T0", default=50, type=int, help='optimizer T0')
parser.add_argument('-sleep', "--sleep", default=50, type=int, help='sleep time')
parser.add_argument('-lrsteps', "--lrsteps", default=50, type=int, help='StepLR steps')
parser.add_argument('-T_mult', "--T_mult", default=2, type=int, help='optimizer T_multi')
parser.add_argument('-patience', "--patience", default=1, type=int, help='optimizer patience')
parser.add_argument('--early_stop_patience', type=int, default=0,
                    help='Early-stopping patience (epochs without val_rmse improvement). '
                         '0 disables early stopping. Recommended: 8 for rolling-windowed CV.')
parser.add_argument('--early_stop_min_delta', type=float, default=1e-4,
                    help='Minimum improvement in val_rmse to count as progress. Default 1e-4.')
parser.add_argument('-test_year', "--test_year", default=2024, type=int, help='test year')
parser.add_argument('-length', "--length", default=5, type=int, help='sequence window length')
parser.add_argument('-z_dim', "--z_dim", default=64, type=int, help='hidden units in RNN')

parser.add_argument('-keep_prob', "--keep_prob", default=1.0, type=float, help='1.-drop out rate')
parser.add_argument('-c1', "--c1", default=1.0, type=float, help='weight on auxiliary loss (years t-2..t-1 in window); 0=disabled')
parser.add_argument('-c2', "--c2", default=1.0, type=float, help='weight on primary loss (final year t in window)')
parser.add_argument('-mode', "--mode", type=str, help='training/test mode')
parser.add_argument('-sche', "--scheduler", default='cosine', choices=['cosine', 'step', 'plateau', 'exp', 'const'], help='lr scheduler')
parser.add_argument('-exp_gamma', "--exp_gamma", default=0.98, type=float, help='exp lr decay gamma')

parser.add_argument('-clip_grad', "--clip_grad", default=10.0, type=float, help='clip_grad')
parser.add_argument('-exclude_bays', "--exclude_bays", default='', type=str,
                    help='Comma-separated bay codes to exclude from training/val/test, e.g. "JX,TB,IR"')
parser.add_argument('--forage_species', default='', type=str,
                    help='Comma-separated species to remove from target list and add as forage input features, '
                         'e.g. "Lagodon rhomboides_A,Lagodon rhomboides_R"')
parser.add_argument('--all_species_targets', action='store_true', default=False,
                    help='Merge the community/feature_species into the target list so '
                         'the model predicts every fish species in the pkl (target + '
                         'community). Community species are removed from X input '
                         'features; their group slices (prey_community, nonprey_community) '
                         'are dropped from the hybrid encoder layout.')
parser.add_argument('--nonprey_targets', action='store_true', default=False,
                    help='Promote ONLY the non-prey community species (Gear-160 '
                         'drop/cast-net species) into the target list. Prey community '
                         'species (seine-only, Gear-20/23) stay as X input features. '
                         'Forage species (if specified via --forage_species) also stay '
                         'as input features. Mutually exclusive with --all_species_targets.')
parser.add_argument('--prevalence_target_min', type=float, default=0.0,
                    help='Prevalence floor (0-1) for auto-promoting community species to '
                         'targets. E.g. 0.05 promotes any community species present in '
                         '>=5%% of training-year hauls. The original target_species are '
                         'always kept (minus --forage_species). Species used as X features '
                         'are the complement. Use 0.0 to disable auto-promotion. '
                         'Mutually exclusive with --all_species_targets and --nonprey_targets.')
parser.add_argument('--nested', action='store_true', default=True,
                    help='Use hierarchical (nested) 2D LSTM: inner LSTM for seasonal dynamics, outer LSTM for interannual trends. Always True for GAT_RNN_V2.')
parser.add_argument('--no_log_transform', default=False, action='store_true',
                    help='Train on raw counts (expm1 of stored log1p values) instead of log1p scale')
parser.add_argument('--transform', choices=['log1p', 'root4'], default='log1p',
                    help='Per-species abundance transform applied to Y at load time. '
                         '"log1p" (default) uses NPZ values as-is. "root4" loads NPZ '
                         'log1p values, back-transforms to counts via expm1, then '
                         'applies y^(1/4). Loss / model operate in the chosen scale; '
                         'predictions are inverse-transformed back to counts at output.')

# Distance-aware GAT model — always enabled for GAT_RNN_V2
parser.add_argument('--use_gat', action='store_true', default=True,
                    help='Use GAT_RNN_V2 (EGATConv with distance edge features). Always True.')
parser.add_argument('--dist_weights_path', type=str, default='',
                    help='Path to FIM_restoration_v2_dist_weights.pkl '
                         '{(src_node, dst_node): np.array([dist_w, hab_sim, env_sim])}')

# Edge feature dimensionality (v2 adds env_sim as 3rd edge feature)
parser.add_argument('--edge_feat_dim', type=int, default=3,
                    help='Edge feature dimensionality for EGATConv. '
                         'v2 default is 3: [dist_w, hab_sim, env_sim]. '
                         'Use 2 for v1 compatibility (no env_sim).')

# Species-specific output heads + post-hoc val calibration
parser.add_argument('--species_heads', action='store_true', default=False,
                    help='Give each species its own final Linear head (instead of '
                         'one shared output layer). Also activates post-hoc OLS '
                         'calibration on the val set before reporting test metrics.')

# Encoder type — v2 default is hybrid
parser.add_argument('-encoder_type', "--encoder_type", default="hybrid",
                    choices=["cnn", "lstm", "gru", "mlp", "transformer", "hybrid"],
                    help='Encoder architecture. "hybrid" uses HybridTransformerEncoder '
                         '(42 individual tokens + 6 group tokens + 14-query attention pool). '
                         'Default: hybrid.')

# FiLM: Feature-wise Linear Modulation — enabled by default in v2
parser.add_argument('--use_film', default=True, action='store_true',
                    help='Add FiLM (Feature-wise Linear Modulation) conditioning. '
                         'Each species gets a learned embedding that generates a '
                         'species-specific (gamma, beta) pair to modulate the shared '
                         'LSTM hidden state before prediction. Default: True in v2.')

# Bayesian variational heads — enabled by default in v2
parser.add_argument('--use_bayes', default=True, action='store_true',
                    help='Replace deterministic FiLM output weights with variational '
                         'Gaussian weights (Bayes by Backprop). Rare species receive '
                         'higher KL weight → stronger shrinkage toward the prior. '
                         'Default: True in v2.')

# Species-month importance weighting
parser.add_argument('--month_weight_path', type=str, default='',
                    help='Path to species_month_weights.npy [12, n_species]. '
                         'Peak recruitment months get higher loss weight. '
                         'Default: empty (uniform weights).')

# MAML / Reptile transfer learning
parser.add_argument('--trunk_ckpt', type=str, default='',
                    help='Path to Reptile trunk checkpoint (fim_reptile_trunk.pt). '
                         'When set, trunk weights (encoder+GAT+LSTM+shared_proj) are '
                         'initialised from this checkpoint before fine-tuning.')
parser.add_argument('--freeze_trunk', action='store_true', default=False,
                    help='Freeze trunk (encoder+GAT+LSTM) during fine-tuning so only '
                         'FiLM species embeddings and heads are updated. '
                         'Requires --trunk_ckpt.')

# Bayesian transfer learning for rare species
parser.add_argument('--beta_kl', type=float, default=1e-3,
                    help='KL divergence weight in ELBO loss (default: 1e-3). '
                         'Only active when --use_bayes is set.')
parser.add_argument('--bayes_prior_std', type=float, default=1.0,
                    help='Std of the isotropic Gaussian prior on species head weights '
                         '(default: 1.0). Smaller values → stronger regularisation.')
parser.add_argument('--transfer_prior_species', type=str, default='',
                    help='Comma-separated substrings of species names to use as Bayesian '
                         'transfer donors. E.g. "rhomboides_A" for Pinfish Adult, or '
                         '"Menidia,Eucinostomus,Mugil" for multiple donors (their posterior '
                         'means are averaged). Use "__prevalent_donors__" to auto-load donors '
                         'from the species pkl (requires "prevalent_donor_species" key). '
                         'Requires --use_bayes --use_film.')
parser.add_argument('--transfer_prior_std', type=float, default=0.3,
                    help='Std of the transfer prior (default: 0.3). Smaller = stronger '
                         'pull toward the donor species. Suggested range: 0.1–0.5.')
parser.add_argument('--transfer_rare_threshold', type=float, default=0.15,
                    help='Species with nonzero_fraction < this get the transfer prior '
                         '(default: 0.15, i.e. present in <15%% of hauls).')
parser.add_argument('--transfer_warmstart_alpha', type=float, default=0.5,
                    help='Fraction of donor weights to blend into rare species init '
                         '(0=no warm-start, 1=full copy; default: 0.5).')
parser.add_argument('--species_pkl', type=str, default='',
                    help='Path to species names pkl for prey-guild donor mapping. '
                         'Auto-derived from --dd if not set.')

# Cross-donor transfer — within 14-species model
parser.add_argument('--use_cross_donors', action='store_true', default=False,
                    help='Within the 14-species FIM model, use ecologically matched '
                         'well-fitted species as cross-donors for rare ones. '
                         'E.g. Seatrout(R) → prior for Snook, Gray Snapper → prior for '
                         'Gag Grouper, Sheepshead → prior for Gray Snapper. '
                         'Donor priors refresh each epoch to track improving posteriors. '
                         'Requires --use_bayes --use_film.')

parser.add_argument('--use_prey_guild_donors', action='store_true', default=False,
                    help='When set with --use_bayes --use_film and a prey-guild NPZ, '
                         'auto-builds a species-specific Bayesian transfer prior map that '
                         'assigns ecologically matched seine-caught prey species as donors for '
                         'each management target species. Requires the matching species pkl to '
                         'contain "prey_guild_species" and "management_target_species" keys. '
                         'Uses set_transfer_priors_mapped() on GAT_RNN_V2.')

# Hurdle (zero-inflated two-stage) head
parser.add_argument('--use_hurdle', action='store_true', default=False,
                    help='Add a parallel presence head per species (Bernoulli) and '
                         'train the abundance head only on observed presences. '
                         'Loss = w*BCE(presence, y>0) + (1-w)*logcosh(abundance, y) | y>0. '
                         'Final prediction at inference = sigmoid(p) * abundance.')
parser.add_argument('--hurdle_pres_weight', type=float, default=0.4,
                    help='Weight on presence loss in hurdle (default 0.4 of total).')

# Hybrid loss: logcosh abundance + BCE presence/absence
parser.add_argument('--bce_weight', type=float, default=0.0,
                    help='Weight w for presence/absence BCE term in hybrid loss: '
                         'total_loss = (1-w)*logcosh_abundance + w*BCE_presence. '
                         'BCE fires on ALL observations (zeros included), giving rare '
                         'species (<5%% haul prevalence) persistent gradient signal. '
                         'Recommended starting value: 0.1.  0=disabled (default).')

# Rolling window cross-validation
parser.add_argument('--rolling', action='store_true', default=False,
                    help='Run walk-forward rolling window cross-validation')
parser.add_argument('--rolling_windowed', action='store_true', default=False,
                    help='Run windowed rolling CV: merge 3-year windows, outer LSTM learns across 10 windows')
parser.add_argument('--rolling_start', type=int, default=2009,
                    help='First test year in rolling CV (default: 2009)')
parser.add_argument('--rolling_end', type=int, default=2024,
                    help='Last test year in rolling CV (default: 2024)')
parser.add_argument('--train_window', type=int, default=10,
                    help='Number of training years per rolling window (default: 10)')
parser.add_argument('--win_size', type=int, default=3,
                    help='Years to merge per window (default: 3)')
parser.add_argument('--n_windows', type=int, default=10,
                    help='Number of sliding windows fed to outer LSTM (default: 10)')

# Device selection
parser.add_argument('--device', type=str, default='auto',
                    choices=['auto', 'cuda', 'mps', 'cpu'],
                    help='Device to train on: auto (cuda→mps→cpu), cuda, mps, or cpu')

# GNN specific
parser.add_argument('-n_layers', "--n_layers", default=2, type=int, help='Number of GAT/GNN layers')
parser.add_argument('-dropout', "--dropout", default=0.5, type=float, help='dropout')
parser.add_argument('-aggregator_type', "--aggregator_type", default="mean", choices=["mean", "gcn", "pool", "lstm"])

# HybridTransformerEncoder / feature layout
parser.add_argument('--individual_indices', type=str, default='',
                    help='Comma-separated int indices for individual feature tokens '
                         '(default: auto-derived as range(0,42)). '
                         'Override only if feature layout changes.')
parser.add_argument('--n_species_for_attn', type=int, default=14,
                    help='Number of learnable query vectors in SpeciesAttentionPool '
                         '(default: 14, matching the 14 target species).')

# Added: dataset params
parser.add_argument('-crop_type', '--crop_type', choices=["corn", "cotton", "sorghum", "soybeans", "spring_wheat", "winter_wheat"])
parser.add_argument('-num_weather_vars', "--num_weather_vars", default=23, type=int, help='Number of daily weather vars')
parser.add_argument('-num_management_vars', "--num_management_vars", default=96, type=int, help='Number of weekly management (crop progress) variables')
parser.add_argument('-num_soil_vars', "--num_soil_vars", default=20, type=int, help='Number of depth-dependent soil vars')
parser.add_argument('-num_extra_vars', "--num_extra_vars", default=6, type=int, help='Number of extra vars')
parser.add_argument('-soil_depths', "--soil_depths", default=6, type=int, help='Number of depths in the gSSURGO dataset')
parser.add_argument('-no_management', "--no_management", default=False, action='store_true', help='Whether to completely ignore management data')
parser.add_argument('-no_soil', "--no_soil", default=False, action='store_true', help='Whether to completely ignore soil data')
parser.add_argument('-train_week_start', "--train_week_start", default=52, type=int)
parser.add_argument('-validation_week', "--validation_week", default=52, type=int)
parser.add_argument('-mask_prob', "--mask_prob", default=1, type=float)
parser.add_argument('-mask_value', "--mask_value", choices=['zero', 'county_avg'], default='zero')
parser.add_argument('--train_start_year', type=int, default=None,
                    help='If set, only include data from this year onward during training.')


args = parser.parse_args()
args.model = "gat_rnn_v2"

# GAT_RNN_V2 always uses nested 2D mode and GAT
args.nested  = True
args.use_gat = True

# ── FIM / stations_monthly OR restoration dataset ─────────────────────────────
if "stations_monthly" in args.data_dir or "stations_monthly" in args.dataset \
        or "restoration" in args.data_dir or "restoration" in args.dataset:
    # Monthly data: one time-step per node per year (no within-year sequence)
    args.time_intervals = 1
    args.progress_indices = []

    # Load species names from pickle next to the data file
    data_dir_path = os.path.dirname(os.path.abspath(args.data_dir))
    is_restoration = "restoration" in args.data_dir or "restoration" in args.dataset

    # Derive prefix from NPZ stem
    _npz_stem = os.path.splitext(os.path.basename(args.data_dir))[0]
    _prefix_candidate = _npz_stem.replace("_stations_monthly", "")
    _spp_candidate    = os.path.join(data_dir_path, f"{_prefix_candidate}_species_names.pkl")

    if os.path.exists(_spp_candidate):
        spp_pkl = _spp_candidate
    elif is_restoration:
        spp_pkl = os.path.join(data_dir_path, "FIM_restoration_species_names.pkl")
    else:
        spp_pkl = os.path.join(data_dir_path, "FIM_species_names.pkl")

    if os.path.exists(spp_pkl):
        with open(spp_pkl, "rb") as f:
            spp_data = pickle.load(f)

        if isinstance(spp_data, dict):
            target_spp        = spp_data['target_species']
            community_species = spp_data['feature_species']

            # Move forage species out of targets → add as input features (other_species_cols)
            forage_set = set()
            if getattr(args, 'forage_species', ''):
                forage_set = {s.strip() for s in args.forage_species.split(',') if s.strip()}
            forage_in_targets = [s for s in target_spp if s in forage_set]
            target_spp_final  = [s for s in target_spp if s not in forage_set]
            if forage_in_targets:
                print(f"  [forage] Moving {forage_in_targets} from targets → input features")

            # output_idx: data cols for target species (skip forage cols)
            all_target_cols = list(range(2, 2 + len(target_spp)))
            forage_cols = [2 + i for i, s in enumerate(target_spp) if s in forage_set]
            target_cols = [c for c in all_target_cols if c not in forage_cols]

            # NPZ column layout (per preprocess_restoration_0412.py):
            #   cols 0-1:                             node_id, Year
            #   cols 2..2+n_target-1:                 14 target species (life-stage)
            #   cols 2+n_target..2+n_target+n_env-1:  58 env features
            #   cols 2+n_target+n_env..end:           92 community species (prey + nonprey)
            n_env = int(spp_data.get('n_env_features', 0))
            community_base = 2 + len(target_spp) + n_env

            # --all_species_targets: promote community/feature_species to targets
            if getattr(args, 'all_species_targets', False):
                community_cols = list(range(
                    community_base,
                    community_base + len(community_species)
                ))
                target_spp_final = target_spp_final + list(community_species)
                target_cols      = target_cols + community_cols
                # Community is now Y, not X — clear the feature list and the group slices
                _merged_n = len(target_spp_final)
                print(f"  [all_species_targets] Promoted {len(community_species)} community "
                      f"species to targets (cols {community_cols[0]}..{community_cols[-1]}). "
                      f"Total targets: {_merged_n}")
                community_species = []   # no longer input features

            # --prevalence_target_min: auto-promote community species by prevalence
            elif float(getattr(args, 'prevalence_target_min', 0.0) or 0.0) > 0.0:
                thresh = float(args.prevalence_target_min)
                # Compute prevalence from the NPZ training years (<= test_year - 2)
                import numpy as _np
                _npz = _np.load(args.data_dir)
                _all_data = _npz['data']
                _years = _all_data[:, 1].astype(int)
                _train_mask = _years <= (args.test_year - 2)
                _train_rows = _all_data[_train_mask]
                promoted_cols, promoted_names = [], []
                demoted_cols,  demoted_names  = [], []
                for i, sp_name in enumerate(community_species):
                    col = community_base + i
                    prev = (_train_rows[:, col] > 0).mean() if _train_mask.any() else 0.0
                    if prev >= thresh:
                        promoted_cols.append(col)
                        promoted_names.append(sp_name)
                    else:
                        demoted_cols.append(col)
                        demoted_names.append(sp_name)
                target_spp_final = target_spp_final + promoted_names
                target_cols      = target_cols + promoted_cols
                # Remaining community species go into X as other_species_cols
                forage_cols = forage_cols + demoted_cols
                community_species = demoted_names
                print(f"  [prevalence_target_min={thresh}] Promoted "
                      f"{len(promoted_names)} community species to targets. "
                      f"Total targets: {len(target_spp_final)}. "
                      f"Demoted {len(demoted_names)} rare species to input features.")

            # --nonprey_targets: promote ONLY non-prey community species to targets
            # Prey community species stay as input features (prepended to X).
            elif getattr(args, 'nonprey_targets', False):
                _n_prey = int(spp_data.get('n_prey_community', len(community_species) // 2))
                prey_species    = list(community_species[:_n_prey])
                nonprey_species = list(community_species[_n_prey:])
                prey_cols = list(range(
                    community_base,
                    community_base + _n_prey
                ))
                nonprey_cols = list(range(
                    community_base + _n_prey,
                    community_base + len(community_species)
                ))
                target_spp_final = target_spp_final + nonprey_species
                target_cols      = target_cols + nonprey_cols
                # Prey stays as input features (prepended via other_species_cols);
                # forage_cols (if any) were already in that list.
                forage_cols = forage_cols + prey_cols
                # community_species list used downstream → keep only prey for grouping
                community_species = prey_species
                print(f"  [nonprey_targets] Promoted {len(nonprey_species)} non-prey "
                      f"community species to targets (cols {nonprey_cols[0]}..{nonprey_cols[-1]}). "
                      f"Total targets: {len(target_spp_final)}. "
                      f"Kept {len(prey_species)} prey species as input features "
                      f"(cols {prey_cols[0]}..{prey_cols[-1]}).")

            args.output_names       = target_spp_final
            args.output_idx         = target_cols
            args.other_species_cols = forage_cols   # prepended to X as forage features
            # Explicit X feature columns — needed because the NPZ layout is
            # [targets | env | community] rather than [all_species | env], so
            # slicing `data[:, feature_start:]` breaks when community species
            # are split between target (nonprey) and feature (prey) sets.
            # X layout (without forage prepend): env(58) + remaining_community
            _env_cols_range = list(range(2 + len(target_spp),
                                         2 + len(target_spp) + n_env))
            _remaining_comm_cols = [c for c in range(community_base,
                                                     community_base + len(spp_data['feature_species']))
                                    if c not in set(target_cols) and c not in set(forage_cols)]
            args.feature_cols = _env_cols_range + _remaining_comm_cols
            args.n_env_feats        = int(spp_data.get('n_env_features', 0))
            args.feature_groups     = spp_data.get('feature_groups', None)
            args.group_names        = spp_data.get('group_names', None)

            # Build HybridTransformerEncoder feature layout — 9 group token design:
            # Feature block: env(6) + restoration(8) + habitat(14) + shoreline(7) +
            #                water_effort(3) + bycatch(14) + na_indicators(6) = 58 ecological
            #                + 92 community species = 150 total
            #
            # 0 individual tokens, 10 group tokens — each semantic block as one contextual embedding:
            #   env(0-5):               abiotic water quality
            #   restoration(6-13):      site restoration status
            #   habitat_structured(14-20): biogenic habitat (seagrass, oyster, mangrove...)
            #   habitat_open(21-27):    unvegetated / open-water habitat types
            #   shoreline(28-34):       shoreline type and length characteristics
            #   water_effort(35-37):    sampling effort (gear, duration, area)
            #   bycatch(38-51):         co-caught community signal
            #   na_indicators(52-57):   data completeness / QC flags
            #   prey_community:         47 species caught exclusively by seine (Gear 20/23)
            #   nonprey_community:      45 species ever caught by Gear 160 (drop/cast net)
            #
            # Total: 0 + 10 = 10 tokens → O(100) attention pairs (18× fewer than original 1764)
            if args.individual_indices:
                args.individual_indices = [int(x) for x in args.individual_indices.split(',')]
            else:
                args.individual_indices = []   # all-group design: no individual tokens

            # Mode flags
            _is_nonprey_mode    = getattr(args, 'nonprey_targets', False)
            _is_all_mode        = getattr(args, 'all_species_targets', False)
            _is_prevalence_mode = (float(getattr(args, 'prevalence_target_min', 0.0) or 0.0) > 0.0)

            n_forage    = len(forage_cols)       # forage + anything prepended to X
            n_community = len(community_species) # species still in X (varies per mode)
            n_eco_total = args.n_env_feats       # 58

            if _is_nonprey_mode:
                # X layout: forage(raw_n_forage) | prey(n_prey) | env(58)
                raw_n_forage = n_forage - n_community
                n_prey_in_X  = n_community
                o = n_forage
            elif _is_prevalence_mode:
                # X layout: forage(n_forage, = original forage + demoted_community) | env(58)
                o = n_forage
            elif _is_all_mode:
                # X layout: forage(n_forage) | env(58)
                o = n_forage
            else:
                # Default: forage(n_forage) | env(58) | community(92 prey+nonprey)
                o = n_forage
                n_prey = int(spp_data.get('n_prey_community', n_community // 2))

            mid_habitat = o + 14 + 7
            args.group_slices = [
                (o+0,            o+6,            'env'),
                (o+6,            o+14,           'restoration'),
                (o+14,           mid_habitat,    'habitat_structured'),
                (mid_habitat,    o+28,           'habitat_open'),
                (o+28,           o+35,           'shoreline'),
                (o+35,           o+38,           'water_effort'),
                (o+38,           o+52,           'bycatch'),
                (o+52,           o+58,           'na_indicators'),
            ]

            if _is_nonprey_mode:
                args.group_slices.insert(
                    0, (raw_n_forage, raw_n_forage + n_prey_in_X, 'prey_community')
                )
                if raw_n_forage > 0:
                    args.group_slices.insert(0, (0, raw_n_forage, 'forage'))
            elif _is_prevalence_mode:
                # Single combined feature block up-front: original forage + demoted community
                if n_forage > 0:
                    args.group_slices.insert(0, (0, n_forage, 'other_species'))
            elif _is_all_mode:
                if n_forage > 0:
                    args.group_slices.insert(0, (0, n_forage, 'forage'))
            else:
                community_start = o + n_eco_total
                community_end   = community_start + n_community
                prey_end        = community_start + n_prey
                args.group_slices.extend([
                    (community_start, prey_end,     'prey_community'),
                    (prey_end,       community_end, 'nonprey_community'),
                ])
                if n_forage > 0:
                    args.group_slices.insert(0, (0, n_forage, 'forage'))

            print(f"FIM targets: {len(target_spp_final)} life-stage species (_A/_R/_SA/_J)")
            print(f"FIM forage (input only): {forage_in_targets}")
            print(f"FIM community species as input features: {len(community_species)}")
            print(f"FIM n_env_feats: {args.n_env_feats}")
            print(f"HybridEncoder individual_indices: {len(args.individual_indices)} features")
            print(f"HybridEncoder group_slices: {args.group_slices}")

        else:
            # Legacy list format
            species_names = spp_data
            if is_restoration:
                target_spp = [s for s in species_names
                              if s.endswith('_A') or s.endswith('_R') or s.endswith('_J')
                              or s.endswith('_SA')]
            else:
                target_spp = [s for s in species_names
                              if s.endswith('_a') or s.endswith('_r') or s.endswith('_j')]
            target_set  = set(target_spp)
            other_spp   = [s for s in species_names if s not in target_set]

            args.output_names       = target_spp
            args.output_idx         = [species_names.index(s) + 2 for s in target_spp]
            args.other_species_cols = [species_names.index(s) + 2 for s in other_spp]
            args.n_env_feats        = 0
            args.feature_groups     = None
            args.group_names        = None

            # Fallback feature layout
            args.individual_indices = list(range(0, 42))
            args.group_slices = [
                (42,  89),
                (89,  134),
                (134, 148),
                (148, 155),
                (155, 158),
                (158, 170),
            ]
            print(f"FIM target species ({len(target_spp)}): {target_spp}")
            print(f"FIM other species as features: {len(other_spp)}")
    else:
        raise ValueError(
            f"FIM species names not found at {spp_pkl}. "
            "Run data/preprocess_FIM.py first."
        )

    print(f"FIM dataset: {len(args.output_names)} species, time_intervals=1")

    # Resolve __prevalent_donors__ shorthand
    if getattr(args, 'use_prey_guild_donors', False):
        args.transfer_prior_species = ''
    if getattr(args, 'transfer_prior_species', '') == '__prevalent_donors__':
        if os.path.exists(spp_pkl):
            with open(spp_pkl, 'rb') as _f:
                _spp = pickle.load(_f)
            if isinstance(_spp, dict) and 'prevalent_donor_species' in _spp:
                donor_list = _spp['prevalent_donor_species']
                args.transfer_prior_species = ','.join(donor_list)
                print(f"  [BayesTransfer] Auto-resolved __prevalent_donors__ → "
                      f"{len(donor_list)} donors: {donor_list}")
            else:
                print("  [BayesTransfer] WARNING: '__prevalent_donors__' used but "
                      "'prevalent_donor_species' key not found in species pkl.")
                args.transfer_prior_species = ''
        else:
            print("  [BayesTransfer] WARNING: '__prevalent_donors__' but species pkl not found.")
            args.transfer_prior_species = ''

    # Bay exclusion: load station metadata and compute excluded node IDs
    _meta_candidate = os.path.join(data_dir_path, f"{_prefix_candidate}_station_metadata.pkl")
    if os.path.exists(_meta_candidate):
        meta_pkl = _meta_candidate
    elif is_restoration:
        meta_pkl = os.path.join(data_dir_path, "FIM_restoration_station_meta.pkl")
    else:
        meta_pkl = os.path.join(data_dir_path, "FIM_station_metadata.pkl")

    args.excluded_station_ids = set()
    if args.exclude_bays and os.path.exists(meta_pkl):
        with open(meta_pkl, "rb") as f:
            station_meta = pickle.load(f)
        bays_to_exclude = {b.strip().upper() for b in args.exclude_bays.split(",") if b.strip()}
        import pandas as _pd
        if isinstance(station_meta, _pd.DataFrame):
            _bay_col = 'Bay' if 'Bay' in station_meta.columns else 'bay'
            excl_mask = station_meta[_bay_col].str.upper().isin(bays_to_exclude)
            args.excluded_station_ids = set(station_meta.loc[excl_mask, 'station_id'].unique())
        else:
            args.excluded_station_ids = {
                sid for sid, v in station_meta.items() if v["bay"].upper() in bays_to_exclude
            }
        print(f"Excluding bays {bays_to_exclude}: {len(args.excluded_station_ids)} stations removed")

else:
    # Crop dataset: validate crop_type
    if args.crop_type is None or args.crop_type not in args.dataset:
        print("Alert! Did you forget to change the 'crop_type' param? You set 'crop_type' to",
              args.crop_type, "but 'dataset' to", args.dataset)
        exit(1)

    args.output_idx = OUTPUT_INDICES[args.crop_type]
    args.output_names = [args.crop_type]
    args.individual_indices = list(range(0, 42))
    args.group_slices = [(42, 89), (89, 134), (134, 148), (148, 155), (155, 158), (158, 170)]

    if "daily" in args.data_dir:
        args.time_intervals = 365
        args.progress_indices = PROGRESS_INDICES_DAILY[args.crop_type]
    elif "weekly" in args.data_dir or args.data_dir.endswith(".npy") or args.data_dir == "soybean_data_full.npz":
        args.time_intervals = 52
        args.progress_indices = PROGRESS_INDICES_WEEKLY[args.crop_type]
    else:
        raise ValueError("Data file must contain the string 'daily' or 'weekly'")

print("Time intervals", args.time_intervals)

if __name__ == "__main__":
    if args.rolling_windowed:
        from train import train_rolling_windowed
        train_rolling_windowed(args)
    elif args.rolling:
        from train import train_rolling
        train_rolling(args)
    elif args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        from train import train  # test mode not yet implemented in v2
        raise NotImplementedError("Use -mode train for v2; test evaluation runs at end of training.")
    else:
        raise ValueError("mode %s is not supported." % args.mode)
