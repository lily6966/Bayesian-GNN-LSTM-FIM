"""
make_prey_guild_npz.py
----------------------
Creates a new NPZ (and matching pkl) where the top-10 seine-caught prey species
are moved from community features (cols 75..166) to become the FIRST 10 target
columns (cols 2..11), with the 14 original management targets shifted to cols 12..25.

Source NPZ column layout (167 cols):
  col 0        : node_id
  col 1        : year
  cols 2..15   : 14 management targets
  cols 16..74  : 59 env features  (feature_start = 16)
  cols 75..166 : 92 community species (feature_species)

Output NPZ column layout (167 cols, same total):
  col 0        : node_id
  col 1        : year
  cols 2..11   : 10 prey guild targets  (PREY FIRST)
  cols 12..25  : 14 management targets
  cols 26..84  : 59 env features        (feature_start = 26)
  cols 85..166 : 82 remaining community species (92 - 10 = 82)

Output files:
  data/FIM_restoration_0412_prey10_stations_monthly.npz
  data/FIM_restoration_0412_prey10_species_names.pkl
"""

import numpy as np
import pickle
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SRC_NPZ     = os.path.join(SCRIPT_DIR, 'FIM_restoration_0412_stations_monthly.npz')
SRC_PKL     = os.path.join(SCRIPT_DIR, 'FIM_restoration_0412_species_names.pkl')
OUT_NPZ     = os.path.join(SCRIPT_DIR, 'FIM_restoration_0412_prey10_stations_monthly.npz')
OUT_PKL     = os.path.join(SCRIPT_DIR, 'FIM_restoration_0412_prey10_species_names.pkl')

# ── Top-10 prey species (ordered by abundance) ───────────────────────────────
PREY_SP_LIST = [
    'Anchoa mitchilli',
    'Eucinostomus spp.',
    'Menidia spp.',
    'Leiostomus xanthurus',
    'Lucania parva',
    'Brevoortia spp.',
    'Harengula jaguana',
    'Eucinostomus harengulus',
    'Eucinostomus gula',
    'Anchoa hepsetus',
]

# ── Load source data ──────────────────────────────────────────────────────────
print(f"Loading {SRC_NPZ} …")
src = np.load(SRC_NPZ, allow_pickle=True)
data = src['data']               # (N, 167)  float32
feature_start_old = int(src['feature_start'][0])   # should be 16
print(f"  data shape: {data.shape}  feature_start_old={feature_start_old}")

print(f"Loading {SRC_PKL} …")
with open(SRC_PKL, 'rb') as f:
    sp = pickle.load(f)

original_target_sp_list = sp['target_species']       # 14 names
feature_species_all     = sp['feature_species']      # 92 names
n_env_features          = sp['n_env_features']       # 59
feature_groups          = sp['feature_groups']
group_names             = sp['group_names']

assert len(original_target_sp_list) == 14, "Expected 14 management targets"
assert len(feature_species_all)     == 92, "Expected 92 community species"
assert n_env_features               == 59, "Expected 59 env features"
assert feature_start_old            == 16, f"Expected feature_start=16, got {feature_start_old}"

# ── Resolve prey species indices within feature_species_all ──────────────────
prey_feat_indices = []   # indices within feature_species_all (0-based, into the 92-sp list)
for pname in PREY_SP_LIST:
    idx = next((i for i, s in enumerate(feature_species_all) if s == pname), None)
    if idx is None:
        raise ValueError(f"Prey species '{pname}' not found in feature_species! "
                         f"Available: {feature_species_all}")
    prey_feat_indices.append(idx)

print(f"\nPrey species → feature_species indices:")
for pname, fi in zip(PREY_SP_LIST, prey_feat_indices):
    print(f"  {fi:3d}: {pname}")

# Remaining 82 community species (not prey)
remaining_feat_indices = [i for i in range(len(feature_species_all))
                          if i not in set(prey_feat_indices)]
remaining_feat_species = [feature_species_all[i] for i in remaining_feat_indices]
assert len(remaining_feat_indices) == 82, f"Expected 82 remaining, got {len(remaining_feat_indices)}"

# ── Map column indices from source to target layout ──────────────────────────
# Source layout:
#   col 0           : node_id
#   col 1           : year
#   cols 2..15      : 14 mgmt targets      (offsets 0..13)
#   cols 16..74     : 59 env features      (feature_start=16)
#   cols 75..166    : 92 community species  (=16+59=75 to 75+92-1=166)
MGMT_COLS_SRC   = list(range(2, 16))          # 14 cols
ENV_COLS_SRC    = list(range(16, 75))         # 59 cols  (feature_start=16)
COMM_COLS_SRC   = list(range(75, 167))        # 92 cols  (community species)

# Prey cols within source (in feature_species_all order → offset within COMM_COLS_SRC)
prey_cols_src       = [COMM_COLS_SRC[fi] for fi in prey_feat_indices]          # 10 cols
remaining_cols_src  = [COMM_COLS_SRC[fi] for fi in remaining_feat_indices]     # 82 cols

# Target layout:
#   col 0           : node_id
#   col 1           : year
#   cols 2..11      : 10 prey targets
#   cols 12..25     : 14 mgmt targets
#   cols 26..84     : 59 env features  (feature_start=26)
#   cols 85..166    : 82 remaining community species
new_col_order = (
    [0, 1]                # node_id, year
    + prey_cols_src       # 10 prey
    + MGMT_COLS_SRC       # 14 mgmt
    + ENV_COLS_SRC        # 59 env
    + remaining_cols_src  # 82 remaining community
)
assert len(new_col_order) == 167, f"Column count mismatch: {len(new_col_order)}"

print(f"\nColumn remapping:")
print(f"  node_id / year            : cols 0-1 (unchanged)")
print(f"  10 prey targets           : cols 2-11")
print(f"  14 mgmt targets           : cols 12-25")
print(f"  59 env features           : cols 26-84  (feature_start=26)")
print(f"  82 remaining community    : cols 85-166")

# ── Build new data array ──────────────────────────────────────────────────────
new_data = data[:, new_col_order].astype(np.float32)
assert new_data.shape == (data.shape[0], 167)

new_feature_start = np.array([26], dtype=np.int64)

# ── Save NPZ ─────────────────────────────────────────────────────────────────
print(f"\nSaving {OUT_NPZ} …")
np.savez_compressed(
    OUT_NPZ,
    data          = new_data,
    feature_start = new_feature_start,
)
print(f"  Saved: shape={new_data.shape}  feature_start={new_feature_start[0]}")

# ── Build new pkl ─────────────────────────────────────────────────────────────
# target_species: prey first (indices 0..9), then management (indices 10..23)
new_target_species = PREY_SP_LIST + original_target_sp_list   # 24 total

# Indices within new_target_species
prey_guild_species_with_idx = list(range(0, 10))             # 0..9
management_target_with_idx  = list(range(10, 24))            # 10..23

new_pkl = {
    'target_species':            new_target_species,                 # 24 names
    'feature_species':           remaining_feat_species,             # 82 names
    'n_env_features':            n_env_features,                     # 59
    'feature_groups':            feature_groups,
    'group_names':               group_names,
    'prey_guild_species':        PREY_SP_LIST,                       # 10 names (indices 0..9)
    'management_target_species': original_target_sp_list,            # 14 names (indices 10..23)
    # Compatibility key for --transfer_prior_species __prevalent_donors__
    'prevalent_donor_species':   PREY_SP_LIST,
}

print(f"\nSaving {OUT_PKL} …")
with open(OUT_PKL, 'wb') as f:
    pickle.dump(new_pkl, f)
print(f"  Saved with keys: {list(new_pkl.keys())}")

# ── Quick sanity checks ───────────────────────────────────────────────────────
print("\n── Sanity checks ────────────────────────────────────────────────────────")

# Verify col 0 and col 1 unchanged
assert np.allclose(new_data[:, 0], data[:, 0]), "node_id column changed!"
assert np.allclose(new_data[:, 1], data[:, 1]), "year column changed!"
print("  node_id / year: unchanged ✓")

# Verify prey col 2 (Anchoa mitchilli) matches source col 75+4=79
src_anchoa_mitchilli = COMM_COLS_SRC[prey_feat_indices[0]]   # prey_feat_indices[0]=4
assert np.allclose(new_data[:, 2], data[:, src_anchoa_mitchilli]), \
    "Anchoa mitchilli column mismatch!"
print(f"  col 2 (Anchoa mitchilli) matches source col {src_anchoa_mitchilli} ✓")

# Verify mgmt col 12 matches source col 2
assert np.allclose(new_data[:, 12], data[:, 2]), "First mgmt target column mismatch!"
print("  col 12 (first mgmt target) matches source col 2 ✓")

# Verify env col 26 matches source col 16
assert np.allclose(new_data[:, 26], data[:, 16]), "First env feature column mismatch!"
print("  col 26 (first env feature) matches source col 16 ✓")

# Check pkl target_species count
assert len(new_pkl['target_species']) == 24, \
    f"Expected 24 target_species, got {len(new_pkl['target_species'])}"
print(f"  target_species: {len(new_pkl['target_species'])} (prey 0..9, mgmt 10..23) ✓")

assert len(new_pkl['feature_species']) == 82, \
    f"Expected 82 feature_species, got {len(new_pkl['feature_species'])}"
print(f"  feature_species: {len(new_pkl['feature_species'])} ✓")

print("\n── Done! ────────────────────────────────────────────────────────────────")
print(f"  NPZ : {OUT_NPZ}")
print(f"  PKL : {OUT_PKL}")
print(f"\n  New target_species (24 total):")
for i, name in enumerate(new_pkl['target_species']):
    tag = 'PREY' if i < 10 else 'MGMT'
    print(f"    [{i:2d}] [{tag}] {name}")
