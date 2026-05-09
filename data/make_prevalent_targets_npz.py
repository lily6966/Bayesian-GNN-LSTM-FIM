"""
Create FIM_restoration_0412_prev10_stations_monthly.npz
  — same as 0412 but top-10 prevalent community species become additional targets.

New column layout (feature_start = 2 + 14 + 10 = 26):
  col 0      : node_id
  col 1      : year
  col 2..15  : 14 original target species (unchanged)
  col 16..25 : 10 prevalent community species (moved from feature area)
  col 26..84 : 59 env/habitat features (unchanged)
  col 85..166: 82 remaining community species (unchanged)
"""

import numpy as np
import pickle
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load existing data ─────────────────────────────────────────────────────
src_npz = os.path.join(DATA_DIR, 'FIM_restoration_0412_stations_monthly.npz')
src_pkl = os.path.join(DATA_DIR, 'FIM_restoration_0412_species_names.pkl')

data_arr = np.load(src_npz, allow_pickle=True)['data']    # (54319, 167)
with open(src_pkl, 'rb') as f:
    sp_meta = pickle.load(f)

target_sp   = sp_meta['target_species']   # 14 species  → data cols 2..15
feature_sp  = sp_meta['feature_species']  # 92 species  → data cols 75..166
n_env       = int(sp_meta['n_env_features'])             # 59
feature_groups = sp_meta['feature_groups']
group_names    = sp_meta.get('group_names', None)

# Community species: cols 75..166  (indices 0..91 within the 92-col block)
comm_start = 2 + len(target_sp) + n_env   # = 75

# ── Compute prevalence on all training rows (year < 2007) ─────────────────
years = data_arr[:, 1]
train_mask = years < 2007
comm_data  = data_arr[train_mask, comm_start : comm_start + len(feature_sp)]
prev_comm  = (comm_data > 0).mean(axis=0)  # (92,)

# Top-10 prevalent community species (indices into feature_sp)
top10_idx  = list(np.argsort(prev_comm)[::-1][:10])
top10_sp   = [feature_sp[i] for i in top10_idx]
top10_prev = prev_comm[top10_idx]

print('Top-10 prevalent community species being added as targets:')
for i, (sp, p) in enumerate(zip(top10_sp, top10_prev)):
    print(f'  {i+1:2d}. {sp:<45} prev={p:.3f}')

# Remaining community species (not in top-10)
remain_idx = [i for i in range(len(feature_sp)) if i not in top10_idx]
remain_sp  = [feature_sp[i] for i in remain_idx]

# ── Build new column order ──────────────────────────────────────────────────
# [node_id, year, 14_orig_targets, 10_prevalent_targets, 59_env_feats, 82_remain_comm]
col_node_year   = [0, 1]
col_orig_target = list(range(2, 16))                               # 14 cols
col_prev_target = [comm_start + i for i in top10_idx]              # 10 cols (from community area)
col_env_feats   = list(range(16, 16 + n_env))                     # 59 cols → orig cols 16..74
col_remain_comm = [comm_start + i for i in remain_idx]             # 82 cols

new_cols = col_node_year + col_orig_target + col_prev_target + col_env_feats + col_remain_comm

assert len(new_cols) == data_arr.shape[1], \
    f'Column count mismatch: {len(new_cols)} vs {data_arr.shape[1]}'

new_data = data_arr[:, new_cols].astype(np.float32)
new_feature_start = 2 + len(target_sp) + len(top10_sp)  # = 26

print(f'\nNew data shape: {new_data.shape}')
print(f'feature_start: {new_feature_start}')
print(f'  cols 2..15   : 14 original targets')
print(f'  cols 16..25  : 10 prevalent targets')
print(f'  cols 26..84  : 59 env features')
print(f'  cols 85..166 : 82 remaining community species')

# ── Save new NPZ ───────────────────────────────────────────────────────────
out_npz = os.path.join(DATA_DIR, 'FIM_restoration_0412_prev10_stations_monthly.npz')
np.savez(out_npz, data=new_data, feature_start=np.array([new_feature_start]))
print(f'\nSaved: {out_npz}')

# ── Save new species pkl ───────────────────────────────────────────────────
new_target_sp = target_sp + top10_sp  # 24 total
new_feature_sp = remain_sp             # 82 remaining

new_sp_meta = {
    'target_species'  : new_target_sp,
    'feature_species' : new_feature_sp,
    'n_env_features'  : n_env,
    'feature_groups'  : feature_groups,
    'group_names'     : group_names,
    # Store top10 metadata for transfer prior setup
    'prevalent_donor_species' : top10_sp,
    'original_target_species' : target_sp,
    'prevalent_prevalence'    : {sp: float(prev_comm[i]) for sp, i in zip(top10_sp, top10_idx)},
}

out_pkl = os.path.join(DATA_DIR, 'FIM_restoration_0412_prev10_species_names.pkl')
with open(out_pkl, 'wb') as f:
    pickle.dump(new_sp_meta, f)
print(f'Saved: {out_pkl}')

print('\nNew target species:')
target_data_train = data_arr[train_mask, 2:16]
prev_target_orig  = (target_data_train > 0).mean(axis=0)
for sp, p in zip(target_sp, prev_target_orig):
    print(f'  [ORIG]  {sp:<45} prev={p:.3f}')
for sp, p in zip(top10_sp, top10_prev):
    print(f'  [DONOR] {sp:<45} prev={p:.3f}')
