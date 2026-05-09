"""
build_v2_adj.py — Build distance-weighted + habitat-similarity + env-similarity
within-bay k-NN adjacency for the GAT_RNN_V2 model.

Extends build_restoration_0412_adj.py by adding a 3rd edge feature:
  env_sim : cosine similarity of per-station median environmental profile
             [Temperature, Salinity, DissolvedO2, Secchi_depth]
             (first 4 features in the environmental feature group)

Input:  data/FIM_restoration_0412_station_metadata.pkl  (DataFrame)
        data/FIM_restoration_0412_stations_monthly.npz  (preprocessed data)
        data/FIM_restoration_0412_species_names.pkl     (feature group boundaries)

Outputs:
  map/FIM_restoration_0412_v2_adj.pkl           – scipy CSR [N_nodes, N_nodes]
  map/FIM_restoration_0412_v2_dist_weights.pkl  – {(src,dst): np.array([dist_w, hab_sim, env_sim])}
  map/FIM_restoration_0412_v2_fid_dict.pkl      – {node_id: sequential_idx}

Edge feature vector (d_edge = 3):
  [0] dist_w    : Gaussian distance weight  exp(-d² / 2σ²), σ = median intra-bay dist
  [1] hab_sim   : cosine similarity between per-station median habitat+shoreline profiles
  [2] env_sim   : cosine similarity between per-station median env profiles
                  (Temperature, Salinity, DissolvedO2, Secchi_depth)
"""

import os, pickle
import numpy as np
import scipy.sparse as sp
from math import radians, sin, cos, sqrt, atan2

MAP_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MAP_DIR, '..', 'data')
META_PKL = os.path.join(DATA_DIR, 'FIM_restoration_0412_station_metadata.pkl')
NPZ_PATH = os.path.join(DATA_DIR, 'FIM_restoration_0412_stations_monthly.npz')
SPP_PKL  = os.path.join(DATA_DIR, 'FIM_restoration_0412_species_names.pkl')
OUT_ADJ  = os.path.join(MAP_DIR,  'FIM_restoration_0412_v2_adj.pkl')
OUT_DIST = os.path.join(MAP_DIR,  'FIM_restoration_0412_v2_dist_weights.pkl')
OUT_FID  = os.path.join(MAP_DIR,  'FIM_restoration_0412_v2_fid_dict.pkl')

K_NEIGHBORS = 5

# Number of leading environmental features (Temperature, Salinity, DO, Secchi)
N_ENV_COLS = 4


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def load_station_habitat_profiles(npz_path, spp_pkl_path):
    """
    Compute per-station median habitat feature vector from the preprocessed data.

    Returns: dict { station_id (int) → unit-normed np.ndarray [15] }
    """
    with open(spp_pkl_path, 'rb') as f:
        spp = pickle.load(f)
    feature_groups = spp.get('feature_groups', None)
    n_targets      = len(spp['target_species'])

    if feature_groups is not None:
        grp_map = {name: (s, e) for name, s, e in feature_groups}
        hab_s, hab_e = grp_map.get('habitat',   (11, 22))
        sho_s, sho_e = grp_map.get('shoreline', (22, 26))
    else:
        hab_s, hab_e = 11, 22
        sho_s, sho_e = 22, 26

    feat_offset = 2 + n_targets

    raw       = np.load(npz_path)
    data      = raw['data']
    node_ids  = data[:, 0].astype(int)
    station_ids = node_ids // 12

    feat_cols = list(range(feat_offset + hab_s, feat_offset + hab_e)) + \
                list(range(feat_offset + sho_s, feat_offset + sho_e))
    hab_feats = data[:, feat_cols].astype(np.float32)

    profiles = {}
    for sid in np.unique(station_ids):
        mask = station_ids == sid
        rows = hab_feats[mask]
        with np.errstate(all='ignore'):
            med = np.nanmedian(rows, axis=0)
        med = np.nan_to_num(med, nan=0.0)
        profiles[int(sid)] = med

    for sid in profiles:
        norm = np.linalg.norm(profiles[sid])
        if norm > 1e-9:
            profiles[sid] = profiles[sid] / norm

    print(f"  Habitat profiles: {len(profiles)} stations, "
          f"{len(feat_cols)} features (habitat={hab_e-hab_s}, shoreline={sho_e-sho_s})")
    return profiles


def load_station_env_profiles(npz_path, spp_pkl_path, n_env_cols=N_ENV_COLS):
    """
    Compute per-station median environmental profile from the first n_env_cols
    features in the environmental feature block:
      [Temperature, Salinity, DissolvedO2, Secchi_depth]

    These are the first 4 columns of the feature block (absolute feature indices
    0..3 within the feature block, i.e. columns feat_offset+0 .. feat_offset+3).

    Returns: dict { station_id (int) → unit-normed np.ndarray [n_env_cols] }
    """
    with open(spp_pkl_path, 'rb') as f:
        spp = pickle.load(f)
    n_targets = len(spp['target_species'])
    feat_offset = 2 + n_targets

    raw       = np.load(npz_path)
    data      = raw['data']
    node_ids  = data[:, 0].astype(int)
    station_ids = node_ids // 12

    # First n_env_cols features in the feature block
    feat_cols = list(range(feat_offset, feat_offset + n_env_cols))
    env_feats = data[:, feat_cols].astype(np.float32)

    profiles = {}
    for sid in np.unique(station_ids):
        mask = station_ids == sid
        rows = env_feats[mask]
        with np.errstate(all='ignore'):
            med = np.nanmedian(rows, axis=0)
        med = np.nan_to_num(med, nan=0.0)
        profiles[int(sid)] = med

    # Unit-normalise for cosine similarity
    for sid in profiles:
        norm = np.linalg.norm(profiles[sid])
        if norm > 1e-9:
            profiles[sid] = profiles[sid] / norm

    print(f"  Env profiles: {len(profiles)} stations, "
          f"{n_env_cols} features [Temperature, Salinity, DO, Secchi_depth]")
    return profiles


def cosine_similarity(profiles, si, sj):
    """Cosine similarity ∈ [-1, 1] between unit-normed profiles."""
    hi = profiles.get(si)
    hj = profiles.get(sj)
    if hi is None or hj is None:
        return 0.0
    return float(np.clip(np.dot(hi, hj), -1.0, 1.0))


def build_adj(meta_df, hab_profiles, env_profiles, k=K_NEIGHBORS):
    """
    Build CSR adjacency and 3-vector edge features.

    Edge feature [dist_w, hab_sim, env_sim]:
      dist_w  : Gaussian distance weight
      hab_sim : cosine similarity of habitat+shoreline feature profiles
      env_sim : cosine similarity of [Temperature, Salinity, DO, Secchi_depth]

    Returns adj, dist_weights, fid_dict, sigma_by_bay
    """
    sta_df = meta_df[meta_df['month'] == 1].copy()
    sta_df = sta_df.sort_values('station_id').reset_index(drop=True)
    N_sta  = len(sta_df)
    N_nod  = N_sta * 12

    print(f"  {N_sta} stations → {N_nod} nodes")

    bays = {}
    for _, row in sta_df.iterrows():
        bays.setdefault(row['Bay'], []).append(row)

    sta_edges    = {}
    sigma_by_bay = {}

    for bay, members in bays.items():
        if len(members) < 2:
            print(f"  Bay {bay}: only 1 station, skipping")
            continue

        dists = {}
        for i, ri in enumerate(members):
            for j, rj in enumerate(members):
                if i == j:
                    continue
                si, sj = int(ri['station_id']), int(rj['station_id'])
                dists[(si, sj)] = haversine_km(ri['lat'], ri['lon'],
                                               rj['lat'], rj['lon'])

        all_d = list(dists.values())
        sigma = max(float(np.median(all_d)) if all_d else 1.0, 0.1)
        sigma_by_bay[bay] = sigma

        sids     = [int(r['station_id']) for r in members]
        hab_sims = [cosine_similarity(hab_profiles, si, sj)
                    for si in sids for sj in sids if si != sj]
        env_sims = [cosine_similarity(env_profiles, si, sj)
                    for si in sids for sj in sids if si != sj]

        print(f"  Bay {bay}: {len(members)} stations, "
              f"median_dist={sigma:.1f} km, "
              f"hab_sim mean={np.mean(hab_sims):.3f} std={np.std(hab_sims):.3f}, "
              f"env_sim mean={np.mean(env_sims):.3f} std={np.std(env_sims):.3f}")

        for si in sids:
            neighbours = sorted(
                [(sj, dists[(si, sj)]) for sj in sids if sj != si],
                key=lambda x: x[1]
            )[:k]
            for sj, d in neighbours:
                dist_w  = float(np.exp(-d**2 / (2 * sigma**2)))
                hab_sim = cosine_similarity(hab_profiles, si, sj)
                env_sim = cosine_similarity(env_profiles, si, sj)
                sta_edges[(si, sj)] = (dist_w, hab_sim, env_sim)

    print(f"\n  Station-level edges: {len(sta_edges)}")

    rows, cols, vals = [], [], []
    dist_weights = {}

    for (si, sj), (dw, hs, es) in sta_edges.items():
        for m in range(12):
            src = si * 12 + m
            dst = sj * 12 + m
            rows.append(src)
            cols.append(dst)
            vals.append(dw)
            dist_weights[(src, dst)] = np.array([dw, hs, es], dtype=np.float32)

    adj = sp.csr_matrix((vals, (rows, cols)), shape=(N_nod, N_nod), dtype=np.float32)
    print(f"  Node adjacency: {adj.shape}, nnz={adj.nnz}")

    all_nodes = sorted(set(rows) | set(cols))
    fid_dict  = {nid: idx for idx, nid in enumerate(all_nodes)}
    print(f"  fid_dict: {len(fid_dict)} active nodes")

    return adj, dist_weights, fid_dict, sigma_by_bay


def main():
    print(f"Loading metadata from {META_PKL}")
    with open(META_PKL, 'rb') as f:
        meta_df = pickle.load(f)
    print(f"  {len(meta_df)} rows (stations × months)")

    print(f"\nComputing habitat similarity profiles from {NPZ_PATH}")
    hab_profiles = load_station_habitat_profiles(NPZ_PATH, SPP_PKL)

    print(f"\nComputing environmental similarity profiles from {NPZ_PATH}")
    env_profiles = load_station_env_profiles(NPZ_PATH, SPP_PKL)

    print(f"\nBuilding within-bay k-NN adjacency (k={K_NEIGHBORS}) with 3 edge features …")
    adj, dist_weights, fid_dict, sigma_by_bay = build_adj(meta_df, hab_profiles, env_profiles)

    with open(OUT_ADJ, 'wb') as f:
        pickle.dump(adj, f)
    print(f"\nSaved {OUT_ADJ}")

    with open(OUT_DIST, 'wb') as f:
        pickle.dump(dist_weights, f)
    print(f"Saved {OUT_DIST}  ({len(dist_weights)} edges, edge_feat_dim=3)")

    with open(OUT_FID, 'wb') as f:
        pickle.dump(fid_dict, f)
    print(f"Saved {OUT_FID}  ({len(fid_dict)} nodes)")

    ew = adj.data
    print(f"\nEdge weight (dist_w) stats: "
          f"min={ew.min():.4f}  mean={ew.mean():.4f}  max={ew.max():.4f}")
    print("Sigma (km) per bay:", {b: f"{s:.1f}" for b, s in sigma_by_bay.items()})

    all_hab = np.array([v[1] for v in dist_weights.values()])
    all_env = np.array([v[2] for v in dist_weights.values()])
    print(f"\nHabitat cosine similarity across all edges:")
    print(f"  mean={all_hab.mean():.3f}  std={all_hab.std():.3f}  "
          f"min={all_hab.min():.3f}  max={all_hab.max():.3f}")
    print(f"\nEnvironmental cosine similarity across all edges:")
    print(f"  mean={all_env.mean():.3f}  std={all_env.std():.3f}  "
          f"min={all_env.min():.3f}  max={all_env.max():.3f}")

    # Top-5 env-similar pairs
    top5 = sorted(dist_weights.items(), key=lambda kv: -kv[1][2])[:5]
    print("\nTop-5 env-similar edges (src, dst → env_sim):")
    for (s, d), v in top5:
        print(f"  station {s//12} → {d//12}   dist_w={v[0]:.3f}  "
              f"hab_sim={v[1]:.3f}  env_sim={v[2]:.3f}")


if __name__ == '__main__':
    main()
