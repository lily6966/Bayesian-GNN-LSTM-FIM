"""
Build distance-weighted + habitat-similarity within-bay k-NN adjacency
for the FIM_restoration_0412 dataset.

Input:  data/FIM_restoration_0412_station_metadata.pkl  (DataFrame)
        data/FIM_restoration_0412_stations_monthly.npz  (preprocessed data)
        data/FIM_restoration_0412_species_names.pkl     (feature group boundaries)

Outputs:
  map/FIM_restoration_0412_adj.pkl           – scipy CSR [N_nodes, N_nodes]
  map/FIM_restoration_0412_dist_weights.pkl  – {(src, dst): np.array([dist_w, hab_sim])}
  map/FIM_restoration_0412_fid_dict.pkl      – {node_id: sequential_idx}

Edge feature vector (d_edge = 2):
  [0] dist_w    : Gaussian distance weight  exp(-d² / 2σ²), σ = median intra-bay dist
  [1] hab_sim   : cosine similarity between per-station median habitat profiles
                  (habitat + shoreline feature groups from the preprocessed data)

Backward compatibility:
  Old code that expected scalar dist_weights still works because values are now
  np.float32 arrays of shape (2,).  The EGATConv layer reads d_edge = values[0].shape[0].
  Old scalar pickles (shape=()) are auto-detected and extended with hab_sim=1.0.
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
OUT_ADJ  = os.path.join(MAP_DIR,  'FIM_restoration_0412_adj.pkl')
OUT_DIST = os.path.join(MAP_DIR,  'FIM_restoration_0412_dist_weights.pkl')
OUT_FID  = os.path.join(MAP_DIR,  'FIM_restoration_0412_fid_dict.pkl')

K_NEIGHBORS = 5


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def load_station_habitat_profiles(npz_path, spp_pkl_path):
    """
    Compute per-station median habitat feature vector from the preprocessed data.

    The feature block layout (from preprocess_restoration_0412.py):
      habitat   cols  11–21  (11 features: BottomVegCover, bMud, bSan, bStr, bUnk,
                                            SAV, Alg, Non, HA, TH, RU)
      shoreline cols  22–25  ( 4 features: Nat, Ove, Str, Eme)

    Both groups capture structural habitat type and are relatively stable across
    time at a given station → use temporal median to get a representative profile.

    Returns: dict { station_id (int) → unit-normed np.ndarray [15] }
    """
    # Get feature group boundaries
    with open(spp_pkl_path, 'rb') as f:
        spp = pickle.load(f)
    feature_groups = spp.get('feature_groups', None)
    n_targets      = len(spp['target_species'])    # 14

    if feature_groups is not None:
        # Locate habitat and shoreline group boundaries within feature block
        grp_map = {name: (s, e) for name, s, e in feature_groups}
        hab_s, hab_e = grp_map.get('habitat',   (11, 22))
        sho_s, sho_e = grp_map.get('shoreline', (22, 26))
    else:
        hab_s, hab_e = 11, 22
        sho_s, sho_e = 22, 26

    # In the data matrix:  col 0 = node_id, col 1 = year,
    #   cols 2 … 2+n_targets-1 = target species,
    #   cols 2+n_targets … = feature block
    feat_offset = 2 + n_targets            # absolute start of feature block

    raw   = np.load(npz_path)
    data  = raw['data']                    # [N_rows, n_cols]
    node_ids = data[:, 0].astype(int)
    station_ids = node_ids // 12           # node_id = station_id * 12 + month_idx

    # Absolute column indices for habitat (incl. shoreline)
    feat_cols = list(range(feat_offset + hab_s, feat_offset + hab_e)) + \
                list(range(feat_offset + sho_s, feat_offset + sho_e))
    hab_feats = data[:, feat_cols].astype(np.float32)

    # Per-station median (ignoring NaN)
    profiles = {}
    for sid in np.unique(station_ids):
        mask  = station_ids == sid
        rows  = hab_feats[mask]
        with np.errstate(all='ignore'):
            med = np.nanmedian(rows, axis=0)
        med = np.nan_to_num(med, nan=0.0)
        profiles[int(sid)] = med

    # Unit-normalise each profile for cosine similarity
    for sid in profiles:
        norm = np.linalg.norm(profiles[sid])
        if norm > 1e-9:
            profiles[sid] = profiles[sid] / norm
        else:
            profiles[sid] = profiles[sid]  # all-zero → cosine will be 0

    print(f"  Habitat profiles: {len(profiles)} stations, "
          f"{len(feat_cols)} features (habitat={hab_e-hab_s}, shoreline={sho_e-sho_s})")
    return profiles


def habitat_cosine(profiles, si, sj):
    """Cosine similarity ∈ [-1, 1] between unit-normed profiles."""
    hi = profiles.get(si)
    hj = profiles.get(sj)
    if hi is None or hj is None:
        return 0.0
    sim = float(np.dot(hi, hj))        # both unit-normed → dot = cosine
    return float(np.clip(sim, -1.0, 1.0))


def build_adj(meta_df, hab_profiles, k=K_NEIGHBORS):
    """
    Build CSR adjacency and 2-vector edge features.

    Edge feature [dist_w, hab_sim]:
      dist_w  : Gaussian distance weight (same as before)
      hab_sim : cosine similarity of habitat+shoreline feature profiles

    Returns adj, dist_weights, fid_dict, sigma_by_bay
    """
    # One row per station
    sta_df = meta_df[meta_df['month'] == 1].copy()
    sta_df = sta_df.sort_values('station_id').reset_index(drop=True)
    N_sta  = len(sta_df)
    N_nod  = N_sta * 12

    print(f"  {N_sta} stations → {N_nod} nodes")

    # Group by bay
    bays = {}
    for _, row in sta_df.iterrows():
        bays.setdefault(row['Bay'], []).append(row)

    sta_edges    = {}    # (src_sid, dst_sid) → (dist_w, hab_sim)
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

        # Habitat similarity stats within this bay
        sids     = [int(r['station_id']) for r in members]
        hab_sims = [habitat_cosine(hab_profiles, si, sj)
                    for si in sids for sj in sids if si != sj]
        print(f"  Bay {bay}: {len(members)} stations, "
              f"median_dist={sigma:.1f} km, "
              f"hab_sim mean={np.mean(hab_sims):.3f} std={np.std(hab_sims):.3f}")

        for si in sids:
            neighbours = sorted(
                [(sj, dists[(si, sj)]) for sj in sids if sj != si],
                key=lambda x: x[1]
            )[:k]
            for sj, d in neighbours:
                dist_w  = float(np.exp(-d**2 / (2 * sigma**2)))
                hab_sim = habitat_cosine(hab_profiles, si, sj)
                sta_edges[(si, sj)] = (dist_w, hab_sim)

    print(f"\n  Station-level edges: {len(sta_edges)}")

    # Expand to all 12 months
    rows, cols, vals = [], [], []
    dist_weights = {}

    for (si, sj), (dw, hs) in sta_edges.items():
        for m in range(12):
            src = si * 12 + m
            dst = sj * 12 + m
            rows.append(src)
            cols.append(dst)
            vals.append(dw)                            # CSR stores dist_w as weight
            # Edge feature vector: [dist_w, hab_sim]
            dist_weights[(src, dst)] = np.array([dw, hs], dtype=np.float32)

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

    print(f"\nBuilding within-bay k-NN adjacency (k={K_NEIGHBORS}) …")
    adj, dist_weights, fid_dict, sigma_by_bay = build_adj(meta_df, hab_profiles)

    with open(OUT_ADJ, 'wb') as f:
        pickle.dump(adj, f)
    print(f"\nSaved {OUT_ADJ}")

    with open(OUT_DIST, 'wb') as f:
        pickle.dump(dist_weights, f)
    print(f"Saved {OUT_DIST}  ({len(dist_weights)} edges, edge_feat_dim=2)")

    with open(OUT_FID, 'wb') as f:
        pickle.dump(fid_dict, f)
    print(f"Saved {OUT_FID}  ({len(fid_dict)} nodes)")

    ew = adj.data
    print(f"\nEdge weight (dist_w) stats: "
          f"min={ew.min():.4f}  mean={ew.mean():.4f}  max={ew.max():.4f}")
    print("Sigma (km) per bay:", {b: f"{s:.1f}" for b, s in sigma_by_bay.items()})

    # Summarise habitat similarity distribution
    all_hab = np.array([v[1] for v in dist_weights.values()])
    print(f"\nHabitat cosine similarity across all edges:")
    print(f"  mean={all_hab.mean():.3f}  std={all_hab.std():.3f}  "
          f"min={all_hab.min():.3f}  max={all_hab.max():.3f}")

    # Sanity check: high-similarity pairs
    top5 = sorted(dist_weights.items(), key=lambda kv: -kv[1][1])[:5]
    print("\nTop-5 habitat-similar edges (src, dst → hab_sim):")
    for (s, d), v in top5:
        print(f"  station {s//12} → {d//12}   dist_w={v[0]:.3f}  hab_sim={v[1]:.3f}")


if __name__ == '__main__':
    main()
