"""
Build distance-weighted within-bay k-NN adjacency for the 26 restoration stations.

Edge weight = Gaussian kernel:  w(i,j) = exp(-d_km^2 / (2 * sigma^2))
  where sigma = median within-bay pairwise distance (per bay).

Outputs:
  map/FIM_restoration_adj.pkl       – scipy CSR sparse matrix  [N_nodes, N_nodes]
                                      N_nodes = 26 stations * 12 months = 312
                                      values = distance weights (NOT row-normalised;
                                      training code row-normalises before use)
  map/FIM_restoration_dist_weights.pkl – {(src_node, dst_node): weight} for DGL edge feats
  map/FIM_restoration_fid_dict.pkl  – {node_id: sequential_idx}
"""

import os, pickle
import numpy as np
import scipy.sparse as sp
from math import radians, sin, cos, sqrt, atan2

MAP_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MAP_DIR, '..', 'data')
META_PKL = os.path.join(DATA_DIR, 'FIM_restoration_station_metadata.pkl')
OUT_ADJ  = os.path.join(MAP_DIR,  'FIM_restoration_adj.pkl')
OUT_DIST = os.path.join(MAP_DIR,  'FIM_restoration_dist_weights.pkl')
OUT_FID  = os.path.join(MAP_DIR,  'FIM_restoration_fid_dict.pkl')

K_NEIGHBORS = 5   # max neighbours per station (within same bay)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def build_adj(station_meta, k=K_NEIGHBORS):
    sids  = sorted(station_meta.keys())           # 0..25
    N_sta = len(sids)
    N_nod = N_sta * 12                            # one node per station-month

    # Group stations by bay
    bays = {}
    for sid in sids:
        b = station_meta[sid]['bay']
        bays.setdefault(b, []).append(sid)

    # Station-level edges (month=0 slice; replicated to all months below)
    sta_edges  = {}    # (src_sid, dst_sid) → weight
    sigma_by_bay = {}

    for bay, members in bays.items():
        # Compute all pairwise distances within this bay
        dists = {}
        for i, si in enumerate(members):
            for j, sj in enumerate(members):
                if i == j: continue
                d = haversine_km(
                    station_meta[si]['lat'], station_meta[si]['lon'],
                    station_meta[sj]['lat'], station_meta[sj]['lon'])
                dists[(si, sj)] = d

        if not dists:
            continue

        all_d = list(dists.values())
        sigma = float(np.median(all_d)) if all_d else 1.0
        sigma = max(sigma, 0.1)               # floor to avoid div-by-zero
        sigma_by_bay[bay] = sigma
        print(f"  Bay {bay}: {len(members)} stations, "
              f"median dist={sigma:.1f} km, range=[{min(all_d):.1f}, {max(all_d):.1f}] km")

        # k-NN per station, distance-weighted
        for si in members:
            neighbours = sorted(
                [(sj, dists[(si, sj)]) for sj in members if sj != si],
                key=lambda x: x[1]
            )[:k]
            for sj, d in neighbours:
                w = float(np.exp(-d**2 / (2 * sigma**2)))
                sta_edges[(si, sj)] = w

    print(f"\n  Station-level edges: {len(sta_edges)}")

    # ── Expand to node-level (all 12 months) ──────────────────────────────────
    rows, cols, vals = [], [], []
    dist_weights = {}     # (node_src, node_dst) → weight

    for (si, sj), w in sta_edges.items():
        for m in range(12):
            src_node = si * 12 + m
            dst_node = sj * 12 + m
            rows.append(src_node)
            cols.append(dst_node)
            vals.append(w)
            dist_weights[(src_node, dst_node)] = w

    adj = sp.csr_matrix((vals, (rows, cols)), shape=(N_nod, N_nod), dtype=np.float32)
    print(f"  Node adjacency: {adj.shape}, nnz={adj.nnz}")

    # fid_dict: node_id → sequential index (needed by utils.py)
    # Only include nodes that appear in the adjacency
    all_nodes = sorted(set(rows) | set(cols))
    fid_dict  = {nid: idx for idx, nid in enumerate(all_nodes)}
    print(f"  fid_dict: {len(fid_dict)} active nodes")

    return adj, dist_weights, fid_dict, sigma_by_bay


def main():
    print(f"Loading station metadata from {META_PKL}")
    with open(META_PKL, 'rb') as f:
        station_meta = pickle.load(f)
    print(f"  {len(station_meta)} stations")

    adj, dist_weights, fid_dict, sigma_by_bay = build_adj(station_meta)

    with open(OUT_ADJ, 'wb') as f:
        pickle.dump(adj, f)
    print(f"\nSaved {OUT_ADJ}")

    with open(OUT_DIST, 'wb') as f:
        pickle.dump(dist_weights, f)
    print(f"Saved {OUT_DIST}  ({len(dist_weights)} edges)")

    with open(OUT_FID, 'wb') as f:
        pickle.dump(fid_dict, f)
    print(f"Saved {OUT_FID}  ({len(fid_dict)} nodes)")

    # Summary
    import scipy.sparse as _sp
    adj_csr = _sp.csr_matrix(adj)
    ew = adj_csr.data
    print(f"\nEdge weight stats: min={ew.min():.4f} mean={ew.mean():.4f} max={ew.max():.4f}")
    print("Sigma (km) per bay:", {b: f"{s:.1f}" for b, s in sigma_by_bay.items()})


if __name__ == '__main__':
    main()
