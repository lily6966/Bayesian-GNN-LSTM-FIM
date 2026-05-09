"""
Build within-bay k-NN adjacency matrix for FIM (station × month) nodes.

Adjacency rules:
  - Stations are connected ONLY if they share the same Bay code.
  - Within a bay, each station-month node is connected to its k nearest
    same-bay, same-month stations (by haversine distance).
  - Self-loops are added (diagonal = 1).
  - No cross-bay edges.
  - Only nodes that actually appear in the preprocessed data are included.

Saves:
  map/FIM_adj.pkl      — keys: 'adj' (scipy.sparse.csr_matrix), 'ctid_to_order'
  map/FIM_fid_dict.pkl — keys: 'fid_dict' (identity mapping node_id→node_id)
"""

import os
import pickle
import math
import numpy as np
import scipy.sparse as sp

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
META_PATH = os.path.join(DATA_DIR, "FIM_station_metadata.pkl")
NPZ_PATH  = os.path.join(DATA_DIR, "FIM_data_stations_monthly.npz")
OUT_ADJ   = os.path.join(os.path.dirname(__file__), "FIM_adj.pkl")
OUT_FID   = os.path.join(os.path.dirname(__file__), "FIM_fid_dict.pkl")

K_NEIGHBORS = 5  # k-NN within same bay, same month


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def build_adjacency(station_meta: dict, actual_node_ids: set, k: int = K_NEIGHBORS):
    """
    Build sparse adjacency matrix for only the node_ids that appear in the data.

    station_meta: {station_id: {'lat': float, 'lon': float, 'bay': str}}
    actual_node_ids: set of node_ids present in the preprocessed npz
    Returns:
      adj: scipy.sparse.csr_matrix (N_nodes × N_nodes)
      ctid_to_order: dict {node_id: row_index}
      order_to_ctid: dict {row_index: node_id}
    """
    # Only include node_ids that actually have data
    node_ids = sorted(actual_node_ids)
    N_nodes = len(node_ids)
    print(f"  Building adj for {N_nodes} actual data nodes")

    ctid_to_order = {nid: i for i, nid in enumerate(node_ids)}
    order_to_ctid = {i: nid for nid, i in ctid_to_order.items()}

    # Use LIL matrix for efficient construction
    adj = sp.lil_matrix((N_nodes, N_nodes), dtype=np.int8)

    # Self-loops
    for i in range(N_nodes):
        adj[i, i] = 1

    # Group stations by bay
    bay_stations = {}
    for sid, info in station_meta.items():
        bay = info["bay"]
        bay_stations.setdefault(bay, []).append(sid)

    # For each month, connect k nearest same-bay stations
    for month_idx in range(12):
        for bay, sids in bay_stations.items():
            if len(sids) <= 1:
                continue

            # Only include stations whose node_id for this month is in the data
            node_indices = []
            coords = []
            for sid in sids:
                nid = sid * 12 + month_idx
                if nid in ctid_to_order:
                    node_indices.append(ctid_to_order[nid])
                    coords.append((station_meta[sid]["lat"], station_meta[sid]["lon"]))

            n = len(node_indices)
            if n <= 1:
                continue
            k_actual = min(k, n - 1)

            # Compute pairwise distances and connect k-NN
            for i in range(n):
                lat_i, lon_i = coords[i]
                dists = []
                for j in range(n):
                    if i == j:
                        continue
                    d = haversine_km(lat_i, lon_i, coords[j][0], coords[j][1])
                    dists.append((d, j))
                dists.sort()

                for _, j in dists[:k_actual]:
                    row_i = node_indices[i]
                    row_j = node_indices[j]
                    adj[row_i, row_j] = 1
                    adj[row_j, row_i] = 1  # symmetric

    # Convert to CSR for efficient arithmetic and slicing
    adj_csr = adj.tocsr()
    return adj_csr, ctid_to_order, order_to_ctid


def main():
    # Load station metadata
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(
            f"Station metadata not found at {META_PATH}. "
            "Run data/preprocess_FIM.py first."
        )
    with open(META_PATH, "rb") as f:
        station_meta = pickle.load(f)
    print(f"Loaded station metadata: {len(station_meta)} stations")

    # Load actual node_ids from the preprocessed npz
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(
            f"Preprocessed data not found at {NPZ_PATH}. "
            "Run data/preprocess_FIM.py first."
        )
    raw = np.load(NPZ_PATH)
    actual_node_ids = set(raw["data"][:, 0].astype(int).tolist())
    print(f"Unique node_ids in data: {len(actual_node_ids)}")

    # Count bays
    bays = set(info["bay"] for info in station_meta.values())
    print(f"Bays: {sorted(bays)}")

    # Build adjacency
    print(f"Building within-bay k-NN adjacency (k={K_NEIGHBORS}) ...")
    adj, ctid_to_order, order_to_ctid = build_adjacency(
        station_meta, actual_node_ids, k=K_NEIGHBORS
    )

    n_edges = adj.nnz - len(actual_node_ids)  # subtract self-loops
    print(f"Adjacency shape: {adj.shape}")
    print(f"Edges (excluding self-loops): {n_edges}")
    mem_mb = (adj.data.nbytes + adj.indices.nbytes + adj.indptr.nbytes) / 1e6
    print(f"Sparse adj memory: {mem_mb:.1f} MB  (vs {adj.shape[0]**2*4/1e9:.1f} GB dense)")

    # Save adjacency pickle
    adj_data = {
        "adj": adj,
        "ctid_to_order": ctid_to_order,
        "order_to_ctid": order_to_ctid,
    }
    with open(OUT_ADJ, "wb") as f:
        pickle.dump(adj_data, f)
    print(f"Saved {OUT_ADJ}")

    # Save identity FID dict (node_id → node_id; no FIPS translation needed)
    fid_dict = {nid: nid for nid in ctid_to_order}
    fid_data = {"fid_dict": fid_dict}
    with open(OUT_FID, "wb") as f:
        pickle.dump(fid_data, f)
    print(f"Saved {OUT_FID}  ({len(fid_dict)} node entries)")


if __name__ == "__main__":
    main()
