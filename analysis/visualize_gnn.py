"""
FIM GNN Graph Visualization
============================
Plots the spatial graph structure used by the GNN:
  - All stations as nodes (lat/lon), colored by bay
  - k-NN edges between stations (within-bay)
  - Zoomed inset for a selected bay

Usage:
    python analysis/visualize_gnn.py \
        --adj   map/FIM_adj.pkl \
        --meta  data/FIM_station_metadata.pkl \
        --out_dir analysis/figures
"""

import argparse
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import scipy.sparse as sp


# ── helpers ───────────────────────────────────────────────────────────────────

def load_data(adj_path, meta_path):
    adj_dict = pickle.load(open(adj_path, "rb"))
    A        = adj_dict["adj"]           # CSR (n_nodes × n_nodes), temporal nodes
    o2c      = adj_dict["order_to_ctid"] # matrix row → composite node id
    meta     = pickle.load(open(meta_path, "rb"))  # station_id → {lat, lon, bay}
    return A, o2c, meta


def build_station_graph(A, o2c, meta, month=0):
    """
    Extract the station-level graph for a fixed month.
    Temporal node = station_id * 12 + month.
    Returns:
        stations : dict  station_id → {lat, lon, bay}
        edges    : list of (sid_i, sid_j) at station level (unique, undirected)
    """
    # Build set of ctids present in the matrix for the chosen month
    # ctid = station_id * 12 + month  →  station_id = ctid // 12
    n = A.shape[0]
    # Map matrix order → station_id for nodes belonging to `month`
    order_to_sid = {}
    for order, ctid in o2c.items():
        if ctid % 12 == month and (ctid // 12) in meta:
            order_to_sid[order] = ctid // 12

    # Build station-level edge list from sparse matrix
    rows, cols = A.nonzero()
    edge_set = set()
    for r, c in zip(rows, cols):
        if r in order_to_sid and c in order_to_sid:
            si, sj = order_to_sid[r], order_to_sid[c]
            if si != sj:
                edge_set.add((min(si, sj), max(si, sj)))

    stations = {sid: meta[sid] for sid in order_to_sid.values()}
    return stations, list(edge_set)


# ── plot 1: full Florida map ──────────────────────────────────────────────────

def plot_florida_map(stations, edges, out_dir, sample_frac=0.3):
    """
    Plot all stations on a lat/lon scatter with edges.
    Edges are sampled for readability.
    """
    bays    = sorted(set(v["bay"] for v in stations.values()))
    cmap    = plt.get_cmap("tab20", len(bays))
    bay_col = {b: cmap(i) for i, b in enumerate(bays)}

    lats = np.array([v["lat"] for v in stations.values()])
    lons = np.array([v["lon"] for v in stations.values()])
    cols = [bay_col[v["bay"]] for v in stations.values()]
    sid_list = list(stations.keys())
    sid_idx  = {s: i for i, s in enumerate(sid_list)}

    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw sampled edges first (behind nodes)
    rng = np.random.default_rng(42)
    edge_sample = edges if len(edges) < 5000 else [
        edges[i] for i in rng.choice(len(edges), 5000, replace=False)
    ]
    for si, sj in edge_sample:
        if si in sid_idx and sj in sid_idx:
            xi, yi = lons[sid_idx[si]], lats[sid_idx[si]]
            xj, yj = lons[sid_idx[sj]], lats[sid_idx[sj]]
            ax.plot([xi, xj], [yi, yj], color="gray", alpha=0.15, linewidth=0.4, zorder=1)

    # Draw nodes
    ax.scatter(lons, lats, c=cols, s=8, alpha=0.7, zorder=2, linewidths=0)

    # Legend (bays)
    patches = [mpatches.Patch(color=bay_col[b], label=b) for b in bays]
    ax.legend(handles=patches, title="Bay", fontsize=7, title_fontsize=8,
              loc="lower right", ncol=2, framealpha=0.9)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"FIM Sampling Stations — GNN Graph (k-NN edges, {len(stations):,} nodes, "
                 f"{len(edges):,} edges)")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    path = os.path.join(out_dir, "gnn_florida_map.png")
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"  Saved: {path}")


# ── plot 2: per-bay zoomed graph ──────────────────────────────────────────────

def plot_bay_graphs(stations, edges, out_dir, bays_to_plot=None):
    """
    For each selected bay, plot the local graph with labelled nodes.
    """
    all_bays = sorted(set(v["bay"] for v in stations.values()))
    if bays_to_plot is None:
        bays_to_plot = all_bays  # plot all

    bay_dir = os.path.join(out_dir, "gnn_bay_graphs")
    os.makedirs(bay_dir, exist_ok=True)

    # Build per-bay station sets
    bay_stations = {b: {} for b in all_bays}
    for sid, v in stations.items():
        bay_stations[v["bay"]][sid] = v

    # Build edge lookup
    edge_set = set(edges)

    for bay in bays_to_plot:
        bs = bay_stations[bay]
        if len(bs) < 2:
            continue

        sid_list = list(bs.keys())
        sid_idx  = {s: i for i, s in enumerate(sid_list)}
        lats = np.array([bs[s]["lat"] for s in sid_list])
        lons = np.array([bs[s]["lon"] for s in sid_list])

        # Edges within this bay
        local_edges = [(si, sj) for si, sj in edges
                       if si in sid_idx and sj in sid_idx]

        fig, ax = plt.subplots(figsize=(8, 7))

        for si, sj in local_edges:
            xi, yi = lons[sid_idx[si]], lats[sid_idx[si]]
            xj, yj = lons[sid_idx[sj]], lats[sid_idx[sj]]
            ax.plot([xi, xj], [yi, yj], color="steelblue",
                    alpha=0.5, linewidth=0.8, zorder=1)

        ax.scatter(lons, lats, color="darkorange", s=20, zorder=2,
                   edgecolors="black", linewidths=0.3)

        ax.set_title(f"Bay: {bay}  ({len(bs)} stations, {len(local_edges)} edges)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(bay_dir, f"gnn_bay_{bay}.png")
        plt.savefig(path, dpi=150)
        plt.close()

    print(f"  Saved bay graphs to: {bay_dir}/")


# ── plot 3: degree distribution ───────────────────────────────────────────────

def plot_degree_distribution(A, o2c, meta, out_dir, month=0):
    """
    Histogram of node degrees in the station-level graph.
    """
    # Compute per-row degree in the full temporal graph
    # then restrict to nodes belonging to `month`
    order_to_sid = {}
    for order, ctid in o2c.items():
        if ctid % 12 == month and (ctid // 12) in meta:
            order_to_sid[order] = ctid // 12

    orders = list(order_to_sid.keys())
    degrees = np.array(A[orders, :].sum(axis=1)).flatten()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(degrees, bins=range(0, int(degrees.max()) + 2),
            color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Node degree (# k-NN neighbours)")
    ax.set_ylabel("Number of stations")
    ax.set_title(f"Degree Distribution — FIM GNN Graph  "
                 f"(mean={degrees.mean():.1f}, median={np.median(degrees):.0f})")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "gnn_degree_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── plot 4: multi-panel overview ──────────────────────────────────────────────

def plot_overview_panel(stations, edges, A, o2c, meta, out_dir, month=0):
    """
    2×2 panel: Florida map | degree dist | two zoomed bays side-by-side.
    """
    bays    = sorted(set(v["bay"] for v in stations.values()))
    cmap    = plt.get_cmap("tab20", len(bays))
    bay_col = {b: cmap(i) for i, b in enumerate(bays)}

    sid_list = list(stations.keys())
    sid_idx  = {s: i for i, s in enumerate(sid_list)}
    lats = np.array([v["lat"] for v in stations.values()])
    lons = np.array([v["lon"] for v in stations.values()])
    cols = [bay_col[v["bay"]] for v in stations.values()]

    # Degree array
    order_to_sid = {o: s for o, s in
                    [(order, ctid // 12) for order, ctid in o2c.items()
                     if ctid % 12 == month and (ctid // 12) in meta]}
    orders  = list(order_to_sid.keys())
    degrees = np.array(A[orders, :].sum(axis=1)).flatten()

    # Two bays to zoom: pick largest two by station count
    bay_counts = {}
    for v in stations.values():
        bay_counts[v["bay"]] = bay_counts.get(v["bay"], 0) + 1
    top2 = sorted(bay_counts, key=bay_counts.get, reverse=True)[:2]

    fig = plt.figure(figsize=(16, 12))
    gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    ax_map  = fig.add_subplot(gs[0, 0])
    ax_deg  = fig.add_subplot(gs[0, 1])
    ax_bay1 = fig.add_subplot(gs[1, 0])
    ax_bay2 = fig.add_subplot(gs[1, 1])

    # ── map ──
    rng = np.random.default_rng(42)
    esample = edges if len(edges) < 3000 else [
        edges[i] for i in rng.choice(len(edges), 3000, replace=False)]
    for si, sj in esample:
        if si in sid_idx and sj in sid_idx:
            ax_map.plot([lons[sid_idx[si]], lons[sid_idx[sj]]],
                        [lats[sid_idx[si]], lats[sid_idx[sj]]],
                        color="gray", alpha=0.1, linewidth=0.3, zorder=1)
    ax_map.scatter(lons, lats, c=cols, s=5, alpha=0.7, zorder=2, linewidths=0)
    patches = [mpatches.Patch(color=bay_col[b], label=b) for b in bays]
    ax_map.legend(handles=patches, title="Bay", fontsize=6, title_fontsize=7,
                  loc="lower right", ncol=2, framealpha=0.9)
    ax_map.set_title(f"Florida Sampling Graph\n({len(stations):,} stations, {len(edges):,} edges)")
    ax_map.set_xlabel("Longitude"); ax_map.set_ylabel("Latitude")
    ax_map.grid(True, alpha=0.3, linewidth=0.4)

    # ── degree distribution ──
    ax_deg.hist(degrees, bins=range(0, int(degrees.max()) + 2),
                color="steelblue", edgecolor="white", alpha=0.85)
    ax_deg.set_xlabel("Degree (# neighbours)")
    ax_deg.set_ylabel("# Stations")
    ax_deg.set_title(f"Degree Distribution\n(mean={degrees.mean():.1f})")
    ax_deg.grid(axis="y", alpha=0.3)

    # ── zoomed bays ──
    for ax_bay, bay in zip([ax_bay1, ax_bay2], top2):
        bs = {sid: v for sid, v in stations.items() if v["bay"] == bay}
        sl = list(bs.keys())
        si_idx = {s: i for i, s in enumerate(sl)}
        blats = np.array([bs[s]["lat"] for s in sl])
        blons = np.array([bs[s]["lon"] for s in sl])
        local_edges = [(si, sj) for si, sj in edges
                       if si in si_idx and sj in si_idx]
        for si, sj in local_edges:
            ax_bay.plot([blons[si_idx[si]], blons[si_idx[sj]]],
                        [blats[si_idx[si]], blats[si_idx[sj]]],
                        color="steelblue", alpha=0.5, linewidth=0.8, zorder=1)
        ax_bay.scatter(blons, blats, color="darkorange", s=18, zorder=2,
                       edgecolors="black", linewidths=0.3)
        ax_bay.set_title(f"Bay: {bay}  ({len(bs)} stations, {len(local_edges)} edges)")
        ax_bay.set_xlabel("Longitude"); ax_bay.set_ylabel("Latitude")
        ax_bay.grid(True, alpha=0.3)

    fig.suptitle("FIM GNN-RNN — Graph Structure Overview", fontsize=15, y=1.01)
    path = os.path.join(out_dir, "gnn_overview.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj",     default="map/FIM_adj.pkl")
    parser.add_argument("--meta",    default="data/FIM_station_metadata.pkl")
    parser.add_argument("--out_dir", default="analysis/figures")
    parser.add_argument("--month",   type=int, default=0,
                        help="Which month slice to use for station graph (0=Jan)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading adjacency and metadata ...")
    A, o2c, meta = load_data(args.adj, args.meta)
    print(f"  Adj shape: {A.shape}, nnz: {A.nnz}")
    print(f"  Stations in metadata: {len(meta)}")

    print(f"\nBuilding station-level graph (month={args.month}) ...")
    stations, edges = build_station_graph(A, o2c, meta, month=args.month)
    print(f"  Station nodes: {len(stations)}, edges: {len(edges)}")

    print("\n=== 1. Florida overview map ===")
    plot_florida_map(stations, edges, args.out_dir)

    print("\n=== 2. Degree distribution ===")
    plot_degree_distribution(A, o2c, meta, args.out_dir, month=args.month)

    print("\n=== 3. Per-bay zoomed graphs ===")
    plot_bay_graphs(stations, edges, args.out_dir)

    print("\n=== 4. Multi-panel overview ===")
    plot_overview_panel(stations, edges, A, o2c, meta, args.out_dir, month=args.month)

    print(f"\nAll GNN visualizations saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
