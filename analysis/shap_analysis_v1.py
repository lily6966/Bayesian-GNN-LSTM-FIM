"""
shap_analysis_v1.py
-------------------
Species-specific permutation-importance analysis for **Run A** (Pinfish-as-feature,
12 management targets, gat-rnn-v2-windowed, maxepoch-30) aggregated **across all
15 walk-forward folds (test_years 2010–2024)**.

Builds on the per-fold logic in `shap_analysis.py` but iterates over every fold
and produces a 15-year mean importance map per (species × feature group).

Method:
  For each test_year T:
    1. Locate fold-specific checkpoint  (model/.../{T}/.../model-N)
    2. Load model + dist_weights
    3. Build windowed XY for T (10 sliding 3-yr windows)
    4. Run permutation importance over the 11 feature groups (env, forage,
       restoration, habitat_*, shoreline, water_effort, bycatch, na_indicators,
       community_A, community_B). For each group:
         - Shuffle that block of columns ACROSS station-month rows in X
         - Re-run inference → MSE per species
         - Importance = (perm_MSE - baseline_MSE) / baseline_MSE
       Repeat `n_repeats` times and average.
    5. Save per-fold importance.

  Aggregate across folds:
    - Mean importance per (species, group) over the 15 folds
    - Per-species top-5 predictor ranking
    - Heatmap of mean importance

Outputs (analysis/figures/):
  shap_v1_importance_long.csv         long-format per-fold per-species per-group
  shap_v1_importance_mean.csv         wide-format mean across 15 folds
  shap_v1_heatmap_15yr.png            12 species × 11 groups heatmap (means)
  shap_v1_topk_per_species.png        per-species top-5 predictor bars
  shap_v1_per_year_heatmap.png        species-mean importance × group × year

Usage (from gnn-lstm-v3/):
    python analysis/shap_analysis_v1.py
    python analysis/shap_analysis_v1.py --years 2014,2018,2022 --n_repeats 3
"""
import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE   = os.path.dirname(os.path.abspath(__file__))
_GNNDIR = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _GNNDIR)

# Import the existing shap_analysis module for its helpers
import shap_analysis as sa

RUN_ID  = os.environ.get("SHAP_RUN_ID", "run_h")
_DS     = "0412G" if RUN_ID in ("run_g", "run_h") else "0412"
_MAXEP  = 35 if RUN_ID == "run_h" else (50 if RUN_ID == "run_g" else 30)
OUT_DIR = os.path.join(_HERE, "figures", RUN_ID, "shap_v1")
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold runner — reuses helpers from shap_analysis (sa)
# ─────────────────────────────────────────────────────────────────────────────

def fold_ckpt_dir(test_year: int, max_epoch: int = _MAXEP) -> str:
    return os.path.join(
        _GNNDIR, "model", f"FIM_restoration_{_DS}_stations_monthly", str(test_year),
        f"gat-rnn-v2-windowed_bs-128_lr-0.001_maxepoch-{max_epoch}"
        f"_testyear-{test_year}_win-3_nwin-10_seed-0",
    )


def run_one_fold(test_year: int, n_repeats: int) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    """Compute permutation importance for a single fold.

    Returns (long_df, info_dict) or (None, None) if checkpoint missing.
    """
    ckpt_dir = fold_ckpt_dir(test_year)
    if not os.path.isdir(ckpt_dir):
        print(f"[skip] no ckpt dir for test_year={test_year}")
        return None, None

    # Monkey-patch the module-level constants used by sa helpers
    sa.TEST_YEAR = test_year
    sa.CKPT_DIR  = ckpt_dir
    sa.N_PERM    = n_repeats

    print(f"\n{'='*70}\n[fold] test_year={test_year}  ckpt_dir=.../{test_year}/.../{os.path.basename(ckpt_dir)[:60]}…\n{'='*70}")

    # Build args namespace
    args = sa.FIMArgs()
    args = sa.load_species_info(sa.SPP_PKL, args)

    # Data + normalisation
    (X_dict, Y_dict, adj, county_set, _county_avg,
     _year_avg_Y, min_year, max_year) = sa.load_data(args)
    args = sa.compute_normalisation(X_dict, Y_dict, county_set, test_year, args)

    # in_dim from raw X
    _s0  = county_set[0]
    _y0  = next(iter(X_dict[_s0]))
    in_dim  = X_dict[_s0][_y0].shape[-1]
    out_dim = len(args.output_names)

    # Build feature groups from args.group_slices (correct prey/nonprey split,
    # forage merged into prey_community).
    feature_groups = sa.build_feature_groups_from_args(args, in_dim)
    group_names = [n for n, _ in feature_groups]

    # Graph + nodeloader
    g, nodeloader = sa.build_graph_and_loader(adj, batch_size=128)

    # Distance weights
    import pickle
    dist_weights = None
    if os.path.exists(sa.DIST_PKL):
        with open(sa.DIST_PKL, "rb") as f:
            dist_weights = pickle.load(f)
        _arr = np.asarray(next(iter(dist_weights.values())))
        args.edge_feat_dim = int(_arr.shape[0]) if _arr.ndim >= 1 else 1
    else:
        args.edge_feat_dim = 3

    # Find checkpoint
    ckpt_path = sa.find_latest_checkpoint(ckpt_dir)
    if ckpt_path is None:
        print(f"[skip] no model-* file under {ckpt_dir}")
        return None, None

    # Load model
    model = sa.load_model(ckpt_path, in_dim, out_dim, args)

    # Windowed data for this T
    windowed_XY_T = sa.build_merged_windows(
        X_dict, Y_dict, county_set, test_year,
        win_size=sa.WIN_SIZE, n_windows=sa.N_WINDOWS,
    )

    # Permutation importance — returns (rel ΔMSE, baseline_MSE_count, abs ΔMSE_count)
    importance, baseline, delta_mse_count = sa.permutation_importance(
        model, nodeloader, windowed_XY_T, county_set,
        args, dist_weights, feature_groups, n_repeats=n_repeats,
    )
    delta_rmse_count = np.sqrt(np.maximum(delta_mse_count, 0))   # fish/obs

    rows = []
    for gi, gname in enumerate(group_names):
        for si, sp_name in enumerate(args.output_names):
            rows.append(dict(
                year=test_year, species=sp_name,
                common=sa.COMMON_NAMES.get(sp_name, sp_name),
                group=gname,
                rel_mse_increase=float(importance[gi, si]),
                delta_mse_count=float(delta_mse_count[gi, si]),
                delta_rmse_count=float(delta_rmse_count[gi, si]),
                baseline_mse_count=float(baseline[si]),
                baseline_rmse_count=float(np.sqrt(max(baseline[si], 0))),
            ))
    long_df = pd.DataFrame(rows)
    info = dict(ckpt_path=ckpt_path, baseline_mse=float(baseline.mean()),
                baseline_rmse_count=float(np.sqrt(np.maximum(baseline, 0)).mean()),
                in_dim=in_dim, out_dim=out_dim,
                group_names=group_names)
    print(f"[fold {test_year}] baseline mean MSE (count) = {info['baseline_mse']:.4f}  "
          f"(RMSE ≈ {info['baseline_rmse_count']:.3f} fish/obs); "
          f"top group: {long_df.groupby('group')['rel_mse_increase'].mean().idxmax()}")
    return long_df, info


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation + plots
# ─────────────────────────────────────────────────────────────────────────────

def _save_long(long: pd.DataFrame) -> str:
    p = os.path.join(OUT_DIR, "shap_v1_importance_long.csv")
    long.to_csv(p, index=False)
    return p


def _build_mean_wide(long: pd.DataFrame) -> pd.DataFrame:
    return (long.groupby(["common", "group"])["rel_mse_increase"]
            .mean().reset_index()
            .pivot(index="common", columns="group", values="rel_mse_increase"))


def _plot_mean_heatmap(M: pd.DataFrame, out_path: str, n_folds: int) -> None:
    sp_order  = M.mean(axis=1).sort_values(ascending=False).index.tolist()
    grp_order = M.mean(axis=0).sort_values(ascending=False).index.tolist()
    M = M.loc[sp_order, grp_order]
    vmax = max(0.05, np.nanpercentile(np.abs(M.values), 95))

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(M.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(M.index, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            color = "white" if abs(v) > vmax * 0.6 else "black"
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=7, color=color)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Rel. MSE increase (mean across folds)", fontsize=10)
    ax.set_title(f"Run A — Predictor importance per species "
                 f"(mean of permutation importance across {n_folds} folds)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_topk(M: pd.DataFrame, out_path: str, k: int = 5) -> None:
    species = M.mean(axis=1).sort_values(ascending=False).index.tolist()
    n_sp = len(species)
    n_cols = 3
    n_rows = int(np.ceil(n_sp / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.6 * n_rows))
    axes = axes.flatten()
    for i, sp in enumerate(species):
        ax  = axes[i]
        row = M.loc[sp].sort_values(ascending=False)
        top = row.head(k)
        colors = ["#2E86AB" if v > 0 else "#E63946" for v in top.values]
        ax.barh(range(len(top)), top.values, color=colors, edgecolor="none")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.5)
        ax.set_title(sp, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)
        for kk, v in enumerate(top.values):
            ax.text(v + 0.001, kk, f"{v:+.3f}", va="center", fontsize=7,
                    ha="left" if v >= 0 else "right", color="black")
    for j in range(n_sp, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Top-{k} predictor groups per species — Run A — mean across folds",
                 fontsize=12, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_per_year(long: pd.DataFrame, out_path: str) -> None:
    """Heatmap: species-averaged importance per group per year — shows
    how the predictor relevance shifts over time."""
    M = (long.groupby(["year", "group"])["rel_mse_increase"]
         .mean().reset_index()
         .pivot(index="year", columns="group", values="rel_mse_increase"))
    grp_order = M.mean(axis=0).sort_values(ascending=False).index.tolist()
    M = M[grp_order]
    vmax = max(0.05, np.nanpercentile(np.abs(M.values), 95))

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(M.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels([str(y) for y in M.index], fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            color = "white" if abs(v) > vmax * 0.6 else "black"
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=7, color=color)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Mean rel. MSE increase (across 12 species)", fontsize=10)
    ax.set_title("Run A — Predictor importance over time "
                 "(species-averaged group importance per year)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Feature group")
    ax.set_ylabel("Test year")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=str, default="2010-2024",
                   help="Year range, e.g. '2010-2024' or '2014,2018,2022'")
    p.add_argument("--n_repeats", type=int, default=3,
                   help="Permutation repeats per fold (default 3 — keeps total runtime ~30 min). "
                        "Higher → more stable but slower.")
    cli = p.parse_args()

    if "-" in cli.years and "," not in cli.years:
        a, b = map(int, cli.years.split("-"))
        years = list(range(a, b + 1))
    else:
        years = [int(y) for y in cli.years.split(",")]

    print(f"[shap_v1] Years: {years}")
    print(f"[shap_v1] Permutation repeats per fold: {cli.n_repeats}")
    print(f"[shap_v1] Total folds × repeats: {len(years) * cli.n_repeats}")

    all_long = []
    info_per_fold = {}
    for ty in years:
        long_df, info = run_one_fold(ty, n_repeats=cli.n_repeats)
        if long_df is None:
            continue
        all_long.append(long_df)
        info_per_fold[ty] = info
        # Snapshot intermediate
        _save_long(pd.concat(all_long, ignore_index=True))

    if not all_long:
        print("[err] no folds completed — nothing to aggregate")
        return

    long = pd.concat(all_long, ignore_index=True)
    long_csv = _save_long(long)
    print(f"\nSaved long-format CSV: {long_csv}  ({len(long)} rows)")

    M = _build_mean_wide(long)
    mean_csv = os.path.join(OUT_DIR, "shap_v1_importance_mean.csv")
    M.to_csv(mean_csv)
    print(f"Saved mean wide CSV: {mean_csv}  shape={M.shape}")

    # Count-scale ΔRMSE wide CSV (mean across folds, by species × group)
    drmse_wide = (long.groupby(["common", "group"])["delta_rmse_count"]
                       .mean().reset_index()
                       .pivot(index="common", columns="group",
                              values="delta_rmse_count"))
    drmse_csv = os.path.join(OUT_DIR, "shap_v1_dRMSE_count_mean.csv")
    drmse_wide.to_csv(drmse_csv)
    print(f"Saved count-scale ΔRMSE wide CSV: {drmse_csv}")

    dmse_wide = (long.groupby(["common", "group"])["delta_mse_count"]
                       .mean().reset_index()
                       .pivot(index="common", columns="group",
                              values="delta_mse_count"))
    dmse_csv = os.path.join(OUT_DIR, "shap_v1_dMSE_count_mean.csv")
    dmse_wide.to_csv(dmse_csv)
    print(f"Saved count-scale ΔMSE wide CSV: {dmse_csv}")

    n_folds = long["year"].nunique()
    _plot_mean_heatmap(M, os.path.join(OUT_DIR, "shap_v1_heatmap_15yr.png"), n_folds)
    _plot_topk        (M, os.path.join(OUT_DIR, "shap_v1_topk_per_species.png"), k=5)
    _plot_per_year    (long, os.path.join(OUT_DIR, "shap_v1_per_year_heatmap.png"))
    print(f"Saved 3 figures to {OUT_DIR}/")

    # ── Console digest ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Run A — Predictor importance summary  (mean across {n_folds} folds)")
    print(f"{'='*70}")
    overall = M.mean(axis=0).sort_values(ascending=False)
    print("\nGroup ranking — RELATIVE ΔMSE (mean across 12 species):")
    for g, v in overall.items():
        bar = "█" * max(1, int(v * 200))
        print(f"  {g:<22s}  {v:+.4f}  {bar}")

    overall_d = drmse_wide.mean(axis=0).sort_values(ascending=False)
    print("\nGroup ranking — ABSOLUTE ΔRMSE (count scale, fish/obs, mean across species):")
    for g, v in overall_d.items():
        bar = "█" * max(1, int(v * 30 / (overall_d.max() + 1e-9)))
        print(f"  {g:<22s}  {v:+.4f} fish  {bar}")

    print(f"\nPer-species baseline RMSE (count scale, mean across folds):")
    base_rmse = (long.groupby("common")["baseline_rmse_count"].mean()
                       .sort_values(ascending=False))
    for sp, v in base_rmse.items():
        print(f"  {sp:<28s}  {v:6.3f} fish/obs")

    print(f"\nPer-species top-3 predictor groups (relative ΔMSE):")
    for sp in M.index:
        top3 = M.loc[sp].sort_values(ascending=False).head(3)
        items = "  ".join([f"{g}({v:+.3f})" for g, v in top3.items()])
        print(f"  {sp:<28s}  {items}")


if __name__ == "__main__":
    main()
