"""
FIM GNN-RNN — Model Evaluation
================================
Computes per-species R², RMSE, MAE, and Pearson correlation on
train / val / test splits (on COUNT scale, i.e. expm1 of log1p predictions),
then generates:

  - 06_metrics_table.csv            per-species metrics across all splits
  - 06_r2_bar_chart.png             R² bar chart (test), adult vs juvenile grouped
  - 06_scatter_{species}.png        predicted vs true scatter for each species
  - 06_overall_summary.txt          plain-text summary

NOTE: All metrics are computed on the raw fish count scale (expm1 applied to
log1p-transformed model inputs/outputs).

Usage:
    python analysis/model_evaluation.py \
        --results_dir  gnn-rnn/results/FIM_stations_monthly/2024/<run>/ \
        --out_dir      analysis/figures
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── helpers ──────────────────────────────────────────────────────────────────

def build_path(path):
    os.makedirs(path, exist_ok=True)


def compute_metrics(y_true, y_pred):
    """Compute metrics on COUNT scale (expm1 applied to log1p inputs)."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return dict(r2=np.nan, rmse=np.nan, mae=np.nan, corr=np.nan, n=0)
    # Convert from log1p space to count scale
    y_true = np.expm1(y_true)
    y_pred = np.clip(np.expm1(y_pred), 0, None)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 1e-9 else 0.0
    return dict(r2=round(r2, 4), rmse=round(rmse, 4),
                mae=round(mae, 4), corr=round(corr, 4), n=int(mask.sum()))


def load_split(results_dir, split):
    path = os.path.join(results_dir, f"{split}_results.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ── analysis 1: metrics table ─────────────────────────────────────────────────

def build_metrics_table(results_dir, species):
    rows = []
    for split in ["train", "val", "test"]:
        df = load_split(results_dir, split)
        if df is None:
            continue
        for s in species:
            m = compute_metrics(
                df[f"true_{s}"].values,
                df[f"predicted_{s}"].values
            )
            rows.append({"split": split, "species": s, **m})
    return pd.DataFrame(rows)


# ── analysis 2: R² bar chart ──────────────────────────────────────────────────

def plot_r2_bar(metrics_df, out_dir):
    test = metrics_df[metrics_df["split"] == "test"].copy()
    adults   = [s for s in test["species"] if s.endswith("_a")]
    juveniles = [s for s in test["species"] if s.endswith("_j")]

    base_names = sorted(set(s[:-2] for s in test["species"]))

    adult_r2   = [test.loc[test["species"] == s, "r2"].values[0] for s in adults]
    juv_r2     = [test.loc[test["species"] == s, "r2"].values[0] for s in juveniles]

    # Pair adult and juvenile by base name
    paired = []
    for b in base_names:
        a_val = test.loc[test["species"] == f"{b}_a", "r2"]
        j_val = test.loc[test["species"] == f"{b}_j", "r2"]
        if not a_val.empty and not j_val.empty:
            paired.append((b, float(a_val.values[0]), float(j_val.values[0])))

    paired.sort(key=lambda x: max(x[1], x[2]), reverse=True)
    labels = [p[0].replace("_", " ") for p in paired]
    a_vals = [p[1] for p in paired]
    j_vals = [p[2] for p in paired]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    bars_a = ax.bar(x - width/2, a_vals, width, label="Adult",    color="steelblue",  alpha=0.85)
    bars_j = ax.bar(x + width/2, j_vals, width, label="Juvenile", color="darkorange", alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("R²  (count scale, test year 2024)")
    ax.set_title("Per-Species R²: Adult vs Juvenile — Test Set (2024, count scale)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "06_r2_bar_chart.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── analysis 3: scatter plots ─────────────────────────────────────────────────

def plot_scatters(results_dir, species, out_dir):
    scatter_dir = os.path.join(out_dir, "scatter_plots")
    build_path(scatter_dir)

    test = load_split(results_dir, "test")
    if test is None:
        print("  No test_results.csv found — skipping scatter plots.")
        return

    # One big 3×6 grid (all 18 species)
    adults    = sorted([s for s in species if s.endswith("_a")])
    juveniles = sorted([s for s in species if s.endswith("_j")])
    ordered   = []
    for a in adults:
        ordered.append(a)
        j = a[:-2] + "_j"
        if j in juveniles:
            ordered.append(j)

    ncols = 6
    nrows = int(np.ceil(len(ordered) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    fig.suptitle("Predicted vs True — Test Set (2024, count scale)", fontsize=14, y=1.01)
    axes = axes.flat

    for ax, s in zip(axes, ordered):
        y_true = test[f"true_{s}"].values
        y_pred = test[f"predicted_{s}"].values
        mask   = ~(np.isnan(y_true) | np.isnan(y_pred))
        yt, yp = y_true[mask], y_pred[mask]
        # Convert to count scale for plotting
        yt_cnt = np.expm1(yt)
        yp_cnt = np.clip(np.expm1(yp), 0, None)

        ax.scatter(yt_cnt, yp_cnt, alpha=0.15, s=4, color="steelblue" if s.endswith("_a") else "darkorange")

        # 1:1 line
        lim = [min(yt_cnt.min(), yp_cnt.min()), max(yt_cnt.max(), yp_cnt.max())]
        ax.plot(lim, lim, "k--", linewidth=0.8, alpha=0.6)

        m = compute_metrics(yt, yp)
        ax.set_title(f"{s.replace('_',' ')}\nR²={m['r2']:.3f}  corr={m['corr']:.3f}", fontsize=8)
        ax.set_xlabel("True count", fontsize=7)
        ax.set_ylabel("Predicted count", fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused axes
    for ax in list(axes)[len(ordered):]:
        ax.set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, "06_scatter_all_species.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── analysis 4: train / val / test R² comparison ─────────────────────────────

def plot_split_comparison(metrics_df, out_dir):
    pivoted = metrics_df.pivot_table(index="species", columns="split", values="r2")
    pivoted = pivoted.reindex(columns=["train", "val", "test"])
    pivoted = pivoted.sort_values("test", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(pivoted))
    w = 0.26
    colours = {"train": "steelblue", "val": "darkorange", "test": "seagreen"}
    for i, split in enumerate(["train", "val", "test"]):
        if split in pivoted.columns:
            ax.bar(x + (i-1)*w, pivoted[split].values, w,
                   label=split.capitalize(), color=colours[split], alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in pivoted.index],
                       rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title("R² by Species and Split (Train / Val / Test)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "06_r2_train_val_test.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── analysis 5: plain-text summary ───────────────────────────────────────────

def write_summary(metrics_df, out_dir):
    lines = []
    lines.append("=" * 70)
    lines.append("FIM GNN-RNN — MODEL EVALUATION SUMMARY  (count scale)")
    lines.append("=" * 70)

    for split in ["train", "val", "test"]:
        sub = metrics_df[metrics_df["split"] == split]
        if sub.empty:
            continue
        lines.append(f"\n{'─'*70}")
        lines.append(f"  SPLIT: {split.upper()}")
        lines.append(f"{'─'*70}")
        lines.append(f"  {'Species':<28} {'R²':>7} {'RMSE':>8} {'MAE':>8} {'Corr':>8}")
        lines.append(f"  {'-'*28} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")
        for _, row in sub.sort_values("r2", ascending=False).iterrows():
            lines.append(
                f"  {row['species']:<28} {row['r2']:>7.4f} {row['rmse']:>8.4f} "
                f"{row['mae']:>8.4f} {row['corr']:>8.4f}"
            )
        # Overall (macro average, ignoring NaN)
        lines.append(f"  {'─'*60}")
        lines.append(
            f"  {'MACRO AVERAGE':<28} "
            f"{sub['r2'].mean():>7.4f} {sub['rmse'].mean():>8.4f} "
            f"{sub['mae'].mean():>8.4f} {sub['corr'].mean():>8.4f}"
        )
        n_pos = (sub["r2"] > 0).sum()
        lines.append(f"  Species with R² > 0:  {n_pos} / {len(sub)}")

    lines.append("\n" + "=" * 70)
    text = "\n".join(lines)
    print(text)
    path = os.path.join(out_dir, "06_overall_summary.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"\n  Saved: {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FIM model evaluation")
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing train/val/test_results.csv")
    parser.add_argument("--out_dir", default="analysis/figures",
                        help="Output directory for figures and CSVs")
    args = parser.parse_args()

    build_path(args.out_dir)

    # Detect species from test CSV
    test = load_split(args.results_dir, "test")
    if test is None:
        raise FileNotFoundError(f"test_results.csv not found in {args.results_dir}")
    species = [c.replace("predicted_", "") for c in test.columns if c.startswith("predicted_")]
    print(f"Found {len(species)} target species: {species}\n")

    print("=== Computing metrics ===")
    metrics_df = build_metrics_table(args.results_dir, species)
    csv_path = os.path.join(args.out_dir, "06_metrics_table.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    print("\n=== R² bar chart (test) ===")
    plot_r2_bar(metrics_df, args.out_dir)

    print("\n=== Train / Val / Test R² comparison ===")
    plot_split_comparison(metrics_df, args.out_dir)

    print("\n=== Scatter plots (test) ===")
    plot_scatters(args.results_dir, species, args.out_dir)

    print("\n=== Overall summary ===")
    write_summary(metrics_df, args.out_dir)


if __name__ == "__main__":
    main()
