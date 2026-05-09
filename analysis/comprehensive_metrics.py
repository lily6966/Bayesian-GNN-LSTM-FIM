"""
comprehensive_metrics.py
========================
Loads all rolling-CV test_results.csv files and computes:
  1. Overall summary (R², RMSE, Corr, MAE, Bias) across 21 windows
  2. Per-species metrics (mean corr, RMSE, bias across all years)
  3. Per-month metrics  (decode month from node_id = fips % 12 + 1)
  4. Per-bay metrics    (station_id = fips // 12 → lookup bay)
  5. Temporal trend     (is performance improving over years?)
  6. Observed seasonal trends (monthly means of TRUE values per species)
  7. Monthly trend lines for observed data
All figures saved to analysis/figures/comprehensive/
"""

import os, glob, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
from scipy.stats import pearsonr

# ── Paths ──────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS  = os.path.join(BASE, "gnn-rnn", "results", "FIM_stations_monthly")
META_PKL = os.path.join(BASE, "data", "FIM_station_metadata.pkl")
OUT_DIR  = os.path.join(BASE, "analysis", "figures", "comprehensive")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "observed_monthly"), exist_ok=True)

MODEL_SUFFIX = ("gnn-rnn_bs-128_lr-0.001_maxepoch-3_sche-cosine_T0-50_etamin-1e-05"
                "_step-50_gamma-0.5_dropout-0.5_sleep-50_testyear-{yr}_aggregator-mean"
                "_encoder-mlp_trainweekstart-52_len-5_weightdecay-1e-05_seed-0")

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Load station metadata ──────────────────────────────────────────────────
with open(META_PKL, "rb") as f:
    station_meta = pickle.load(f)   # {station_id: {"bay": ..., ...}}

def node_to_bay(node_id):
    sid = int(node_id) // 12
    info = station_meta.get(sid, {})
    return info.get("bay", "Unknown") if isinstance(info, dict) else "Unknown"

def node_to_month(node_id):
    return int(node_id) % 12 + 1   # 1-12

# ── Collect all test CSV paths ─────────────────────────────────────────────
years = list(range(2004, 2025))
all_dfs = []
for yr in years:
    model_dir = os.path.join(RESULTS, str(yr), MODEL_SUFFIX.format(yr=yr))
    csv_path  = os.path.join(model_dir, "test_results.csv")
    if not os.path.exists(csv_path):
        print(f"  MISSING: {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    df["test_year"] = yr
    df["month"]     = df["fips"].apply(node_to_month)
    df["bay"]       = df["fips"].apply(node_to_bay)
    all_dfs.append(df)
    print(f"  Loaded {yr}: {len(df)} rows")

df_all = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal rows: {len(df_all)}")

# ── Identify species columns ───────────────────────────────────────────────
pred_cols = [c for c in df_all.columns if c.startswith("predicted_")]
true_cols = [c.replace("predicted_", "true_") for c in pred_cols]
species   = [c.replace("predicted_", "") for c in pred_cols]
n_sp      = len(species)
print(f"Species tracked: {n_sp}")

# ────────────────────────────────────────────────────────────────────────────
# 1. OVERALL ROLLING SUMMARY
# ────────────────────────────────────────────────────────────────────────────
rolling_csv = os.path.join(RESULTS, "rolling", "rolling_results.csv")
roll = pd.read_csv(rolling_csv)
print("\n=== Overall Rolling Summary ===")
stats_summary = {}
for col in ["test_rmse", "test_r2", "test_corr"]:
    stats_summary[col] = {
        "mean":   roll[col].mean(),
        "median": roll[col].median(),
        "std":    roll[col].std(),
        "min":    roll[col].min(),
        "max":    roll[col].max(),
    }
    print(f"  {col}: mean={stats_summary[col]['mean']:.4f}  "
          f"median={stats_summary[col]['median']:.4f}  "
          f"std={stats_summary[col]['std']:.4f}  "
          f"range=[{stats_summary[col]['min']:.4f}, {stats_summary[col]['max']:.4f}]")

# ────────────────────────────────────────────────────────────────────────────
# 2. PER-SPECIES METRICS
# ────────────────────────────────────────────────────────────────────────────
sp_metrics = []
for sp, pc, tc in zip(species, pred_cols, true_cols):
    pred = df_all[pc].values
    true = df_all[tc].values
    mask = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[mask], true[mask]
    if len(pred) < 10:
        continue
    corr_val = pearsonr(pred, true)[0] if np.std(pred) > 0 and np.std(true) > 0 else 0.0
    ss_tot = np.sum((true - true.mean())**2)
    ss_res = np.sum((true - pred)**2)
    r2_val  = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse_val = np.sqrt(np.mean((pred - true)**2))
    mae_val  = np.mean(np.abs(pred - true))
    bias_val = np.mean(pred - true)
    sp_metrics.append({
        "species": sp, "corr": corr_val, "r2": r2_val,
        "rmse": rmse_val, "mae": mae_val, "bias": bias_val, "n": len(pred)
    })

sp_df = pd.DataFrame(sp_metrics).sort_values("corr", ascending=False)
sp_df.to_csv(os.path.join(OUT_DIR, "species_metrics.csv"), index=False)
print("\n=== Top-10 Species by Correlation ===")
print(sp_df.head(10).to_string(index=False))
print("\n=== Bottom-10 Species by Correlation ===")
print(sp_df.tail(10).to_string(index=False))

# Plot species metrics
fig, axes = plt.subplots(1, 3, figsize=(18, max(6, n_sp * 0.35)))
sp_sorted = sp_df.sort_values("corr", ascending=True)
colors = ["#e74c3c" if c < 0 else "#2ecc71" if c > 0.4 else "#f39c12"
          for c in sp_sorted["corr"]]
axes[0].barh(sp_sorted["species"], sp_sorted["corr"], color=colors)
axes[0].axvline(0, color="black", linewidth=0.8)
axes[0].set_xlabel("Pearson Correlation"); axes[0].set_title("Per-Species Correlation")
axes[0].set_xlim(-0.3, 1.0)

axes[1].barh(sp_sorted["species"], sp_sorted["rmse"], color="#3498db")
axes[1].set_xlabel("RMSE (count scale)"); axes[1].set_title("Per-Species RMSE")

bias_colors = ["#e74c3c" if b > 0 else "#2980b9" for b in sp_sorted["bias"]]
axes[2].barh(sp_sorted["species"], sp_sorted["bias"], color=bias_colors)
axes[2].axvline(0, color="black", linewidth=0.8)
axes[2].set_xlabel("Mean Bias (pred − true)"); axes[2].set_title("Per-Species Bias")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "species_metrics.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved species_metrics.png")

# ────────────────────────────────────────────────────────────────────────────
# 3. PER-MONTH METRICS
# ────────────────────────────────────────────────────────────────────────────
month_metrics = []
for m in range(1, 13):
    sub = df_all[df_all["month"] == m]
    m_corrs, m_rmses, m_biases = [], [], []
    for pc, tc in zip(pred_cols, true_cols):
        pred = sub[pc].values; true = sub[tc].values
        mask = np.isfinite(pred) & np.isfinite(true)
        pred, true = pred[mask], true[mask]
        if len(pred) < 5: continue
        if np.std(pred) > 0 and np.std(true) > 0:
            m_corrs.append(pearsonr(pred, true)[0])
        m_rmses.append(np.sqrt(np.mean((pred - true)**2)))
        m_biases.append(np.mean(pred - true))
    month_metrics.append({
        "month": m, "month_name": MONTH_NAMES[m-1],
        "mean_corr": np.mean(m_corrs) if m_corrs else 0,
        "mean_rmse": np.mean(m_rmses) if m_rmses else 0,
        "mean_bias": np.mean(m_biases) if m_biases else 0,
        "n_species":  len(m_corrs)
    })

month_df = pd.DataFrame(month_metrics)
month_df.to_csv(os.path.join(OUT_DIR, "monthly_metrics.csv"), index=False)
print("\n=== Per-Month Metrics ===")
print(month_df[["month_name","mean_corr","mean_rmse","mean_bias"]].to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].bar(month_df["month_name"], month_df["mean_corr"],
            color=cm.RdYlGn(month_df["mean_corr"] / month_df["mean_corr"].max()))
axes[0].set_title("Mean Correlation by Month"); axes[0].set_ylabel("Mean Corr")
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(month_df["month_name"], month_df["mean_rmse"], color="#3498db")
axes[1].set_title("Mean RMSE by Month"); axes[1].set_ylabel("RMSE")
axes[1].tick_params(axis='x', rotation=45)

bias_colors = ["#e74c3c" if b > 0 else "#2980b9" for b in month_df["mean_bias"]]
axes[2].bar(month_df["month_name"], month_df["mean_bias"], color=bias_colors)
axes[2].axhline(0, color="black", linewidth=0.8)
axes[2].set_title("Mean Bias by Month"); axes[2].set_ylabel("Bias (pred − true)")
axes[2].tick_params(axis='x', rotation=45)

plt.suptitle("Model Performance by Calendar Month (all species, 2004–2024)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "monthly_metrics.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved monthly_metrics.png")

# ────────────────────────────────────────────────────────────────────────────
# 4. PER-BAY METRICS
# ────────────────────────────────────────────────────────────────────────────
bay_metrics = []
for bay in sorted(df_all["bay"].unique()):
    sub = df_all[df_all["bay"] == bay]
    b_corrs, b_rmses, b_biases = [], [], []
    for pc, tc in zip(pred_cols, true_cols):
        pred = sub[pc].values; true = sub[tc].values
        mask = np.isfinite(pred) & np.isfinite(true)
        pred, true = pred[mask], true[mask]
        if len(pred) < 5: continue
        if np.std(pred) > 0 and np.std(true) > 0:
            b_corrs.append(pearsonr(pred, true)[0])
        b_rmses.append(np.sqrt(np.mean((pred - true)**2)))
        b_biases.append(np.mean(pred - true))
    bay_metrics.append({
        "bay": bay, "n_rows": len(sub),
        "mean_corr": np.mean(b_corrs) if b_corrs else 0,
        "mean_rmse": np.mean(b_rmses) if b_rmses else 0,
        "mean_bias": np.mean(b_biases) if b_biases else 0
    })

bay_df = pd.DataFrame(bay_metrics).sort_values("mean_corr", ascending=False)
bay_df.to_csv(os.path.join(OUT_DIR, "bay_metrics.csv"), index=False)
print("\n=== Per-Bay Metrics ===")
print(bay_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
bays_sorted = bay_df.sort_values("mean_corr", ascending=True)
corr_vals = bays_sorted["mean_corr"].values
axes[0].barh(bays_sorted["bay"], corr_vals,
             color=["#e74c3c" if c < 0 else "#2ecc71" if c > 0.4 else "#f39c12" for c in corr_vals])
axes[0].axvline(0, color="black", linewidth=0.8)
axes[0].set_title("Mean Correlation by Bay"); axes[0].set_xlabel("Mean Corr")

axes[1].barh(bays_sorted["bay"], bays_sorted["mean_rmse"], color="#3498db")
axes[1].set_title("Mean RMSE by Bay"); axes[1].set_xlabel("RMSE")

bias_colors = ["#e74c3c" if b > 0 else "#2980b9" for b in bays_sorted["mean_bias"]]
axes[2].barh(bays_sorted["bay"], bays_sorted["mean_bias"], color=bias_colors)
axes[2].axvline(0, color="black", linewidth=0.8)
axes[2].set_title("Mean Bias by Bay"); axes[2].set_xlabel("Bias (pred − true)")

plt.suptitle("Model Performance by Bay (all species, 2004–2024)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "bay_metrics.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved bay_metrics.png")

# ────────────────────────────────────────────────────────────────────────────
# 5. TEMPORAL TREND: performance over rolling test years
# ────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
roll_sorted = roll.sort_values("test_year")
for ax, col, label, color in zip(
        axes,
        ["test_corr", "test_r2", "test_rmse"],
        ["Pearson Correlation", "R²", "RMSE (count scale)"],
        ["#2ecc71", "#3498db", "#e74c3c"]):
    ax.plot(roll_sorted["test_year"], roll_sorted[col],
            "o-", color=color, linewidth=2, markersize=7)
    # rolling 5-year trend line
    if len(roll_sorted) >= 5:
        slope, intercept, r, p, _ = stats.linregress(
            roll_sorted["test_year"], roll_sorted[col])
        ax.plot(roll_sorted["test_year"],
                slope * roll_sorted["test_year"] + intercept,
                "--", color="gray", linewidth=1.5,
                label=f"trend slope={slope:.4f}/yr  p={p:.3f}")
        ax.legend(fontsize=9)
    ax.set_ylabel(label, fontsize=11)
    ax.axhline(roll_sorted[col].mean(), color=color, linewidth=1, linestyle=":")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Test Year", fontsize=12)
fig.suptitle("GNN-RNN Rolling CV Performance Over Time (2004–2024)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "temporal_trend.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved temporal_trend.png")

# ────────────────────────────────────────────────────────────────────────────
# 6. OBSERVED SEASONAL TRENDS (monthly means of true values per species)
# ────────────────────────────────────────────────────────────────────────────
obs_monthly = []
for m in range(1, 13):
    sub = df_all[df_all["month"] == m]
    row = {"month": m, "month_name": MONTH_NAMES[m-1]}
    for sp, tc in zip(species, true_cols):
        row[sp] = sub[tc].mean()
    obs_monthly.append(row)

obs_df = pd.DataFrame(obs_monthly)
obs_df.to_csv(os.path.join(OUT_DIR, "observed_monthly", "observed_monthly_means.csv"),
              index=False)

# Heatmap: species × month
obs_mat = obs_df[species].values.T    # (n_sp, 12)
# row-normalise per species for visibility
row_max = obs_mat.max(axis=1, keepdims=True)
row_max[row_max == 0] = 1
obs_norm = obs_mat / row_max

fig, ax = plt.subplots(figsize=(14, max(6, n_sp * 0.4)))
im = ax.imshow(obs_norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
ax.set_xticks(range(12)); ax.set_xticklabels(MONTH_NAMES, fontsize=10)
ax.set_yticks(range(n_sp)); ax.set_yticklabels(species, fontsize=8)
plt.colorbar(im, ax=ax, label="Normalised mean count (per species)")
ax.set_title("Observed Mean Monthly Catch by Species\n(row-normalised, 2004–2024 test predictions)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "observed_monthly", "observed_seasonal_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved observed_seasonal_heatmap.png")

# ────────────────────────────────────────────────────────────────────────────
# 7. MONTHLY TREND LINES: observed true values per species per month over years
# ────────────────────────────────────────────────────────────────────────────
# For each species: line plot of annual mean (aggregated across stations) per month
os.makedirs(os.path.join(OUT_DIR, "observed_monthly", "species_trends"), exist_ok=True)

# Aggregate: mean true count per (test_year, month) per species
yr_mo_means = df_all.groupby(["test_year", "month"])[true_cols].mean().reset_index()
yr_mo_means.columns = ["test_year", "month"] + species

for sp in species:
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharey=False)
    axes = axes.flatten()
    for mi, (m, ax) in enumerate(zip(range(1, 13), axes)):
        sub = yr_mo_means[yr_mo_means["month"] == m].sort_values("test_year")
        if len(sub) < 3:
            ax.set_visible(False); continue
        ax.plot(sub["test_year"], sub[sp], "o-", color="#e67e22", linewidth=1.5, markersize=4)
        if len(sub) >= 4:
            slope, intercept, r, p, _ = stats.linregress(sub["test_year"], sub[sp])
            ax.plot(sub["test_year"], slope * sub["test_year"] + intercept,
                    "--", color="#2c3e50", linewidth=1.2)
            ax.set_title(f"{MONTH_NAMES[m-1]}  (slope={slope:.3f}/yr, p={p:.2f})",
                         fontsize=9)
        else:
            ax.set_title(MONTH_NAMES[m-1], fontsize=9)
        ax.set_xlabel("Year", fontsize=7); ax.set_ylabel("Mean count", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Observed monthly trends — {sp}", fontsize=13)
    plt.tight_layout()
    safe_name = sp.replace("/", "_").replace(" ", "_")
    plt.savefig(os.path.join(OUT_DIR, "observed_monthly", "species_trends",
                             f"{safe_name}_monthly_trends.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

print(f"Saved {n_sp} species monthly trend plots.")

# ────────────────────────────────────────────────────────────────────────────
# 8. SUMMARY TABLE: all metrics combined
# ────────────────────────────────────────────────────────────────────────────
summary_lines = [
    "=" * 65,
    "GNN-RNN Rolling CV — Comprehensive Metrics Summary",
    f"Model: maxepoch-3, len-5, month-weighted loss",
    f"Test years: 2004–2024  ({len(all_dfs)} windows loaded)",
    "=" * 65,
    "",
    "── Overall ──────────────────────────────────────────",
    f"  Mean Corr   : {roll['test_corr'].mean():.4f}  (σ={roll['test_corr'].std():.4f})",
    f"  Mean R²     : {roll['test_r2'].mean():.4f}  (σ={roll['test_r2'].std():.4f})",
    f"  Mean RMSE   : {roll['test_rmse'].mean():.4f}  (σ={roll['test_rmse'].std():.4f})",
    f"  Best year   : {roll.loc[roll['test_corr'].idxmax(), 'test_year']}  (corr={roll['test_corr'].max():.4f})",
    f"  Worst year  : {roll.loc[roll['test_corr'].idxmin(), 'test_year']}  (corr={roll['test_corr'].min():.4f})",
    "",
    "── Per-Species (mean across 21 windows) ─────────────",
    f"  Best species  : {sp_df.iloc[0]['species']}  (corr={sp_df.iloc[0]['corr']:.4f})",
    f"  Worst species : {sp_df.iloc[-1]['species']}  (corr={sp_df.iloc[-1]['corr']:.4f})",
    f"  Species with corr>0.4: {(sp_df['corr']>0.4).sum()}/{len(sp_df)}",
    f"  Species with corr<0.0: {(sp_df['corr']<0.0).sum()}/{len(sp_df)}",
    "",
    "── Per-Month (mean across species & years) ──────────",
]
for _, row in month_df.iterrows():
    summary_lines.append(
        f"  {row['month_name']:4s}: corr={row['mean_corr']:.4f}  "
        f"rmse={row['mean_rmse']:.3f}  bias={row['mean_bias']:.3f}")
summary_lines += [
    "",
    "── Per-Bay (mean across species & years) ────────────",
]
for _, row in bay_df.iterrows():
    summary_lines.append(
        f"  {row['bay']:6s}: corr={row['mean_corr']:.4f}  "
        f"rmse={row['mean_rmse']:.3f}  bias={row['mean_bias']:.3f}  n={row['n_rows']:,}")
summary_lines.append("=" * 65)

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(os.path.join(OUT_DIR, "metrics_summary.txt"), "w") as f:
    f.write(summary_text)
print(f"\nAll outputs saved to {OUT_DIR}/")
