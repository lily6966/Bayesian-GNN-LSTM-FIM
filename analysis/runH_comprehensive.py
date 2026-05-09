"""
runH_comprehensive.py
=====================
Comprehensive analysis of Run H rolling-windowed CV
(FIM_restoration_0412G, 12 species/life-stages, max_epoch=35, test_years 2010-2024).

Outputs (analysis/figures/runH_comprehensive/):
    1. overall_summary.csv          per-fold + grand summary
    2. per_species_metrics.csv/png  R², RMSE, corr, bias by species
    3. per_month_metrics.csv/png    seasonal performance
    4. per_bay_metrics.csv/png      spatial performance
    5. temporal_trend.png           is performance changing over years?
    6. pred_vs_obs_overall.png      hexbin scatter
    7. species_pred_vs_obs.png      grid of per-species scatter
"""
import os, glob, pickle, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

BASE     = "/Users/liyingnceas/GitHub/GNN-RNN-main"
RES_ROOT = f"{BASE}/gnn-lstm-v4/results/FIM_restoration_0412G_stations_monthly"
META_PKL = f"{BASE}/data/FIM_restoration_0412G_station_metadata.pkl"
OUT      = f"{BASE}/gnn-lstm-v4/analysis/figures/runH_comprehensive"
os.makedirs(OUT, exist_ok=True)

MODEL_DIR_TAG = "maxepoch-35"   # selects Run H folder over the maxepoch-50 (Run G)
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Load metadata ─────────────────────────────────────────────────────────
meta = pickle.load(open(META_PKL, "rb"))
# NOTE: `fips` column in test_results.csv is actually station_id (0..595),
# not node_id (0..7151). Build station_id → Bay.
stid_to_bay = (meta.drop_duplicates("station_id")
                   .set_index("station_id")["Bay"].to_dict())

# ── Locate Run H fold result CSVs ─────────────────────────────────────────
fold_csvs = []
for yr_dir in sorted(glob.glob(f"{RES_ROOT}/2*/")):
    yr = int(os.path.basename(yr_dir.rstrip("/")))
    matches = glob.glob(f"{yr_dir}gat-rnn-v2-windowed_*{MODEL_DIR_TAG}*/test_results.csv")
    if matches:
        fold_csvs.append((yr, matches[0]))
print(f"Found {len(fold_csvs)} Run-H fold CSVs")

# ── Stack all folds ───────────────────────────────────────────────────────
dfs = []
for yr, p in fold_csvs:
    d = pd.read_csv(p)
    d["test_year"] = yr
    dfs.append(d)
all_df = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(all_df):,}")

species_cols = [c.replace("predicted_", "") for c in all_df.columns if c.startswith("predicted_")]
print(f"Species: {len(species_cols)} → {species_cols}")

# Add bay/month from fips (which == node_id here)
all_df["bay"] = all_df["fips"].map(stid_to_bay).fillna("Unknown")

# ── Long-form (one row per obs × species) ─────────────────────────────────
long_rows = []
for sp in species_cols:
    sub = all_df[["fips","year","month","bay","test_year",
                  f"predicted_{sp}", f"true_{sp}"]].copy()
    sub.columns = ["fips","year","month","bay","test_year","pred","true"]
    sub["species"] = sp
    long_rows.append(sub)
long = pd.concat(long_rows, ignore_index=True)
long = long.dropna(subset=["pred","true"])
long = long[np.isfinite(long["pred"]) & np.isfinite(long["true"])]
print(f"Long rows after dropna: {len(long):,}")

# ── Helpers ───────────────────────────────────────────────────────────────
def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    if len(y) < 2 or np.std(y) == 0:
        return dict(n=len(y), rmse=np.nan, mae=np.nan, bias=np.nan, r2=np.nan, corr=np.nan)
    rmse = np.sqrt(np.mean((p-y)**2))
    mae  = np.mean(np.abs(p-y))
    bias = np.mean(p-y)
    ss_res = np.sum((y-p)**2); ss_tot = np.sum((y-y.mean())**2)
    r2 = 1 - ss_res/ss_tot
    corr = pearsonr(y, p)[0] if np.std(p) > 0 else np.nan
    return dict(n=len(y), rmse=rmse, mae=mae, bias=bias, r2=r2, corr=corr)

# ── 1. Overall + per-fold summary ─────────────────────────────────────────
rows = []
for yr, sub in long.groupby("test_year"):
    rows.append({"test_year": yr, **metrics(sub["true"], sub["pred"])})
rows.append({"test_year": "ALL", **metrics(long["true"], long["pred"])})
overall = pd.DataFrame(rows)
overall.to_csv(f"{OUT}/overall_summary.csv", index=False)
print("\n=== OVERALL ===")
print(overall.to_string(index=False))

# ── 2. Per-species ────────────────────────────────────────────────────────
sp_rows = []
for sp, sub in long.groupby("species"):
    sp_rows.append({"species": sp, **metrics(sub["true"], sub["pred"])})
sp_df = pd.DataFrame(sp_rows).sort_values("r2", ascending=False)
sp_df.to_csv(f"{OUT}/per_species_metrics.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
order = sp_df["species"].tolist()
axes[0].barh(order, sp_df["r2"], color="steelblue"); axes[0].set_title("R² by species"); axes[0].axvline(0, c='k', lw=0.5)
axes[1].barh(order, sp_df["corr"], color="seagreen"); axes[1].set_title("Pearson r by species")
axes[2].barh(order, sp_df["rmse"], color="indianred"); axes[2].set_title("RMSE by species")
for ax in axes: ax.invert_yaxis()
plt.tight_layout(); plt.savefig(f"{OUT}/per_species_metrics.png", dpi=120); plt.close()

# ── 3. Per-month ──────────────────────────────────────────────────────────
mo_rows = []
for m, sub in long.groupby("month"):
    mo_rows.append({"month": m, **metrics(sub["true"], sub["pred"])})
mo_df = pd.DataFrame(mo_rows).sort_values("month")
mo_df.to_csv(f"{OUT}/per_month_metrics.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
x = mo_df["month"].values
axes[0].plot(x, mo_df["r2"], "o-"); axes[0].set_title("R² by month"); axes[0].axhline(0, c='k', lw=0.5)
axes[1].plot(x, mo_df["corr"], "o-", c='g'); axes[1].set_title("Corr by month")
axes[2].plot(x, mo_df["rmse"], "o-", c='r'); axes[2].set_title("RMSE by month")
for ax in axes:
    ax.set_xticks(range(1,13)); ax.set_xticklabels(MONTH_NAMES, rotation=45)
plt.tight_layout(); plt.savefig(f"{OUT}/per_month_metrics.png", dpi=120); plt.close()

# ── 4. Per-bay ────────────────────────────────────────────────────────────
bay_rows = []
for b, sub in long.groupby("bay"):
    bay_rows.append({"bay": b, **metrics(sub["true"], sub["pred"])})
bay_df = pd.DataFrame(bay_rows).sort_values("r2", ascending=False)
bay_df.to_csv(f"{OUT}/per_bay_metrics.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(18, max(4, 0.4*len(bay_df))))
order = bay_df["bay"].tolist()
axes[0].barh(order, bay_df["r2"], color="steelblue"); axes[0].set_title("R² by bay"); axes[0].axvline(0, c='k', lw=0.5)
axes[1].barh(order, bay_df["corr"], color="seagreen"); axes[1].set_title("Corr by bay")
axes[2].barh(order, bay_df["rmse"], color="indianred"); axes[2].set_title("RMSE by bay")
for ax in axes: ax.invert_yaxis()
plt.tight_layout(); plt.savefig(f"{OUT}/per_bay_metrics.png", dpi=120); plt.close()

# ── 5. Temporal trend ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
fold_df = overall[overall["test_year"] != "ALL"].copy()
fold_df["test_year"] = fold_df["test_year"].astype(int)
ax2 = ax.twinx()
ax.plot(fold_df["test_year"], fold_df["r2"],   "o-", c="steelblue", label="R²")
ax.plot(fold_df["test_year"], fold_df["corr"], "s-", c="seagreen",  label="Corr")
ax2.plot(fold_df["test_year"], fold_df["rmse"],"^-", c="indianred", label="RMSE")
ax.set_xlabel("Test year"); ax.set_ylabel("R² / Corr"); ax2.set_ylabel("RMSE")
ax.legend(loc="upper left"); ax2.legend(loc="upper right")
ax.set_title("Run H — performance vs test year")
plt.tight_layout(); plt.savefig(f"{OUT}/temporal_trend.png", dpi=120); plt.close()

# ── 6. Overall hexbin pred vs obs ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7,6))
hb = ax.hexbin(long["true"], long["pred"], gridsize=60, bins='log', cmap='viridis')
lo, hi = long[["true","pred"]].min().min(), long[["true","pred"]].max().max()
ax.plot([lo, hi], [lo, hi], "r--", lw=1)
plt.colorbar(hb, ax=ax, label="log10(N)")
ax.set_xlabel("Observed (log1p)"); ax.set_ylabel("Predicted (log1p)")
ov = metrics(long["true"], long["pred"])
ax.set_title(f"Run H — all folds | R²={ov['r2']:.3f} corr={ov['corr']:.3f} N={ov['n']:,}")
plt.tight_layout(); plt.savefig(f"{OUT}/pred_vs_obs_overall.png", dpi=120); plt.close()

# ── 7. Per-species pred vs obs grid ───────────────────────────────────────
nsp = len(species_cols)
ncol = 4; nrow = int(np.ceil(nsp/ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 3.5*nrow))
axes = axes.flatten()
for i, sp in enumerate(species_cols):
    sub = long[long["species"] == sp]
    ax = axes[i]
    ax.hexbin(sub["true"], sub["pred"], gridsize=40, bins='log', cmap='viridis')
    lo, hi = min(sub["true"].min(), sub["pred"].min()), max(sub["true"].max(), sub["pred"].max())
    ax.plot([lo,hi],[lo,hi],"r--",lw=0.8)
    m = metrics(sub["true"], sub["pred"])
    ax.set_title(f"{sp}\nR²={m['r2']:.2f} r={m['corr']:.2f}", fontsize=9)
for j in range(i+1, len(axes)): axes[j].axis('off')
plt.tight_layout(); plt.savefig(f"{OUT}/species_pred_vs_obs.png", dpi=120); plt.close()

print(f"\n✅ All artifacts → {OUT}")
