#!/bin/bash
# ============================================================================
# RUN H  — gnn-lstm-v4    (Re-run of Run A's setup with two fixes)
# ============================================================================
#   • Pinfish-as-input feature (12 management targets)
#   • log1p transform; predictions back-transformed to counts at result time
#     via expm1 in train.py result-CSV writer
#   • Loss is log1p-scale standardized per species (uses args.means/args.stds)
#     so abundance magnitudes are aligned across species
#
# Two data/training fixes vs original Run A:
#   1. Gear-020 Red Drum filter
#      Training NPZ is FIM_restoration_0412G_*
#      → Sciaenops ocellatus_R counts in Gear-020 hauls are zeroed
#   2. 35 epochs / early_stop_patience 25
#      → Faster convergence, stop when val_rmse plateaus
#
# Other settings (Run A baseline):
#   • 12 management targets (orig 14 minus 2 forage Pinfish)
#   • 92 community species + Pinfish (forage) as X input features
#   • MLP encoder, GAT spatial, two-level LSTM
#   • Rolling-windowed CV: 15 prediction years (2010–2024) × 10 sliding windows
# ============================================================================

cd "$(dirname "$0")"

PYTHONUNBUFFERED=1 /Users/liyingnceas/anaconda3/envs/gnnrnn/bin/python -u main.py \
  -dataset FIM_restoration_0412G_stations_monthly \
  -dd ../data/FIM_restoration_0412G_stations_monthly.npz \
  -adj ../map/FIM_restoration_0412_v2_adj.pkl \
  -fid_map ../map/FIM_restoration_0412_v2_fid_dict.pkl \
  --dist_weights_path ../map/FIM_restoration_0412_v2_dist_weights.pkl \
  --no_soil --max_epoch 35 \
  --encoder_type mlp \
  --rolling_windowed \
  --rolling_start 2010 --rolling_end 2024 \
  --win_size 3 --n_windows 10 \
  --forage_species "Lagodon rhomboides_A,Lagodon rhomboides_R" \
  --transform log1p \
  --early_stop_patience 25
