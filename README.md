# Bayesian GNN-LSTM for Florida Estuarine Fish Community Modeling

Bayesian Graph Neural Network with two-level Long Short-Term Memory (GNN-LSTM) for spatiotemporal joint species distribution modeling of Florida's estuarine fish communities, using Fisheries-Independent Monitoring (FIM) seine sampling data.

Adapted from [Fan et al. (2022)](https://arxiv.org/pdf/2111.08900.pdf) GNN-RNN framework (AAAI 2022).

## Environment Setup

```bash
conda create -n gnnrnn python=3.10
conda activate gnnrnn

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c dglteam/label/th24_cu124 dgl
conda install numpy scipy scikit-learn pandas matplotlib tensorboard
pip install torch-geometric geopandas
```

### Verify

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import dgl; print('DGL:', dgl.__version__)"
python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)"
```

## File Structure

```
Bayesian-GNN-LSTM-FIM/
|
|-- data/                          # All data and adjacency files
|   |-- preprocess_restoration_0412.py           # Raw FIM CSV -> NPZ
|   |-- make_prevalent_targets_npz.py            # Prevalence-based target selection
|   |-- make_prey_guild_npz.py                   # Prey guild NPZ builder
|   |-- FIM_restoration_0412G_stations_monthly.npz   # Training data (Gear-020 filtered)
|   |-- FIM_restoration_0412_stations_monthly.npz    # Training data (unfiltered)
|   |-- FIM_restoration_0412G_species_names.pkl      # Species name list
|   |-- FIM_restoration_0412G_station_metadata.pkl   # Station lat/lon metadata
|   |-- FIM_restoration_0412_v2_adj.pkl              # Sparse CSR adjacency matrix
|   |-- FIM_restoration_0412_v2_fid_dict.pkl         # Node ID mapping dictionary
|   |-- FIM_restoration_0412_v2_dist_weights.pkl     # Edge distance weights
|   `-- FIM_with_restoration_75000_0412.csv          # Raw CSV source
|
|-- main.py                        # Entry point and argument parsing
|-- model.py                       # GNN-LSTM architecture (GAT + two-level LSTM + FiLM)
|-- train.py                       # Training loop, rolling windowed CV, loss functions
|-- test.py                        # Test evaluation pipeline
|-- utils.py                       # Data loading utilities
|-- utils/                         # Modular utilities
|   |-- data_utils.py              # Data loading and sequence building
|   |-- losses.py                  # Stable logcosh, hurdle loss
|   `-- metrics.py                 # Evaluation metrics
|-- species_metrics.py             # Per-species R2/RMSE/correlation analysis
|-- model_evaluation.py            # Model comparison metrics
|-- run_train.sh                   # Training launcher script
|
|-- analysis/                      # Post-hoc analysis scripts
|   |-- shap_analysis.py           # SHAP feature importance
|   |-- pdp_analysis.py            # Partial dependence plots
|   |-- bay_trends.py              # Bay-level population trend analysis
|   |-- species_summary.py         # Species-level summary statistics
|   |-- compute_month_weights.py   # Per-species monthly abundance weights
|   |-- visualize_gnn.py           # Spatial graph visualization
|   |-- runH_comprehensive.py      # Comprehensive Run H evaluation
|   |-- compare_pred_obs.py        # Predicted vs observed comparison
|   `-- comprehensive_metrics.py   # Detailed metric tables
|
|-- results/                       # Model output CSVs
|-- model/                         # Saved checkpoints
|-- logs/                          # TensorBoard logs
`-- summary/                       # Summary tables
```

## Quick Start

### 1. Preprocess Data

```bash
cd data
python preprocess_restoration_0412.py
```

### 2. Train the Model

```bash
bash run_train.sh
```

This runs windowed rolling cross-validation (2010-2024) with:
- Gear-020 Red Drum filter
- Pinfish as forage input feature
- log1p transform
- 35 epochs with early stopping (patience 25)
- 10 sliding windows of 3-year merges

### 3. Evaluate Results

```bash
python species_metrics.py --results_dir results/FIM_restoration_0412G_stations_monthly/
```

Results CSVs contain columns: `fips` (node ID), `year`, `predicted_{species}`, `true_{species}`.

## Shell Script Reference

| Script | Description |
|--------|-------------|
| `run_train.sh` | Run H: GAT + two-level LSTM, Pinfish forage, Gear-020 Red Drum filter, rolling windowed CV (2010-2024), 35 epochs |

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{fan2022gnn,
  title={A GNN-RNN approach for harnessing geospatial and temporal information: application to crop yield prediction},
  author={Fan, Joshua and Bai, Junwen and Li, Zhiyun and Ortiz-Bobea, Ariel and Gomes, Carla P},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={11},
  pages={11873--11881},
  year={2022}
}
```

For the FIM estuarine fish community adaptation (manuscript in preparation):

```bibtex
@article{li2025gnnlstm,
  title={Bayesian Graph Neural Network with Long Short-Term Memory for Joint Species Distribution Modeling of Florida Estuarine Fish Communities},
  author={Li, Ying and [co-authors]},
  journal={[in preparation]},
  year={2025}
}
```

## References

- [GNN-RNN (Fan et al. 2022)](https://arxiv.org/pdf/2111.08900.pdf)
- [FWC FIM Program](https://myfwc.com/research/saltwater/fishstats/fim/)

## License

Please contact the authors for licensing information.
