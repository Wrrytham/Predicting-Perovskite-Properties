# Predicting Perovskite Properties with Graph Neural Networks

A machine learning pipeline that predicts two key material properties of perovskite compounds — **formation energy** (eV/atom) and **band gap** (eV) — using Graph Neural Networks (GNNs) with composition-based feature vectors (CBFV).

---

## Overview

Perovskites are a family of materials with enormous applications in solar cells, superconductors, and catalysis. Computationally predicting their properties avoids costly DFT simulations and accelerates materials discovery.

This project represents each perovskite as a molecular graph where:
- **Nodes** are individual atoms, featurized using CBFV elemental descriptors
- **Edges** connect each atom to the compositionally dominant "center" atom, featurized with pairwise CBFV features and structural parameters
- **Global features** encode the full composition CBFV vector plus crystal structure information (density, volume, magnetization, crystal system)

Three GNN architectures are trained and compared, along with a composition-only MLP baseline. An ensemble of each GNN with the MLP is also evaluated to improve band gap predictions.

---

## Repository Structure

```
Predicting-Perovskite-Properties/
├── GraphBuild2.py              # Graph construction & CBFV featurization
├── model2.py                   # GNN architectures, training, evaluation
├── compare_featurizers.py      # Cross-featurizer comparison experiments
├── featurization_analysis.py   # Feature importance & analysis utilities
├── DDM Report.pdf              # Project report
├── GRAPH NEURAL NETWORK.pdf    # Background literature / methodology notes
└── LICENSE                     # MIT License
```

---

## Models

| Model | Architecture | Description |
|---|---|---|
| `MaterialsGNN` | Graph Attention Network (GAT) | 6-layer GAT with multi-head attention (12 heads), residual connections, BatchNorm |
| `MaterialsGCN` | Graph Convolutional Network (GCN) | 4-layer GCN with edge-weight projection from CBFV features |
| `MaterialsGraphSAGE` | GraphSAGE | 4-layer mean-aggregation SAGE convolution |
| `CompositionMLP` | MLP baseline | Operates on global CBFV + structural features only, no graph structure |

Each GNN output is ensembled with the composition MLP (weighted average, 60% MLP / 40% GNN for band gap) to boost band gap prediction accuracy.

---

## Featurizers

Three CBFV presets from the [CBFV library](https://github.com/kaaiian/CBFV) are supported:

| Preset | Description | # Features (selected) |
|---|---|---|
| `oliynyk` | Atomic/electronic properties (radii, electronegativity, ionization energies, cohesive energy) | 35 |
| `magpie` | Statistical + electronic descriptors (Mendeleev number, valence, space group) | 24 |
| `jarvis` | Physical & chemical properties (coordination, oxidation states, thermal conductivity) | 41 |

Features are computed as **average** (`avg_`) and **deviation** (`dev_`) statistics over the elements in a composition.

---

## Data Format

The pipeline expects a CSV file named `Perovskite_data_cleaned.csv` with the following columns:

| Column | Description |
|---|---|
| `formula` | Chemical formula string (e.g., `CaTiO3`) |
| `composition` | Space-separated atom tokens (e.g., `Ca Ti O O O`) |
| `crystal_system` | One of: cubic, tetragonal, orthorhombic, monoclinic, triclinic, hexagonal, trigonal |
| `a_edge (angstrom)` | Lattice parameter *a* |
| `alpha_ang (deg)` | Lattice angle *α* |
| `density (g/cc)` | Density |
| `total_magnetisation (bohr)` | Total magnetization |
| `volume (cubic-angstrom)` | Unit cell volume |
| `formation_energy (eV/atom)` | **Target 1** |
| `band_gap (eV)` | **Target 2** |

Rows with `band_gap = 0` or missing critical columns are automatically dropped. Outliers (> 3σ) in structural columns are clipped to the median.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Wrrytham/Predicting-Perovskite-Properties.git
cd Predicting-Perovskite-Properties

# Install dependencies
pip install torch torch-geometric
pip install pandas numpy scikit-learn matplotlib networkx
pip install CBFV
```

> **Note:** Make sure your PyTorch and PyTorch Geometric versions are compatible. See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for details.

---

## Usage

### Training

Place your cleaned dataset (`Perovskite_data_cleaned.csv`) in the project root, then run:

```bash
python model2.py
```

By default, this trains all three GNN architectures across all three CBFV featurizer presets (`oliynyk`, `magpie`, `jarvis`) with the following hyperparameters:

```python
model_params = {
    'hidden_dim': 256,
    'num_conv_layers': 6,
    'dropout': 0.2
}

training_params = {
    'num_epochs': 300,
    'lr': 0.001,
    'batch_size': 64,
    'patience': 25      # Early stopping patience
}
```

To use a specific featurizer in isolation:

```python
from model2 import train_models

train_models(
    csv_file="Perovskite_data_cleaned.csv",
    CBFV_preset="oliynyk",   # or 'magpie', 'jarvis'
    model_params=model_params,
    training_params=training_params
)
```

### Graph Construction Only

```python
from GraphBuild2 import process_csv_CBFV

graphs, featurizer, y_scaler = process_csv_CBFV(
    "Perovskite_data_cleaned.csv",
    CBFV_preset="oliynyk"
)

print(f"Graphs built: {len(graphs)}")
print(f"Node feature dim: {graphs[0].x.shape[1]}")
print(f"Edge feature dim: {graphs[0].edge_attr.shape[1]}")
print(f"Global feature dim: {graphs[0].u.shape[1]}")
```

### Visualizing a Graph

```python
from GraphBuild2 import process_csv_CBFV, plot_graph_with_CBFV

graphs, featurizer, _ = process_csv_CBFV("Perovskite_data_cleaned.csv")
plot_graph_with_CBFV(graphs[0], featurizer, filename="example_graph.png")
```

---

## Outputs

After training, the following files are saved:

```
outputs/
├── training_curves_<model>.png          # Train/val loss curves
├── predictions_<model>_<featurizer>.png # Parity plots for formation energy & band gap

models/
├── <model>_<featurizer>.pth             # Saved model weights

feature_info_<model>_<featurizer>.json   # Feature dimensions and selected feature names
test_predictions_<model>_<featurizer>.csv
train_predictions_<model>_<featurizer>.csv
metrics_<model>_<featurizer>_test.csv
test_predictions_<model>_ensembled.csv   # Ensemble (GNN + composition MLP)
dropped_rows.csv                         # Rows removed due to missing values
```

---

## Evaluation Metrics

Each model is evaluated on:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of determination

Reported separately for formation energy and band gap on both training and test splits. If a `y_scaler` is available, metrics are reported in the original physical units (eV/atom and eV).

---

## License

This project is licensed under the [MIT License](LICENSE).
