# Benchmark_GAD

Benchmark_GAD is a benchmark-oriented toolkit for studying **Graph Anomaly Detection (GAD)** under more realistic dataset conditions.

Most existing GAD experiments focus on a fixed set of small or medium-scale graph datasets. However, real-world graph anomaly detection scenarios often involve larger graphs, highly imbalanced anomaly ratios, incomplete node attributes, and distribution shifts between benchmark datasets and deployment data. This repository provides data construction and transformation utilities to support systematic evaluation of GAD models under these conditions.

## Overview

This repository focuses on controlled benchmark construction for GAD. It provides utilities for:

- **Graph scale expansion** while preserving key structural and feature distributions.
- **Anomaly ratio adjustment** to create datasets with different levels of anomaly imbalance.
- **Attribute missingness simulation** with configurable missing ratios and filling strategies.
- **Distribution validation**, including statistics such as anomaly ratio, edge-node ratio, degree distribution, feature statistics, and KS tests.

The generated datasets are saved in the PyTorch Geometric `Data` format and can be directly used by GNN-based anomaly detection models.

## Key Features

### 1. Distribution-preserving graph expansion

The graph expansion script increases the number of nodes to a target scale while attempting to preserve the original dataset properties, including:

- anomaly ratio;
- edge-node ratio;
- degree distribution;
- normal and anomaly feature distributions;
- high-, medium-, and low-degree node connection preferences.

It also supports post-expansion validation using the Kolmogorov-Smirnov (KS) test for degree distributions and feature dimensions.

### 2. Controlled anomaly ratio adjustment

The anomaly ratio adjustment script reduces the anomaly ratio to a target value while keeping the graph topology unchanged.

Supported anomaly preservation strategies include:

- `core`: preserve anomaly nodes from the largest anomaly cluster;
- `edge`: preserve anomaly nodes from the smallest anomaly cluster;
- `random`: randomly preserve anomaly nodes.

Converted anomaly nodes are relabeled as normal nodes, and their features are replaced using normal-neighbor statistics when available, or global normal statistics otherwise.

### 3. Attribute missingness simulation

The attribute missingness script introduces missing values into node features with separate missing ratios for normal and anomaly nodes.

Supported filling strategies include:

- `mean`: fill missing values with class-wise feature means;
- `median`: fill missing values with class-wise feature medians;
- `neighbor`: fill missing values using same-class neighbor statistics, with global statistics as fallback.

The script stores the missing mask together with the processed PyTorch Geometric data object.

## Repository Structure

A typical structure is shown below.

```
Benchmark_GAD/
├── data/
│   ├── credit.pt
│   ├── twitter.pt
│   └── ...
├── data_expand.py
├── data_miss.py
├── data_ratio.py
├── README.md
└── requirements.txt
```

If your local file names are different, update the script names and paths accordingly.

## Installation

Create a Python environment and install the required dependencies.

```bash
conda create -n benchmark_gad python=3.9
conda activate benchmark_gad
```

Install the core packages:

```bash
pip install numpy networkx scikit-learn scipy
pip install torch
pip install torch-geometric
```

Depending on your CUDA and PyTorch versions, you may need to install PyTorch and PyTorch Geometric using their official installation commands.

## Data Format

Input datasets should be stored as PyTorch Geometric `Data` objects.

Each dataset is expected to contain at least:

```python
data.x           # Node feature matrix, shape: [num_nodes, num_features]
data.edge_index  # Graph edges, shape: [2, num_edges]
data.y           # Node labels, 0 for normal nodes and 1 for anomaly nodes
```

Example:

```python
from torch_geometric.data import Data

data = Data(
    x=x,
    edge_index=edge_index,
    y=y,
    num_nodes=x.shape[0]
)
```

Place the original dataset under the `data/` directory, for example:

```text
data/credit.pt
data/twitter.pt
```

## Usage

### 1. Expand graph scale

Configure the dataset name and target node number in the graph expansion script.

```python
DATASET = "credit"
TARGET_NODE_NUM = 100000
```

Run:

```bash
python scripts/expand_graph_preserve_dist.py
```

The output file will be saved to:

```text
data/expanded_<scale>_<dataset>.pt
```

For example:

```text
data/expanded_10.0_credit.pt
```

### 2. Simulate attribute missingness

Configure the normal/anomaly missing ratios and filling strategy.

```python
DATASET = "credit"
NORMAL_NODE_MISS_RATIO = 0.5
ANOMALY_NODE_MISS_RATIO = 0.5
FILL_STRATEGY = "mean"
```

Run:

```bash
python scripts/add_attr_missing_and_fill.py
```

The output file will be saved to:

```text
data/miss_<ratio>_<fill_strategy>_<dataset>.pt
```

### 3. Adjust anomaly ratio

Configure the target anomaly ratio and anomaly preservation strategy.

```python
DATASET = "credit"
TARGET_ANOMALY_RATIO = 0.005
SELECT_METHOD = "core"  # options: "core", "edge", "random"
```

Run:

```bash
python scripts/reduce_anomaly_ratio.py
```

The output file will be saved to:

```text
data/ratio_<method><ratio>%_<dataset>.pt
```

## Validation

The scripts report key statistics before and after transformation, including:

- number of nodes;
- number of edges;
- anomaly ratio;
- edge-node ratio;
- degree distribution statistics;
- feature dimensionality;
- missing-value ratio;
- whether NaN values remain after filling.

For graph expansion, KS tests can be used to compare the original and expanded distributions.

A typical validation requirement is:

```text
KS p-value > 0.05
```

The degree distribution is validated using the original and expanded node degree arrays.  
The feature matrix is validated by applying the KS test to each feature dimension independently.

When the graph is very large, the KS test may become sensitive to very small differences. In that case, it is recommended to report both:

```text
KS statistic
p-value
```

rather than relying only on the p-value.

## Configuration Summary

### Graph Expansion

| Variable | Description |
| --- | --- |
| `DATASET` | Dataset name |
| `TARGET_NODE_NUM` | Target number of nodes after expansion |
| `ORIGINAL_PT_PATH` | Path to the original `.pt` dataset |
| `SAVE_PT_PATH` | Path to save the expanded dataset |
| `SEED` | Random seed |

### Attribute Missingness

| Variable | Description |
| --- | --- |
| `NORMAL_NODE_MISS_RATIO` | Missing ratio for normal-node features |
| `ANOMALY_NODE_MISS_RATIO` | Missing ratio for anomaly-node features |
| `FILL_STRATEGY` | Missing-value filling strategy: `mean`, `median`, or `neighbor` |
| `DATASET` | Dataset name |
| `SEED` | Random seed |

### Anomaly Ratio Adjustment

| Variable | Description |
| --- | --- |
| `TARGET_ANOMALY_RATIO` | Target anomaly ratio |
| `SELECT_METHOD` | Anomaly preservation strategy: `core`, `edge`, or `random` |
| `DATASET` | Dataset name |
| `SEED` | Random seed |

## Reproducibility

All scripts use a fixed random seed by default:

```python
SEED = 20
```

The seed is applied to:

- Python `random`;
- NumPy;
- PyTorch CPU;
- PyTorch CUDA, if available.

For more robust evaluation, it is recommended to run each setting with multiple random seeds and report the mean and standard deviation.

## Notes

- The scripts are designed for node-level graph anomaly detection datasets.
- Labels are assumed to be binary: `0` for normal nodes and `1` for anomaly nodes.
- The current scripts focus on dataset construction and transformation, not model training.
- Generated datasets are saved as `.pt` files and can be loaded directly with `torch.load`.

## Citation

If you use this repository in your research, please cite our work:

```bibtex
@article{benchmarkgad2026,
  title   = {Benchmarking Graph Anomaly Detection under Realistic Dataset Conditions},
  author  = {Your Name and Coauthors},
  journal = {arXiv preprint},
  year    = {2026}
}
```

Please replace the placeholder citation with the final paper information after publication.

## License

Please add a license file before public release. For academic open-source projects, commonly used licenses include MIT, Apache-2.0, and BSD-3-Clause.

## Contact

For questions, issues, or suggestions, please open an issue in this repository.
