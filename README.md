# Benchmark_GAD

Benchmark_GAD is a benchmark-oriented toolkit for studying **Graph Anomaly Detection (GAD)** under more realistic dataset conditions.

Most existing GAD experiments focus on a fixed set of small or medium-scale graph datasets. However, real-world graph anomaly detection scenarios often involve larger graphs, highly imbalanced anomaly ratios, incomplete node attributes, and distribution shifts between benchmark datasets and deployment data. This repository provides data construction and transformation utilities to support systematic evaluation of GAD models under these conditions.

## Overview

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
│   ├── twitter.pt
│   └── ...
├── expand_data/
│   ├── expanded_100_twitter.pt
│   └── ...
├── miss_data/
│   ├── miss_30%_mean_pubmed.pt
│   └── ...
├── ratio_data/
│   ├── ratio_core0.1%_credit.pt
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
pip install -r requirements.txt
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

Place the original dataset under the `data/` directory

## Configuration Summary

### Graph Expansion

| Variable | Description |
| --- | --- |
| `SEED` | Random seed |
| `DATASET` | Dataset name |
| `TARGET_NODE_NUM` | Target number of nodes after expansion |

### Attribute Missingness

| Variable | Description |
| --- | --- |
| `SEED` | Random seed |
| `DATASET` | Dataset name |
| `NORMAL_NODE_MISS_RATIO` | Missing ratio for normal-node features |
| `ANOMALY_NODE_MISS_RATIO` | Missing ratio for anomaly-node features |
| `FILL_STRATEGY` | Missing-value filling strategy: `mean`, `median`, or `neighbor` |


### Anomaly Ratio Adjustment

| Variable | Description |
| --- | --- |
| `SEED` | Random seed |
| `DATASET` | Dataset name |
| `TARGET_ANOMALY_RATIO` | Target anomaly ratio |
| `SELECT_METHOD` | Anomaly preservation strategy: `core`, `edge`, or `random` |



## Notes

- The scripts are designed for node-level graph anomaly detection datasets.
- Labels are assumed to be binary: `0` for normal nodes and `1` for anomaly nodes.
- The current scripts focus on dataset construction and transformation, not model training.
- Generated datasets are saved as `.pt` files and can be loaded directly with `torch.load`.
