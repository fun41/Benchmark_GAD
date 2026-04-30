import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import os
import random

# 1. Missing-rate configuration
NORMAL_NODE_MISS_RATIO = 0.5 
ANOMALY_NODE_MISS_RATIO = 0.5
# 2. Imputation strategy configuration (options: mean, median, neighbor)
FILL_STRATEGY = "mean" 
SEED = 20  
DATASET = "credit"
ORIGINAL_PT_PATH = "./data/"+DATASET+".pt" 
SAVE_PT_PATH = "./miss_data/miss_"+str(int(NORMAL_NODE_MISS_RATIO*100))+"%_"+FILL_STRATEGY+"_"+DATASET+".pt" 

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ===================== 1. Load the original data =====================
def load_original_pyg_data(pt_path):
    """Load the original PyG data, separate normal/anomaly nodes, and collect core statistics"""
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Original file does not exist: {pt_path}")
    
    data = torch.load(pt_path, map_location="cpu")
    edge_index = data.edge_index
    features = data.x.numpy().astype(np.float32)
    labels = data.y.numpy().astype(int)
    
    edges = edge_index.T.numpy()
    G = nx.from_edgelist(edges, create_using=nx.DiGraph())
    all_nodes = set(range(len(features)))
    G.add_nodes_from(all_nodes - set(G.nodes()))
    
    normal_node_indices = np.where(labels == 0)[0]
    anomaly_node_indices = np.where(labels == 1)[0]
    
    normal_node_set = set(normal_node_indices)
    anomaly_node_set = set(anomaly_node_indices)
    
    feat_dim = features.shape[1] if len(features.shape) > 1 else 1
    normal_feat_global_mean = np.nanmean(features[normal_node_indices], axis=0) if len(normal_node_indices) >0 else np.zeros(feat_dim)
    anomaly_feat_global_mean = np.nanmean(features[anomaly_node_indices], axis=0) if len(anomaly_node_indices) >0 else np.zeros(feat_dim)
    all_feat_global_mean = np.nanmean(features, axis=0) if len(features) >0 else np.zeros(feat_dim)

    total_node_num = len(G)
    total_edge_num = len(G.edges())
    anomaly_node_num = sum(labels)
    anomaly_ratio = anomaly_node_num / total_node_num
    original_degrees = np.array([G.degree(n) for n in G.nodes()])
    
    print(f"Original data statistics:")
    print(f"  - Total nodes: {total_node_num}, Total edges: {total_edge_num}")
    print(f"  - Number of anomaly nodes: {anomaly_node_num}, anomaly ratio: {anomaly_ratio:.4f}")
    print(f"  - feature dimension: {feat_dim}, mean degree: {original_degrees.mean():.2f}")
    
    original_info = {
        "G": G,
        "total_node_num": total_node_num,
        "total_edge_num": total_edge_num,
        "anomaly_ratio": anomaly_ratio,
        "feat_dim": feat_dim,
        "original_degree_mean": original_degrees.mean(),
        "normal_node_indices": normal_node_indices,
        "anomaly_node_indices": anomaly_node_indices,
        "normal_node_set": normal_node_set,    
        "anomaly_node_set": anomaly_node_set,  
        "normal_feat_global_mean": normal_feat_global_mean,
        "anomaly_feat_global_mean": anomaly_feat_global_mean, 
        "all_feat_global_mean": all_feat_global_mean         
    }
    
    return G, features, labels, original_info

# ===================== 2. Core class-aware attribute-missingness and imputation function =====================
def add_attr_missing_and_fill(features, labels, original_info, normal_miss_ratio, anomaly_miss_ratio, fill_strategy):
    """
    Step 1: add class-aware attribute missingness
    Step 2: impute missing values by class, remove NaNs, and support multiple imputation strategies
    """
    # Step 1: boundary check
    def check_miss_ratio(ratio, name):
        if not (0 <= ratio <= 1):
            print(f"{name} missing ratio ({ratio}) is outside the [0, 1] range and is automatically reset to 0.0")
            return 0.0
        return ratio
    
    normal_miss_ratio = check_miss_ratio(normal_miss_ratio, "normal nodes")
    anomaly_miss_ratio = check_miss_ratio(anomaly_miss_ratio, "anomaly nodes")
    
    # Step 2: copy original features and add missing values
    modified_features = features.copy().astype(np.float32)
    feat_dim = original_info["feat_dim"]
    G = original_info["G"]
    normal_node_indices = original_info["normal_node_indices"]
    anomaly_node_indices = original_info["anomaly_node_indices"]

    normal_node_set = original_info["normal_node_set"]
    anomaly_node_set = original_info["anomaly_node_set"]

    normal_feat_global_mean = original_info["normal_feat_global_mean"]
    anomaly_feat_global_mean = original_info["anomaly_feat_global_mean"]
    all_feat_global_mean = original_info["all_feat_global_mean"]
    
    print(f"\nAttribute-missingness modification plan:")
    print(f"  - normal nodes: {len(normal_node_indices)}, target feature missing ratio: {normal_miss_ratio:.4f}")
    print(f"  - anomaly nodes: {len(anomaly_node_indices)}, target feature missing ratio: {anomaly_miss_ratio:.4f}")
    
    miss_mask = np.zeros_like(modified_features, dtype=bool)
    
    # Step 3: add missing values to normal nodes and record the mask
    if len(normal_node_indices) > 0 and normal_miss_ratio > 0:
        normal_feat_shape = (len(normal_node_indices), feat_dim)
        normal_miss_mask = np.random.choice(
            [0, 1], 
            size=normal_feat_shape, 
            p=[1 - normal_miss_ratio, normal_miss_ratio]
        ).astype(bool)
        
        normal_features = modified_features[normal_node_indices]
        normal_features[normal_miss_mask] = np.nan
        modified_features[normal_node_indices] = normal_features
        miss_mask[normal_node_indices] = normal_miss_mask
    
    # Step 4: add missing values to anomaly nodes and record the mask
    if len(anomaly_node_indices) > 0 and anomaly_miss_ratio > 0:
        anomaly_feat_shape = (len(anomaly_node_indices), feat_dim)
        anomaly_miss_mask = np.random.choice(
            [0, 1], 
            size=anomaly_feat_shape, 
            p=[1 - anomaly_miss_ratio, anomaly_miss_ratio]
        ).astype(bool)
        
        anomaly_features = modified_features[anomaly_node_indices]
        anomaly_features[anomaly_miss_mask] = np.nan
        modified_features[anomaly_node_indices] = anomaly_features
        miss_mask[anomaly_node_indices] = anomaly_miss_mask
    
    # Step 5: impute missing values by class
    def fill_normal_node_missing(feat, normal_indices, miss_mask, fill_strategy):
        """Impute missing values for normal nodes"""
        normal_miss_mask = miss_mask[normal_indices]
        
        if fill_strategy == "mean":
            normal_feat_mean = np.nanmean(feat[normal_indices], axis=0)
            for i in range(len(normal_indices)):
                feat[normal_indices[i]][normal_miss_mask[i]] = normal_feat_mean[normal_miss_mask[i]]
        
        elif fill_strategy == "median":
            normal_feat_median = np.nanmedian(feat[normal_indices], axis=0)
            for i in range(len(normal_indices)):
                feat[normal_indices[i]][normal_miss_mask[i]] = normal_feat_median[normal_miss_mask[i]]
        
        elif fill_strategy == "neighbor":
            for node_id in normal_indices:
                node_miss_mask = miss_mask[node_id]
                if not np.any(node_miss_mask): continue 
                
                # 1. Select same-class neighbors
                neighbors = list(G.neighbors(node_id))
                normal_neighbors = [n for n in neighbors if n in normal_node_set]
                
                fill_vals = normal_feat_global_mean.copy()
                
                # First-level imputation: compute the non-NaN neighbor mean
                if len(normal_neighbors) > 0:
                    neighbor_feat = feat[normal_neighbors]
                    neighbor_mean = np.nanmean(neighbor_feat, axis=0)
                    for dim in range(feat_dim):
                        if not np.isnan(neighbor_mean[dim]):
                            fill_vals[dim] = neighbor_mean[dim]
                        else:
                            fill_vals[dim] = normal_feat_global_mean[dim]
                
                # Second-level imputation: use the global graph mean
                for dim in range(feat_dim):
                    if np.isnan(fill_vals[dim]):
                        fill_vals[dim] = all_feat_global_mean[dim]
                
                fill_vals = np.nan_to_num(fill_vals, nan=0.0)
                
                feat[node_id][node_miss_mask] = fill_vals[node_miss_mask]
        
        return feat
    
    def fill_anomaly_node_missing(feat, anomaly_indices, miss_mask, fill_strategy):
        """Impute missing values for anomaly nodes"""
        anomaly_miss_mask = miss_mask[anomaly_indices]
        
        if fill_strategy == "mean":
            anomaly_feat_mean = np.nanmean(feat[anomaly_indices], axis=0)
            for i in range(len(anomaly_indices)):
                feat[anomaly_indices[i]][anomaly_miss_mask[i]] = anomaly_feat_mean[anomaly_miss_mask[i]]
        
        elif fill_strategy == "median":
            anomaly_feat_median = np.nanmedian(feat[anomaly_indices], axis=0)
            for i in range(len(anomaly_indices)):
                feat[anomaly_indices[i]][anomaly_miss_mask[i]] = anomaly_feat_median[anomaly_miss_mask[i]]
        
        elif fill_strategy == "neighbor":
            for node_id in anomaly_indices:
                node_miss_mask = miss_mask[node_id]
                if not np.any(node_miss_mask): continue 
                
                # 1. Select same-class neighbors
                neighbors = list(G.neighbors(node_id))
                anomaly_neighbors = [n for n in neighbors if n in anomaly_node_set]
                
                fill_vals = anomaly_feat_global_mean.copy()
                
                # First-level imputation: compute the non-NaN neighbor mean
                if len(anomaly_neighbors) > 0:
                    neighbor_feat = feat[anomaly_neighbors]
                    neighbor_mean = np.nanmean(neighbor_feat, axis=0)
                    for dim in range(feat_dim):
                        if not np.isnan(neighbor_mean[dim]):
                            fill_vals[dim] = neighbor_mean[dim]
                        else:
                            fill_vals[dim] = anomaly_feat_global_mean[dim]
                
                # Second-level imputation: use the global graph mean
                for dim in range(feat_dim):
                    if np.isnan(fill_vals[dim]):
                        fill_vals[dim] = all_feat_global_mean[dim]
                
                fill_vals = np.nan_to_num(fill_vals, nan=0.0)
                
                feat[node_id][node_miss_mask] = fill_vals[node_miss_mask]
        
        return feat
    
    print(f"\nStart imputing missing values; strategy: {fill_strategy}")
    if len(normal_node_indices) > 0 and normal_miss_ratio > 0:
        modified_features = fill_normal_node_missing(modified_features, normal_node_indices, miss_mask, fill_strategy)
    if len(anomaly_node_indices) > 0 and anomaly_miss_ratio > 0:
        modified_features = fill_anomaly_node_missing(modified_features, anomaly_node_indices, miss_mask, fill_strategy)

    modified_features = np.nan_to_num(modified_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Step 6: validate imputation results
    def calc_actual_miss_ratio(miss_mask, node_indices):
        if len(node_indices) == 0: return 0.0
        node_miss_mask = miss_mask[node_indices]
        total_elem_num = node_miss_mask.size
        miss_elem_num = node_miss_mask.sum()
        return miss_elem_num / total_elem_num if total_elem_num > 0 else 0.0
    
    normal_actual_miss = calc_actual_miss_ratio(miss_mask, normal_node_indices)
    anomaly_actual_miss = calc_actual_miss_ratio(miss_mask, anomaly_node_indices)
    overall_actual_miss = calc_actual_miss_ratio(miss_mask, np.arange(len(modified_features)))
    has_nan = np.isnan(modified_features).any() 
    
    print(f"\nAttribute-missingness and imputation validation:")
    print(f"  - normal nodes: target missing ratio = {normal_miss_ratio:.4f}, actual missing ratio = {normal_actual_miss:.4f}")
    print(f"  - anomaly nodes: target missing ratio = {anomaly_miss_ratio:.4f}, actual missing ratio = {anomaly_actual_miss:.4f}")
    print(f"  - overall dataset: actual missing ratio = {overall_actual_miss:.4f}, contains NaN after imputation: {has_nan}")
    print(f"\nOther core-attribute validation (should match the original data):")
    print(f"  - feature dimension: {modified_features.shape[1]} (original: {feat_dim})")
    print(f"  - number of nodes: {len(modified_features)} (original: {original_info['total_node_num']})")
    
    return modified_features, miss_mask

# ===================== 3. Save modified data =====================
def save_filled_data(G, modified_features, labels, miss_mask, original_info, save_path):
    """Save the imputed data with the missing-value mask for graph-model training"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    edges = np.array(list(G.edges()))
    edge_index = torch.tensor(edges.T, dtype=torch.long) if len(edges) > 0 else torch.tensor([[], []], dtype=torch.long)
    
    x = torch.tensor(modified_features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    miss_mask_tensor = torch.tensor(miss_mask, dtype=torch.bool)
    
    pyg_data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=len(G),
        normal_attr_miss_ratio=NORMAL_NODE_MISS_RATIO,
        anomaly_attr_miss_ratio=ANOMALY_NODE_MISS_RATIO,
        fill_strategy=FILL_STRATEGY,
        attr_miss_mask=miss_mask_tensor
    )
    
    torch.save(pyg_data, save_path)
    print(f"\nModified data saved to: {save_path}")
    print(f"Final data information:")
    print(f"  - number of nodes: {pyg_data.num_nodes}, number of edges: {pyg_data.edge_index.shape[1]}")
    print(f"  - feature dimension: {pyg_data.x.shape[1]}, contains NaN: {torch.isnan(pyg_data.x).any().item()}")
    print(f"  - anomaly ratio: {pyg_data.y.sum().item() / pyg_data.num_nodes:.4f} (same as the original)")
    print(f"  - includes missing-value mask: {pyg_data.attr_miss_mask is not None}")

# ===================== Main workflow =====================
if __name__ == "__main__":
    try:
        G, features, labels, original_info = load_original_pyg_data(ORIGINAL_PT_PATH)
        modified_features, miss_mask = add_attr_missing_and_fill(
            features,
            labels,
            original_info,
            NORMAL_NODE_MISS_RATIO,
            ANOMALY_NODE_MISS_RATIO,
            FILL_STRATEGY
        )
        save_filled_data(G, modified_features, labels, miss_mask, original_info, SAVE_PT_PATH)
        print("\n✅ Class-aware attribute missingness and imputation completed. The data can be used directly for model training.")
    except Exception as e:
        print(f"\n❌ Execution failed: {type(e).__name__} - {e}")
        raise