import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import torch
from torch_geometric.data import Data
import os
import random

SEED = 20
DATASET = "credit"
SELECT_METHOD = "core" # Three anomaly-node retention strategies (options: "core" / "edge" / "random")
TARGET_ANOMALY_RATIO = 0.005 

ORIGINAL_PT_PATH = "./data/"+DATASET+".pt" 
SAVE_PT_PATH = "./ratio_data/ratio_"+SELECT_METHOD+str(TARGET_ANOMALY_RATIO*100)+"%_"+DATASET+".pt" 

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ===================== 1. Load the original data =====================
def load_original_pyg_data(pt_path):
    """Load the original PyG data, convert it to a NetworkX graph, and collect core statistics"""
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
    
    total_node_num = len(G)
    original_anomaly_num = sum(labels)
    original_anomaly_ratio = original_anomaly_num / total_node_num
    original_edge_num = len(G.edges())
    original_degrees = np.array([G.degree(n) for n in G.nodes()])
    
    print(f"Original data statistics:")
    print(f"  - Total nodes: {total_node_num}, Total edges: {original_edge_num}")
    print(f"  - Number of anomaly nodes: {original_anomaly_num}, anomaly ratio: {original_anomaly_ratio:.4f}")
    print(f"  - mean degree: {original_degrees.mean():.2f}")
    
    original_info = {
        "total_node_num": total_node_num,
        "original_edge_num": original_edge_num,
        "original_degree_mean": original_degrees.mean(),
        "normal_feats": features[labels == 0]
    }
    
    return G, features, labels, original_info, original_anomaly_ratio

# ===================== 2. Three strategies: select anomaly nodes to keep or convert =====================
def select_anomaly_nodes_to_keep(anomaly_indices, anomaly_feats, target_anomaly_num, method, seed=SEED):
    """
    Select anomaly nodes to keep and anomaly nodes to convert according to the three strategies
    :param anomaly_indices: Index array of all anomaly nodes
    :param anomaly_feats: Feature array of all anomaly nodes
    :param target_anomaly_num: Target number of anomaly nodes to keep
    :param method: Retention strategy ("core" / "edge" / "random")
    :param seed: Random seed for reproducibility
    :return: keep_indices (nodes to keep), convert_indices (nodes to convert)
    """
    total_anomaly_num = len(anomaly_indices)
    if total_anomaly_num <= target_anomaly_num:
        return anomaly_indices, np.array([])
    
    # Strategy 1: keep the core cluster (the cluster with the largest sample size)
    if method == "core":
        print("\n--- Strategy 1: keep the core cluster ---")
        kmeans = KMeans(n_clusters=3, random_state=seed, n_init=10).fit(anomaly_feats)
        anomaly_cluster_labels = kmeans.labels_
        cluster_sizes = [sum(anomaly_cluster_labels == i) for i in range(3)]
        
        core_cluster_id = np.argmax(cluster_sizes)
        core_keep_mask = (anomaly_cluster_labels == core_cluster_id)
        core_keep_indices = anomaly_indices[core_keep_mask]
        core_keep_num = len(core_keep_indices)
        
        print(f"  Cluster sizes: {cluster_sizes} core cluster ID:  {core_cluster_id} (keep {core_keep_num} nodes)")
        
        if core_keep_num >= target_anomaly_num:
            core_center = kmeans.cluster_centers_[core_cluster_id]
            core_feats = anomaly_feats[core_keep_mask]
            core_distances = np.linalg.norm(core_feats - core_center, axis=1)
            core_sorted_indices = core_keep_indices[np.argsort(core_distances)]
            keep_indices = core_sorted_indices[:target_anomaly_num]
            convert_indices = np.concatenate([anomaly_indices[~core_keep_mask], core_sorted_indices[target_anomaly_num:]])
        
        else:
            keep_indices = core_keep_indices
            remaining_need_keep = target_anomaly_num - core_keep_num
            
            non_core_mask = ~core_keep_mask
            non_core_indices = anomaly_indices[non_core_mask]
            non_core_feats = anomaly_feats[non_core_mask]
            non_core_cluster_labels = anomaly_cluster_labels[non_core_mask]
            
            non_core_keep = []
            for cluster_id in range(3):
                if cluster_id == core_cluster_id:
                    continue
                cluster_mask = (non_core_cluster_labels == cluster_id)
                if not np.any(cluster_mask):
                    continue
                cluster_indices = non_core_indices[cluster_mask]
                cluster_feats = non_core_feats[cluster_mask]
                cluster_center = kmeans.cluster_centers_[cluster_id]
                
                cluster_distances = np.linalg.norm(cluster_feats - cluster_center, axis=1)
                cluster_sorted_indices = cluster_indices[np.argsort(cluster_distances)]
                pick_num = min(remaining_need_keep, len(cluster_sorted_indices))
                non_core_keep.extend(cluster_sorted_indices[:pick_num])
                remaining_need_keep -= pick_num
                if remaining_need_keep <= 0:
                    break
            
            keep_indices = np.concatenate([keep_indices, np.array(non_core_keep)])
            convert_indices = np.array([idx for idx in anomaly_indices if idx not in keep_indices])
    
    # Strategy 2: keep the edge cluster (the cluster with the smallest sample size)
    elif method == "edge":
        print("\n--- Strategy 2: keep the edge cluster ---")
        kmeans = KMeans(n_clusters=3, random_state=seed, n_init=10).fit(anomaly_feats)
        anomaly_cluster_labels = kmeans.labels_
        cluster_sizes = [sum(anomaly_cluster_labels == i) for i in range(3)]
        
        edge_cluster_id = np.argmin(cluster_sizes)
        edge_keep_mask = (anomaly_cluster_labels == edge_cluster_id)
        edge_keep_indices = anomaly_indices[edge_keep_mask]
        edge_keep_num = len(edge_keep_indices)
        
        print(f"  Cluster sizes: {cluster_sizes} edge cluster ID:  {edge_cluster_id} (keep {edge_keep_num} nodes)")
        
        if edge_keep_num >= target_anomaly_num:
            edge_center = kmeans.cluster_centers_[edge_cluster_id]
            edge_feats = anomaly_feats[edge_keep_mask]
            edge_distances = np.linalg.norm(edge_feats - edge_center, axis=1)
            edge_sorted_indices = edge_keep_indices[np.argsort(edge_distances)]
            keep_indices = edge_sorted_indices[:target_anomaly_num]
            convert_indices = np.concatenate([anomaly_indices[~edge_keep_mask], edge_sorted_indices[target_anomaly_num:]])
        
        else:
            keep_indices = edge_keep_indices
            remaining_need_keep = target_anomaly_num - edge_keep_num
            
            non_edge_mask = ~edge_keep_mask
            non_edge_indices = anomaly_indices[non_edge_mask]
            non_edge_feats = anomaly_feats[non_edge_mask]
            non_edge_cluster_labels = anomaly_cluster_labels[non_edge_mask]
            
            non_edge_keep = []
            for cluster_id in range(3):
                if cluster_id == edge_cluster_id:
                    continue
                cluster_mask = (non_edge_cluster_labels == cluster_id)
                if not np.any(cluster_mask):
                    continue
                cluster_indices = non_edge_indices[cluster_mask]
                cluster_feats = non_edge_feats[cluster_mask]
                cluster_center = kmeans.cluster_centers_[cluster_id]
                
                cluster_distances = np.linalg.norm(cluster_feats - cluster_center, axis=1)
                cluster_sorted_indices = cluster_indices[np.argsort(cluster_distances)[::-1]]
                pick_num = min(remaining_need_keep, len(cluster_sorted_indices))
                non_edge_keep.extend(cluster_sorted_indices[:pick_num])
                remaining_need_keep -= pick_num
                if remaining_need_keep <= 0:
                    break
            
            keep_indices = np.concatenate([keep_indices, np.array(non_edge_keep)])
            convert_indices = np.array([idx for idx in anomaly_indices if idx not in keep_indices])
    
    # Strategy 3: randomly keep anomaly nodes
    elif method == "random":
        print("\n--- Strategy 3: randomly keep anomaly nodes ---")
        np.random.seed(seed)
        shuffled_anomaly_indices = anomaly_indices.copy()
        np.random.shuffle(shuffled_anomaly_indices)
        
        keep_indices = shuffled_anomaly_indices[:target_anomaly_num]
        convert_indices = shuffled_anomaly_indices[target_anomaly_num:]
        print(f"  Randomly kept {len(keep_indices)} nodes, randomly converted {len(convert_indices)} nodes")
    
    keep_indices = np.unique(keep_indices)
    convert_indices = np.unique(convert_indices)
    
    if len(keep_indices) < target_anomaly_num and len(convert_indices) > 0:
        supplement_num = target_anomaly_num - len(keep_indices)
        supplement_indices = convert_indices[:supplement_num]
        keep_indices = np.concatenate([keep_indices, supplement_indices])
        convert_indices = convert_indices[supplement_num:]
    
    print(f"  Final kept anomaly nodes:  {len(keep_indices)}, anomaly nodes to convert: {len(convert_indices)}")
    return keep_indices, convert_indices

# ===================== 3. Core anomaly-ratio reduction function =====================
def reduce_anomaly_ratio(G, features, labels, original_info, original_anomaly_ratio, target_anomaly_ratio):
    """
    Integrate the three retention strategies, reduce the anomaly ratio to the target value, and keep anomaly nodes according to the selected strategy
    """
    total_node_num = original_info["total_node_num"]
    normal_feats_global = original_info["normal_feats"]
    global_normal_mean = np.mean(normal_feats_global, axis=0).astype(np.float32)
    
    # Step 1: boundary check
    if target_anomaly_ratio >= original_anomaly_ratio:
        print(f"Target anomaly ratio ({target_anomaly_ratio:.4f}) >= original anomaly ratio ({original_anomaly_ratio:.4f}); no modification is needed")
        return G, features, labels
    if target_anomaly_ratio < 0:
        print("Target anomaly ratio cannot be smaller than 0 and is automatically reset to 0.001")
        target_anomaly_ratio = 0.001
    
    # Step 2: compute the target number of anomaly nodes to keep
    target_anomaly_num = int(total_node_num * target_anomaly_ratio)
    original_anomaly_num = sum(labels)
    anomaly_indices = np.where(labels == 1)[0]
    anomaly_feats = features[anomaly_indices]
    
    print(f"\nAnomaly-ratio modification target:  ")
    print(f"  Original anomaly nodes:  {original_anomaly_num}, target kept anomaly nodes:  {target_anomaly_num}")
    print(f"  Anomaly nodes to convert: {original_anomaly_num - target_anomaly_num}")
    
    # Step 3: select nodes to keep/convert using the three strategies
    keep_indices, convert_indices = select_anomaly_nodes_to_keep(
        anomaly_indices=anomaly_indices,
        anomaly_feats=anomaly_feats,
        target_anomaly_num=target_anomaly_num,
        method=SELECT_METHOD
    )
    
    if len(convert_indices) <= 0:
        print("There are not enough anomaly nodes to convert, so the anomaly ratio cannot be reduced")
        return G, features, labels
    
    # Step 4: copy the data and perform conversion
    modified_features = features.copy().astype(np.float32)
    modified_labels = labels.copy().astype(int)
    
    print("\n--- Start converting anomaly nodes into normal nodes ---")
    for node_id in convert_indices:
        modified_labels[node_id] = 0
        
        neighbors = list(G.neighbors(node_id))
        if neighbors:
            normal_neighbors = [n for n in neighbors if labels[n] == 0]
            if normal_neighbors:
                neighbor_normal_feats = features[normal_neighbors]
                local_normal_mean = np.mean(neighbor_normal_feats, axis=0).astype(np.float32)
                modified_features[node_id] = local_normal_mean
                continue
        
        modified_features[node_id] = global_normal_mean
    
    # Step 5: validate the modified result
    new_anomaly_num = sum(modified_labels)
    new_anomaly_ratio = new_anomaly_num / total_node_num
    new_edge_num = len(G.edges())
    new_degree_mean = np.array([G.degree(n) for n in G.nodes()]).mean()
    
    print(f"\n--- Modification validation ---")
    print(f"  Modified anomaly nodes:  {new_anomaly_num}, anomaly ratio: {new_anomaly_ratio:.4f} (target: {target_anomaly_ratio:.4f})")
    print(f"  Total nodes: {total_node_num} (same as the original), Total edges: {new_edge_num} (same as the original)")
    print(f"  mean degree: {new_degree_mean: .2f} (original: {original_info['original_degree_mean']:.2f})")
    
    return G, modified_features, modified_labels

# ===================== 4. Save modified data =====================
def save_modified_data(G, features, labels, save_path):
    """Save the modified data as a PyG-format .pt file while preserving the full topology and features"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    edges = np.array(list(G.edges()))
    edge_index = torch.tensor(edges.T, dtype=torch.long) if len(edges) > 0 else torch.tensor([[], []], dtype=torch.long)
    
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    
    pyg_data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=len(G)
    )
    
    torch.save(pyg_data, save_path)
    print(f"\nModified data saved to: {save_path}")
    print(f"Final data information:")
    print(f"  - number of nodes: {pyg_data.num_nodes}, number of edges: {pyg_data.edge_index.shape[1]}")
    print(f"  - feature dimension: {pyg_data.x.shape[1]}, final anomaly ratio:  {pyg_data.y.sum().item() / pyg_data.num_nodes:.4f}")

# ===================== Main workflow =====================
if __name__ == "__main__":
    try:
        G, features, labels, original_info, original_anomaly_ratio = load_original_pyg_data(ORIGINAL_PT_PATH)
        
        G, modified_features, modified_labels = reduce_anomaly_ratio(
            G, features, labels, original_info, original_anomaly_ratio, TARGET_ANOMALY_RATIO
        )
        
        save_modified_data(G, modified_features, modified_labels, SAVE_PT_PATH)
        
        print("\n✅ Anomaly-ratio reduction completed. All other core distributions are consistent with the original data.")
    except Exception as e:
        print(f"\n❌ Execution failed: {type(e).__name__} - {e}")
        raise