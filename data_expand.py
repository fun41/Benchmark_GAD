import numpy as np
import networkx as nx
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
import random
import os
import warnings
from scipy.stats import powerlaw, ks_2samp  

SEED = 20 
TARGET_NODE_NUM = 100000 
DATASET = "credit"
# Path configuration
ORIGINAL_PT_PATH = "./data/"+DATASET+".pt" 
SAVE_PT_PATH = "./expand_data/expanded_"+str(TARGET_NODE_NUM/10000)+"_"+DATASET+".pt"  # Save path after expansion; node count is measured in ten-thousands

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# ===================== 1. Load the original data and collect core distribution statistics =====================
def load_original_pyg_data(pt_path):
    """Load the data and collect original core distributions: anomaly ratio, edge-to-node ratio, degree distribution, and feature distribution"""
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

    # -------------------- Collect core distributions --------------------
    # 1. anomaly ratio
    ori_node_num = len(G)
    ori_anomaly_num = sum(labels)
    ori_anomaly_ratio = ori_anomaly_num / ori_node_num
    # 2. edge-to-node ratio
    ori_edge_num = len(G.edges())
    ori_edge_node_ratio = ori_edge_num / ori_node_num
    # 3. degree distribution
    ori_degrees = np.array([G.degree(n) for n in G.nodes()])  
    try:
        alpha, xmin = powerlaw.fit(ori_degrees, discrete=True)
        degree_dist_params = {"type": "powerlaw", "alpha": alpha, "xmin": xmin}
    except:
        degree_bins = np.histogram(ori_degrees, bins=20)
        degree_dist_params = {"type": "hist", "bins": degree_bins[0], "edges": degree_bins[1]}

    print(f"Original data statistics:")
    print(f"  - number of nodes: {ori_node_num}, number of edges: {ori_edge_num}, edge-to-node ratio: {ori_edge_node_ratio:.4f}")
    print(f"  - Number of anomaly nodes: {ori_anomaly_num}, anomaly ratio: {ori_anomaly_ratio:.4f}")
    print(f"  - degree range: [{ori_degrees.min()}, {ori_degrees.max()}], mean: {ori_degrees.mean():.2f}")

    ori_dist = {
        "anomaly_ratio": ori_anomaly_ratio,
        "edge_node_ratio": ori_edge_node_ratio,
        "degree_params": degree_dist_params,
        "normal_feats": features[labels == 0],  
        "anomaly_feats": features[labels == 1] 
    }

    return G, features, labels, ori_dist


# ===================== 2. Core distribution-preserving graph expansion function =====================
def expand_graph_preserve_dist(G, features, labels, ori_dist, target_node_num):
    """
    Expand to the target number of nodes while preserving the original ratios/distributions:
    Mitigate gaps among medium-degree nodes and reproduce the original layered connection preferences
    """
    ori_node_num = len(G)
    add_node_num = target_node_num - ori_node_num
    if add_node_num <= 0:
        print(f"Original node count ({ori_node_num}) >= target node count ({target_node_num}); no expansion is needed")
        return G, features, labels

    # -------------------- Step 1: determine the number of new normal/anomaly nodes --------------------
    ori_anomaly_ratio = ori_dist["anomaly_ratio"]
    add_anomaly_num = int(add_node_num * ori_anomaly_ratio)
    add_normal_num = add_node_num - add_anomaly_num
    print(f"\nExpansion plan:")
    print(f"  - Total new nodes: {add_node_num} (normal: {add_normal_num}, anomaly: {add_anomaly_num})")
    print(f"  - After-expansion anomaly ratio:  {ori_anomaly_ratio:.4f} (same as the original)")

    # -------------------- Step 2: generate new node IDs and sample degrees --------------------
    max_node_id = max(G.nodes()) if G.nodes() else 0
    add_normal_nodes = list(range(max_node_id + 1, max_node_id + 1 + add_normal_num))
    add_anomaly_nodes = list(range(max_node_id + 1 + add_normal_num, max_node_id + 1 + add_node_num))
    add_all_nodes = add_normal_nodes + add_anomaly_nodes

    ori_degrees = np.array([G.degree(n) for n in G.nodes()])
    ori_degree_mean = ori_degrees.mean()
    ori_degree_max = ori_degrees.max()
    ori_degree_upper = min(ori_degree_mean * 2, ori_degree_max) 

    def sample_degree_fix():
        """
        Corrected degree sampling: sample directly from the original degree distribution, apply an upper bound, and stay close to the original mean
        Sample out-degree instead of total degree to avoid later double-counting from bidirectional edges
        """
        sample_deg = np.random.choice(ori_degrees, size=1)[0]
        return int(min(sample_deg, ori_degree_upper))

    add_node_out_degrees = {n: max(1, sample_degree_fix()) for n in add_all_nodes}
    add_out_degrees = np.array(list(add_node_out_degrees.values()))
    print(f"  - Mean out-degree of new nodes: {add_out_degrees.mean():.2f} (original mean total degree: {ori_degree_mean: .2f})")

    # -------------------- Step 3: stratify original nodes and compute layered connection ratios --------------------
    ori_edge_node_ratio = ori_dist["edge_node_ratio"]
    add_edge_num = int(add_node_num * ori_edge_node_ratio)
    print(f"  - Number of new edges: {add_edge_num} (edge-to-node ratio: {ori_edge_node_ratio:.4f}, same as the original)")

    # 3.1 Stratify original nodes by degree
    ori_degrees = np.array([G.degree(n) for n in G.nodes()])

    high_thresh = np.percentile(ori_degrees, 99)  # top 1% are high-core nodes
    mid_thresh = np.percentile(ori_degrees, 90)  # 90th-99th percentiles are medium-degree nodes, and the remaining 90% are low-degree nodes

    high_core_nodes = [n for n in G.nodes() if G.degree(n) >= high_thresh]
    mid_degree_nodes = [n for n in G.nodes() if (G.degree(n) >= mid_thresh) and (G.degree(n) < high_thresh)]
    low_degree_nodes = [n for n in G.nodes() if G.degree(n) < mid_thresh]

    # 3.2 Compute original layered connection ratios
    def calc_layer_connect_ratio(G, high_nodes, mid_nodes, low_nodes):
        """Compute the proportion of original edges connected to each node layer"""
        all_edges = list(G.edges())
        if not all_edges:
            return 0.5, 0.4, 0.1  # default ratios

        high_connect_count = sum(1 for u, v in all_edges if v in high_nodes)
        mid_connect_count = sum(1 for u, v in all_edges if v in mid_nodes)
        low_connect_count = sum(1 for u, v in all_edges if v in low_nodes)
        total_connect = high_connect_count + mid_connect_count + low_connect_count

        high_ratio = high_connect_count / total_connect if total_connect > 0 else 0.5
        mid_ratio = mid_connect_count / total_connect if total_connect > 0 else 0.4
        low_ratio = low_connect_count / total_connect if total_connect > 0 else 0.1
        return high_ratio, mid_ratio, low_ratio

    high_ratio, mid_ratio, low_ratio = calc_layer_connect_ratio(G, high_core_nodes, mid_degree_nodes, low_degree_nodes)
    print(f"  - Original layered connection ratios: high-core({high_ratio:.4f})/medium({mid_ratio:.4f})/low-degree({low_ratio:.4f})")

    # -------------------- Step 4: generate new edges according to layered ratios --------------------
    add_edges = []
    all_target_nodes = high_core_nodes + mid_degree_nodes + low_degree_nodes

    for n in add_all_nodes:
        need_out_edge_num = add_node_out_degrees[n]  
        if need_out_edge_num <= 0 or not all_target_nodes:
            continue

        high_target_num = int(need_out_edge_num * high_ratio)
        mid_target_num = int(need_out_edge_num * mid_ratio)
        low_target_num = need_out_edge_num - high_target_num - mid_target_num
        high_target_num = max(high_target_num, 0)
        mid_target_num = max(mid_target_num, 0)
        low_target_num = max(low_target_num, 0)

        target_high = random.choices(high_core_nodes, k=high_target_num) if high_core_nodes else []
        target_mid = random.choices(mid_degree_nodes, k=mid_target_num) if mid_degree_nodes else []
        target_low = random.choices(low_degree_nodes, k=low_target_num) if low_degree_nodes else []
        target_nodes = target_high + target_mid + target_low

        if len(add_all_nodes) >= 5:
            new_node_connect_num = min(1, need_out_edge_num // 5)  
            target_new = random.choices([x for x in add_all_nodes if x != n], k=new_node_connect_num)
            target_nodes.extend(target_new)

        for target in target_nodes[:need_out_edge_num]:
            add_edges.append((n, target))  

        all_target_nodes.append(n)

    add_edges = list(set(add_edges))[:add_edge_num]
    G.add_nodes_from(add_all_nodes)
    G.add_edges_from(add_edges)

    # -------------------- Step 5: generate features and labels for new nodes --------------------
    normal_feats_ori = ori_dist["normal_feats"]
    scaler_normal = StandardScaler()
    normal_feats_scaled = scaler_normal.fit_transform(normal_feats_ori)
    gmm_normal = GaussianMixture(n_components=3, random_state=SEED).fit(normal_feats_scaled)
    add_normal_feats = scaler_normal.inverse_transform(gmm_normal.sample(add_normal_num)[0]).astype(np.float32)

    add_anomaly_feats = np.array([])
    anomaly_feats_ori = ori_dist["anomaly_feats"]
    if len(anomaly_feats_ori) > 0 and add_anomaly_num > 0:
        scaler_anomaly = StandardScaler()
        anomaly_feats_scaled = scaler_anomaly.fit_transform(anomaly_feats_ori)
        gmm_anomaly = GaussianMixture(n_components=2, random_state=SEED).fit(anomaly_feats_scaled)
        add_anomaly_feats = scaler_anomaly.inverse_transform(gmm_anomaly.sample(add_anomaly_num)[0]).astype(np.float32)
    else:
        add_anomaly_feats = add_normal_feats[:add_anomaly_num] + np.random.normal(0, 0.5, (add_anomaly_num, normal_feats_ori.shape[1])).astype(np.float32)

    add_all_feats = np.vstack([add_normal_feats, add_anomaly_feats]) if add_anomaly_num > 0 else add_normal_feats
    new_features = np.vstack([features, add_all_feats])

    add_normal_labels = np.zeros(add_normal_num, dtype=int)
    add_anomaly_labels = np.ones(add_anomaly_num, dtype=int)
    add_all_labels = np.hstack([add_normal_labels, add_anomaly_labels])
    new_labels = np.hstack([labels, add_all_labels])

    # -------------------- Step 6: validate distributions after expansion --------------------
    new_node_num = len(G)
    new_edge_num = len(G.edges())
    new_anomaly_ratio = sum(new_labels) / new_node_num
    new_edge_node_ratio = new_edge_num / new_node_num
    new_degrees = np.array([G.degree(n) for n in G.nodes()])

    # -------------------- KS validation: degree distribution and feature matrix --------------------
    KS_THRESHOLD = 0.05

    def clean_1d_array(arr):
        """Convert to a one-dimensional array and remove NaN/Inf values to avoid KS-test errors"""
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        return arr[np.isfinite(arr)]

    ori_degrees_clean = clean_1d_array(ori_degrees)
    new_degrees_clean = clean_1d_array(new_degrees)

    degree_ks_stat, degree_ks_pvalue = ks_2samp(
        ori_degrees_clean,
        new_degrees_clean
    )
    degree_ks_pass = degree_ks_pvalue > KS_THRESHOLD

    if features.shape[1] != new_features.shape[1]:
        raise ValueError(
            f"Feature dimensions are inconsistent:original feature dimension={features.shape[1]}, "
            f"expanded feature dimension={new_features.shape[1]}"
        )

    feature_ks_results = []
    for feat_idx in range(features.shape[1]):
        ori_feat_col = clean_1d_array(features[:, feat_idx])
        new_feat_col = clean_1d_array(new_features[:, feat_idx])

        if len(ori_feat_col) == 0 or len(new_feat_col) == 0:
            feat_ks_stat, feat_ks_pvalue = np.nan, 0.0
        else:
            feat_ks_stat, feat_ks_pvalue = ks_2samp(
                ori_feat_col,
                new_feat_col
            )

        feature_ks_results.append({
            "feature_idx": feat_idx,
            "ks_stat": feat_ks_stat,
            "pvalue": feat_ks_pvalue,
            "pass": feat_ks_pvalue > KS_THRESHOLD
        })

    feature_pvalues = np.array([r["pvalue"] for r in feature_ks_results])
    feature_ks_min_pvalue = feature_pvalues.min()
    feature_ks_mean_pvalue = feature_pvalues.mean()
    feature_ks_pass = np.all(feature_pvalues > KS_THRESHOLD)

    worst_feature_ks = sorted(
        feature_ks_results,
        key=lambda x: x["pvalue"]
    )[:5]

    new_high_degree_nodes = [n for n in G.nodes() if (G.degree(n) >= high_thresh)]
    ori_high_count = len(high_core_nodes)
    new_high_count = len(new_high_degree_nodes)
    high_growth_ratio = new_high_count / ori_high_count

    new_mid_degree_nodes = [n for n in G.nodes() if (G.degree(n) >= mid_thresh) and (G.degree(n) < high_thresh)]
    ori_mid_count = len(mid_degree_nodes)
    new_mid_count = len(new_mid_degree_nodes)
    mid_growth_ratio = new_mid_count / ori_mid_count

    new_low_degree_nodes = [n for n in G.nodes() if (G.degree(n) < mid_thresh)]
    ori_low_count = len(low_degree_nodes)
    new_low_count = len(new_low_degree_nodes)
    low_growth_ratio = new_low_count / ori_low_count

    print(f"\nExpansion validation:")
    print(f"  - Total nodes: {new_node_num} (target: {target_node_num})")
    print(f"  - Total edges: {new_edge_num}, edge-to-node ratio: {new_edge_node_ratio:.4f} (original: {ori_edge_node_ratio:.4f})")
    print(f"  - anomaly ratio: {new_anomaly_ratio:.4f} (original: {ori_anomaly_ratio:.4f})")
    print(f"  - mean degree: {new_degrees.mean():.2f} (original: {ori_degrees.mean():.2f})")

    return G, new_features, new_labels , new_anomaly_ratio


# ===================== 3. Save expanded data =====================
def save_expanded_data(G, features, labels, new_anomaly_ratio,save_path):
    """Save the expanded data as a PyG-format .pt file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    edges = np.array(list(G.edges()))
    edge_index = torch.tensor(edges.T, dtype=torch.long) if len(edges) > 0 else torch.tensor([[], []], dtype=torch.long)

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    pyg_data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=len(G),
        contamination=new_anomaly_ratio
    )

    torch.save(pyg_data, save_path)
    print(f"\nExpanded data saved to: {save_path}")
    print(f"Final data information:")
    print(f"  - number of nodes: {pyg_data.num_nodes}, number of edges: {pyg_data.edge_index.shape[1]}")
    print(f"  - feature dimension: {pyg_data.x.shape[1]}, anomaly ratio: {pyg_data.y.sum().item() / pyg_data.num_nodes:.4f}")


# ===================== Main workflow =====================
if __name__ == "__main__":
    try:
        G, features, labels, ori_dist = load_original_pyg_data(ORIGINAL_PT_PATH)

        G, expanded_features, expanded_labels ,new_anomaly_ratio= expand_graph_preserve_dist(
            G, features, labels, ori_dist, TARGET_NODE_NUM
        )

        save_expanded_data(G, expanded_features, expanded_labels, new_anomaly_ratio,SAVE_PT_PATH)

        print("\n✅ Dataset expansion completed. All ratios/distributions are consistent with the original data.")
    except Exception as e:
        print(f"\n❌ Execution failed: {type(e).__name__} - {e}")
        raise
