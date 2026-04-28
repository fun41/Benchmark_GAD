import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import torch
from torch_geometric.data import Data
import os
import random

# ===================== 全局配置（核心：切换三种保留策略） =====================
TARGET_ANOMALY_RATIO = 0.005  # 目标异常比例
SEED = 20  # 可复现种子
ORIGINAL_PT_PATH = "./data/credit.pt"  # 原始.pt文件路径
SAVE_PT_PATH = "./data/ratio_random0.5%_credit.pt"  # 改造后保存路径

# 三种异常节点保留策略（可选："core_cluster" / "edge_cluster" / "random"）
SELECT_METHOD = "random"  # 可切换为"edge_cluster"或"random"

# 固定随机种子
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ===================== 1. 加载原始数据（保留所有核心属性，无修改） =====================
def load_original_pyg_data(pt_path):
    """加载原始PyG数据，转换为NetworkX图，统计核心信息"""
    print("加载原始.pt数据集...")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"原始文件不存在：{pt_path}")
    
    # 加载PyG数据
    data = torch.load(pt_path, map_location="cpu")
    edge_index = data.edge_index  # (2, E)
    features = data.x.numpy().astype(np.float32)
    labels = data.y.numpy().astype(int)
    
    # 转换为NetworkX有向图（保留完整拓扑）
    edges = edge_index.T.numpy()
    G = nx.from_edgelist(edges, create_using=nx.DiGraph())
    all_nodes = set(range(len(features)))
    G.add_nodes_from(all_nodes - set(G.nodes()))  # 补充孤立节点，保证节点完整
    
    # 统计核心信息
    total_node_num = len(G)
    original_anomaly_num = sum(labels)
    original_anomaly_ratio = original_anomaly_num / total_node_num
    original_edge_num = len(G.edges())
    original_degrees = np.array([G.degree(n) for n in G.nodes()])
    
    # 打印原始统计
    print(f"原始数据统计：")
    print(f"  - 总节点数：{total_node_num}, 总边数：{original_edge_num}")
    print(f"  - 异常节点数：{original_anomaly_num}, 异常比例：{original_anomaly_ratio:.4f}")
    print(f"  - 度分布均值：{original_degrees.mean():.2f}")
    
    # 保存原始核心信息（供验证使用）
    original_info = {
        "total_node_num": total_node_num,
        "original_edge_num": original_edge_num,
        "original_degree_mean": original_degrees.mean(),
        "normal_feats": features[labels == 0]  # 正常节点特征（供异常节点特征替换使用）
    }
    
    return G, features, labels, original_info, original_anomaly_ratio

# ===================== 2. 三种策略：筛选待保留/待转化的异常节点 =====================
def select_anomaly_nodes_to_keep(anomaly_indices, anomaly_feats, target_anomaly_num, method, seed=SEED):
    """
    根据三种策略，筛选待保留的异常节点和待转化的异常节点
    :param anomaly_indices: 所有异常节点的索引数组
    :param anomaly_feats: 所有异常节点的特征数组
    :param target_anomaly_num: 目标保留的异常节点数量
    :param method: 保留策略（"core_cluster" / "edge_cluster" / "random"）
    :param seed: 随机种子（保证可复现）
    :return: keep_indices（待保留）, convert_indices（待转化）
    """
    total_anomaly_num = len(anomaly_indices)
    if total_anomaly_num <= target_anomaly_num:
        # 异常节点总数不足目标，全部保留，无需转化
        return anomaly_indices, np.array([])
    
    # 策略1：保留核心簇（样本量最大的聚类簇）
    if method == "core_cluster":
        print("\n--- 策略1：保留核心簇（样本量最大簇） ---")
        # KMeans聚类异常节点
        kmeans = KMeans(n_clusters=3, random_state=seed, n_init=10).fit(anomaly_feats)
        anomaly_cluster_labels = kmeans.labels_
        cluster_sizes = [sum(anomaly_cluster_labels == i) for i in range(3)]
        
        # 筛选核心簇（样本量最大）
        core_cluster_id = np.argmax(cluster_sizes)
        core_keep_mask = (anomaly_cluster_labels == core_cluster_id)
        core_keep_indices = anomaly_indices[core_keep_mask]
        core_keep_num = len(core_keep_indices)
        
        print(f"  聚类簇大小：{cluster_sizes}，核心簇ID：{core_cluster_id}（保留{core_keep_num}个节点）")
        
        # 若核心簇节点数≥目标，从核心簇边缘（距中心最远）筛选待转化
        if core_keep_num >= target_anomaly_num:
            # 计算核心簇内节点到簇中心的距离
            core_center = kmeans.cluster_centers_[core_cluster_id]
            core_feats = anomaly_feats[core_keep_mask]
            core_distances = np.linalg.norm(core_feats - core_center, axis=1)
            # 按距离升序排序（近=核心内核心，远=核心内边缘），保留前target_anomaly_num个
            core_sorted_indices = core_keep_indices[np.argsort(core_distances)]
            keep_indices = core_sorted_indices[:target_anomaly_num]
            convert_indices = np.concatenate([anomaly_indices[~core_keep_mask], core_sorted_indices[target_anomaly_num:]])
        
        # 若核心簇节点数<目标，先保留核心簇，再从其他簇补充（选距各自簇中心最近的）
        else:
            # 先保留核心簇所有节点
            keep_indices = core_keep_indices
            remaining_need_keep = target_anomaly_num - core_keep_num
            
            # 从非核心簇中，筛选距各自簇中心最近的节点（代表性最弱，补充保留）
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
                
                # 按距离升序（近=代表性弱），挑选补充保留
                cluster_distances = np.linalg.norm(cluster_feats - cluster_center, axis=1)
                cluster_sorted_indices = cluster_indices[np.argsort(cluster_distances)]
                pick_num = min(remaining_need_keep, len(cluster_sorted_indices))
                non_core_keep.extend(cluster_sorted_indices[:pick_num])
                remaining_need_keep -= pick_num
                if remaining_need_keep <= 0:
                    break
            
            # 整合保留节点和待转化节点
            keep_indices = np.concatenate([keep_indices, np.array(non_core_keep)])
            convert_indices = np.array([idx for idx in anomaly_indices if idx not in keep_indices])
    
    # 策略2：保留边缘簇（样本量最小的聚类簇）
    elif method == "edge_cluster":
        print("\n--- 策略2：保留边缘簇（样本量最小簇） ---")
        # KMeans聚类异常节点
        kmeans = KMeans(n_clusters=3, random_state=seed, n_init=10).fit(anomaly_feats)
        anomaly_cluster_labels = kmeans.labels_
        cluster_sizes = [sum(anomaly_cluster_labels == i) for i in range(3)]
        
        # 筛选边缘簇（样本量最小）
        edge_cluster_id = np.argmin(cluster_sizes)
        edge_keep_mask = (anomaly_cluster_labels == edge_cluster_id)
        edge_keep_indices = anomaly_indices[edge_keep_mask]
        edge_keep_num = len(edge_keep_indices)
        
        print(f"  聚类簇大小：{cluster_sizes}，边缘簇ID：{edge_cluster_id}（保留{edge_keep_num}个节点）")
        
        # 若边缘簇节点数≥目标，从边缘簇内筛选（距中心最近=边缘内核心，保留；远=边缘内边缘，转化）
        if edge_keep_num >= target_anomaly_num:
            edge_center = kmeans.cluster_centers_[edge_cluster_id]
            edge_feats = anomaly_feats[edge_keep_mask]
            edge_distances = np.linalg.norm(edge_feats - edge_center, axis=1)
            edge_sorted_indices = edge_keep_indices[np.argsort(edge_distances)]
            keep_indices = edge_sorted_indices[:target_anomaly_num]
            convert_indices = np.concatenate([anomaly_indices[~edge_keep_mask], edge_sorted_indices[target_anomaly_num:]])
        
        # 若边缘簇节点数<目标，先保留边缘簇，再从其他簇补充（选距各自簇中心最远的）
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
                
                # 按距离降序（远=非边缘簇的边缘），挑选补充保留
                cluster_distances = np.linalg.norm(cluster_feats - cluster_center, axis=1)
                cluster_sorted_indices = cluster_indices[np.argsort(cluster_distances)[::-1]]
                pick_num = min(remaining_need_keep, len(cluster_sorted_indices))
                non_edge_keep.extend(cluster_sorted_indices[:pick_num])
                remaining_need_keep -= pick_num
                if remaining_need_keep <= 0:
                    break
            
            keep_indices = np.concatenate([keep_indices, np.array(non_edge_keep)])
            convert_indices = np.array([idx for idx in anomaly_indices if idx not in keep_indices])
    
    # 策略3：随机保留异常节点
    elif method == "random":
        print("\n--- 策略3：随机保留异常节点 ---")
        # 随机洗牌异常节点索引，挑选前target_anomaly_num个保留
        np.random.seed(seed)
        shuffled_anomaly_indices = anomaly_indices.copy()
        np.random.shuffle(shuffled_anomaly_indices)
        
        keep_indices = shuffled_anomaly_indices[:target_anomaly_num]
        convert_indices = shuffled_anomaly_indices[target_anomaly_num:]
        print(f"  随机保留{len(keep_indices)}个，随机转化{len(convert_indices)}个")
    
    # 无效策略处理
    else:
        raise ValueError(f"无效的保留策略：{method}，可选：core_cluster / edge_cluster / random")
    
    # 去重+格式整理
    keep_indices = np.unique(keep_indices)
    convert_indices = np.unique(convert_indices)
    
    # 验证保留数量（确保达标）
    if len(keep_indices) < target_anomaly_num and len(convert_indices) > 0:
        # 补充保留部分待转化节点，确保达标
        supplement_num = target_anomaly_num - len(keep_indices)
        supplement_indices = convert_indices[:supplement_num]
        keep_indices = np.concatenate([keep_indices, supplement_indices])
        convert_indices = convert_indices[supplement_num:]
    
    print(f"  最终保留异常节点：{len(keep_indices)}个，待转化异常节点：{len(convert_indices)}个")
    return keep_indices, convert_indices

# ===================== 3. 异常比例减小核心函数（整合三种策略） =====================
def reduce_anomaly_ratio(G, features, labels, original_info, original_anomaly_ratio, target_anomaly_ratio):
    """
    整合三种保留策略，精准减小异常比例至目标值，保留对应策略的异常节点
    """
    total_node_num = original_info["total_node_num"]
    normal_feats_global = original_info["normal_feats"]
    global_normal_mean = np.mean(normal_feats_global, axis=0).astype(np.float32)
    
    # 步骤1：边界判断
    if target_anomaly_ratio >= original_anomaly_ratio:
        print(f"目标异常比例({target_anomaly_ratio:.4f})≥原始异常比例({original_anomaly_ratio:.4f})，无需改造")
        return G, features, labels
    if target_anomaly_ratio < 0:
        print("目标异常比例不能小于0，自动修正为0.001")
        target_anomaly_ratio = 0.001
    
    # 步骤2：计算目标保留的异常节点数量
    target_anomaly_num = int(total_node_num * target_anomaly_ratio)
    original_anomaly_num = sum(labels)
    anomaly_indices = np.where(labels == 1)[0]
    anomaly_feats = features[anomaly_indices]
    
    print(f"\n异常比例改造目标：")
    print(f"  原始异常节点数：{original_anomaly_num}，目标保留异常节点数：{target_anomaly_num}")
    print(f"  需转化异常节点数：{original_anomaly_num - target_anomaly_num}")
    
    # 步骤3：三种策略筛选待保留/待转化节点
    keep_indices, convert_indices = select_anomaly_nodes_to_keep(
        anomaly_indices=anomaly_indices,
        anomaly_feats=anomaly_feats,
        target_anomaly_num=target_anomaly_num,
        method=SELECT_METHOD
    )
    
    if len(convert_indices) <= 0:
        print("无足够异常节点可转化，无法降低异常比例")
        return G, features, labels
    
    # 步骤4：复制数据，执行转化（标签+特征同步正常化）
    modified_features = features.copy().astype(np.float32)
    modified_labels = labels.copy().astype(int)
    
    print("\n--- 开始转化异常节点为正常节点（标签+特征同步正常化） ---")
    for node_id in convert_indices:
        # 标签置为正常（0）
        modified_labels[node_id] = 0
        
        # 特征替换：优先邻居正常均值，降级全局均值
        neighbors = list(G.neighbors(node_id))
        if neighbors:
            normal_neighbors = [n for n in neighbors if labels[n] == 0]  # 用原始标签筛选正常邻居
            if normal_neighbors:
                neighbor_normal_feats = features[normal_neighbors]
                local_normal_mean = np.mean(neighbor_normal_feats, axis=0).astype(np.float32)
                modified_features[node_id] = local_normal_mean
                continue
        
        # 无正常邻居，用全局正常均值填充
        modified_features[node_id] = global_normal_mean
    
    # 步骤5：验证改造后结果
    new_anomaly_num = sum(modified_labels)
    new_anomaly_ratio = new_anomaly_num / total_node_num
    new_edge_num = len(G.edges())
    new_degree_mean = np.array([G.degree(n) for n in G.nodes()]).mean()
    
    print(f"\n--- 改造完成验证 ---")
    print(f"  改造后异常节点数：{new_anomaly_num}，异常比例：{new_anomaly_ratio:.4f}（目标：{target_anomaly_ratio:.4f}）")
    print(f"  总节点数：{total_node_num}（与原始一致），总边数：{new_edge_num}（与原始一致）")
    print(f"  度分布均值：{new_degree_mean:.2f}（原始：{original_info['original_degree_mean']:.2f}）")
    
    return G, modified_features, modified_labels

# ===================== 4. 保存改造后数据（无修改） =====================
def save_modified_data(G, features, labels, save_path):
    """保存改造后的数据为PyG格式.pt文件，保留完整拓扑和特征"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 转换边为PyG格式（保留原始边结构）
    edges = np.array(list(G.edges()))
    edge_index = torch.tensor(edges.T, dtype=torch.long) if len(edges) > 0 else torch.tensor([[], []], dtype=torch.long)
    
    # 转换为Tensor
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    
    # 构建PyG Data对象
    pyg_data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=len(G)
    )
    
    # 保存
    torch.save(pyg_data, save_path)
    print(f"\n改造后数据已保存至：{save_path}")
    print(f"最终数据信息：")
    print(f"  - 节点数：{pyg_data.num_nodes}, 边数：{pyg_data.edge_index.shape[1]}")
    print(f"  - 特征维度：{pyg_data.x.shape[1]}, 最终异常比例：{pyg_data.y.sum().item() / pyg_data.num_nodes:.4f}")

# ===================== 主流程（整合三种策略，可切换） =====================
if __name__ == "__main__":
    try:
        # 1. 加载原始数据
        G, features, labels, original_info, original_anomaly_ratio = load_original_pyg_data(ORIGINAL_PT_PATH)
        
        # 2. 核心步骤：减小异常比例（整合三种保留策略）
        G, modified_features, modified_labels = reduce_anomaly_ratio(
            G, features, labels, original_info, original_anomaly_ratio, TARGET_ANOMALY_RATIO
        )
        
        # 3. 保存改造后数据
        save_modified_data(G, modified_features, modified_labels, SAVE_PT_PATH)
        
        print("\n✅ 异常比例减小改造完成！所有其他核心分布均与原始一致。")
    except Exception as e:
        print(f"\n❌ 执行出错：{type(e).__name__} - {e}")
        raise