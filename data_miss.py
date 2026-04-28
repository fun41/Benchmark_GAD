import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import os
import random

# ===================== 全局配置（核心：分类型配置缺失+填充策略） =====================
# 1. 缺失比例配置（可独立设置）
NORMAL_NODE_MISS_RATIO = 0.5  # 正常节点的特征缺失比例
ANOMALY_NODE_MISS_RATIO = 0.5  # 异常节点的特征缺失比例
# 2. 填充策略配置（可选："mean"（均值）、"median"（中位数）、"neighbor"（同类邻居））
FILL_STRATEGY = "mean"  # 邻居填充，修复后无NaN
SEED = 20  # 可复现种子
# 路径配置
ORIGINAL_PT_PATH = "./data/credit.pt"  # 原始.pt文件路径
SAVE_PT_PATH = "./data/miss_"+str(int(NORMAL_NODE_MISS_RATIO*100))+"%_"+FILL_STRATEGY+"_credit.pt"  # 填充后保存路径

# 固定随机种子
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ===================== 1. 加载原始数据（修复+优化：转集合+预计算均值，核心修改点1） =====================
def load_original_pyg_data(pt_path):
    """加载原始PyG数据，分离正常/异常节点，统计核心信息"""
    print("加载原始.pt数据集...")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"原始文件不存在：{pt_path}")
    
    # 加载PyG数据
    data = torch.load(pt_path, map_location="cpu")
    edge_index = data.edge_index  # (2, E)
    features = data.x.numpy().astype(np.float32)
    labels = data.y.numpy().astype(int)
    
    # 转换为NetworkX有向图（保留完整拓扑结构）
    edges = edge_index.T.numpy()
    G = nx.from_edgelist(edges, create_using=nx.DiGraph())
    all_nodes = set(range(len(features)))
    G.add_nodes_from(all_nodes - set(G.nodes()))  # 补充孤立节点，保证节点完整
    
    # 分离正常/异常节点索引
    normal_node_indices = np.where(labels == 0)[0]
    anomaly_node_indices = np.where(labels == 1)[0]
    
    # ========== 修复点1：转为集合，邻居筛选O(1)速度+精准 ==========
    normal_node_set = set(normal_node_indices)
    anomaly_node_set = set(anomaly_node_indices)
    
    # ========== 修复点2：预计算所有填充均值，避免循环内重复计算 ==========
    feat_dim = features.shape[1] if len(features.shape) > 1 else 1
    normal_feat_global_mean = np.nanmean(features[normal_node_indices], axis=0) if len(normal_node_indices) >0 else np.zeros(feat_dim)
    anomaly_feat_global_mean = np.nanmean(features[anomaly_node_indices], axis=0) if len(anomaly_node_indices) >0 else np.zeros(feat_dim)
    all_feat_global_mean = np.nanmean(features, axis=0) if len(features) >0 else np.zeros(feat_dim)

    # 统计核心信息
    total_node_num = len(G)
    total_edge_num = len(G.edges())
    anomaly_node_num = sum(labels)
    anomaly_ratio = anomaly_node_num / total_node_num
    original_degrees = np.array([G.degree(n) for n in G.nodes()])
    
    # 打印原始统计
    print(f"原始数据统计：")
    print(f"  - 总节点数：{total_node_num}, 总边数：{total_edge_num}")
    print(f"  - 异常节点数：{anomaly_node_num}, 异常比例：{anomaly_ratio:.4f}")
    print(f"  - 特征维度：{feat_dim}, 度分布均值：{original_degrees.mean():.2f}")
    
    # 保存原始核心信息 【新增集合+预计算均值】
    original_info = {
        "G": G,
        "total_node_num": total_node_num,
        "total_edge_num": total_edge_num,
        "anomaly_ratio": anomaly_ratio,
        "feat_dim": feat_dim,
        "original_degree_mean": original_degrees.mean(),
        "normal_node_indices": normal_node_indices,
        "anomaly_node_indices": anomaly_node_indices,
        "normal_node_set": normal_node_set,    # 新增
        "anomaly_node_set": anomaly_node_set,  # 新增
        "normal_feat_global_mean": normal_feat_global_mean, # 新增
        "anomaly_feat_global_mean": anomaly_feat_global_mean, # 新增
        "all_feat_global_mean": all_feat_global_mean          # 新增
    }
    
    return G, features, labels, original_info

# ===================== 2. 分类型属性缺失+填充核心函数（彻底修复NaN残留，核心修改点2） =====================
def add_attr_missing_and_fill(features, labels, original_info, normal_miss_ratio, anomaly_miss_ratio, fill_strategy):
    """
    步骤1：分类型添加属性缺失
    步骤2：分类型填充缺失值，消除NaN，支持多种填充策略
    【核心修复】neighbor填充彻底无NaN，四级降级+维度校验+全局扫尾
    """
    # 步骤1：边界校验（缺失比例）
    def check_miss_ratio(ratio, name):
        if not (0 <= ratio <= 1):
            print(f"{name}缺失比例({ratio})超出0~1范围，自动修正为0.0（无缺失）")
            return 0.0
        return ratio
    
    normal_miss_ratio = check_miss_ratio(normal_miss_ratio, "正常节点")
    anomaly_miss_ratio = check_miss_ratio(anomaly_miss_ratio, "异常节点")
    
    # 步骤2：复制原始特征，添加缺失（生成NaN）
    modified_features = features.copy().astype(np.float32)
    feat_dim = original_info["feat_dim"]
    G = original_info["G"]
    normal_node_indices = original_info["normal_node_indices"]
    anomaly_node_indices = original_info["anomaly_node_indices"]
    # 取集合，用于邻居筛选
    normal_node_set = original_info["normal_node_set"]
    anomaly_node_set = original_info["anomaly_node_set"]
    # 取预计算的均值
    normal_feat_global_mean = original_info["normal_feat_global_mean"]
    anomaly_feat_global_mean = original_info["anomaly_feat_global_mean"]
    all_feat_global_mean = original_info["all_feat_global_mean"]
    
    print(f"\n属性缺失改造计划：")
    print(f"  - 正常节点：{len(normal_node_indices)}个，目标特征缺失比例：{normal_miss_ratio:.4f}")
    print(f"  - 异常节点：{len(anomaly_node_indices)}个，目标特征缺失比例：{anomaly_miss_ratio:.4f}")
    
    # 记录缺失掩码（哪些位置是缺失的，后续填充后可保留该标记）
    miss_mask = np.zeros_like(modified_features, dtype=bool)
    
    # 步骤3：为正常节点添加缺失并记录掩码
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
    
    # 步骤4：为异常节点添加缺失并记录掩码
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
    
    # 步骤5：分类型填充缺失值（消除NaN，核心修复：彻底解决neighbor填充NaN问题）
    def fill_normal_node_missing(feat, normal_indices, miss_mask, fill_strategy):
        """填充正常节点的缺失值 - 核心修复neighbor填充逻辑"""
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
            # ========== 核心修复：正常节点-四级递进式邻居填充，彻底无NaN ==========
            for node_id in normal_indices:
                node_miss_mask = miss_mask[node_id]
                if not np.any(node_miss_mask): continue # 无缺失跳过
                
                # 1. 筛选同类邻居（正常节点），用集合查询，精准高效
                neighbors = list(G.neighbors(node_id))
                normal_neighbors = [n for n in neighbors if n in normal_node_set]
                
                # 初始化填充值：默认用正常全局均值
                fill_vals = normal_feat_global_mean.copy()
                
                # 一级填充：有同类邻居 → 计算邻居非NaN均值
                if len(normal_neighbors) > 0:
                    neighbor_feat = feat[normal_neighbors]
                    neighbor_mean = np.nanmean(neighbor_feat, axis=0)
                    # 维度级校验：逐个维度判断，NaN则用下一级均值替换
                    for dim in range(feat_dim):
                        if not np.isnan(neighbor_mean[dim]):
                            fill_vals[dim] = neighbor_mean[dim]
                        else:
                            fill_vals[dim] = normal_feat_global_mean[dim]
                
                # 二级填充：邻居均值有NaN → 用全图均值兜底
                for dim in range(feat_dim):
                    if np.isnan(fill_vals[dim]):
                        fill_vals[dim] = all_feat_global_mean[dim]
                
                # 终极填充：所有均值都失效 → 0值兜底，彻底无NaN
                fill_vals = np.nan_to_num(fill_vals, nan=0.0)
                
                # 执行填充
                feat[node_id][node_miss_mask] = fill_vals[node_miss_mask]
        
        return feat
    
    def fill_anomaly_node_missing(feat, anomaly_indices, miss_mask, fill_strategy):
        """填充异常节点的缺失值 - 核心修复neighbor填充逻辑，和正常节点对称"""
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
            # ========== 核心修复：异常节点-四级递进式邻居填充，彻底无NaN ==========
            for node_id in anomaly_indices:
                node_miss_mask = miss_mask[node_id]
                if not np.any(node_miss_mask): continue # 无缺失跳过
                
                # 1. 筛选同类邻居（异常节点），集合查询
                neighbors = list(G.neighbors(node_id))
                anomaly_neighbors = [n for n in neighbors if n in anomaly_node_set]
                
                # 初始化填充值：默认用异常全局均值
                fill_vals = anomaly_feat_global_mean.copy()
                
                # 一级填充：有同类邻居 → 计算邻居非NaN均值
                if len(anomaly_neighbors) > 0:
                    neighbor_feat = feat[anomaly_neighbors]
                    neighbor_mean = np.nanmean(neighbor_feat, axis=0)
                    # 维度级校验：逐个维度判断，NaN则用下一级均值替换
                    for dim in range(feat_dim):
                        if not np.isnan(neighbor_mean[dim]):
                            fill_vals[dim] = neighbor_mean[dim]
                        else:
                            fill_vals[dim] = anomaly_feat_global_mean[dim]
                
                # 二级填充：邻居均值有NaN → 用全图均值兜底
                for dim in range(feat_dim):
                    if np.isnan(fill_vals[dim]):
                        fill_vals[dim] = all_feat_global_mean[dim]
                
                # 终极填充：所有均值都失效 → 0值兜底
                fill_vals = np.nan_to_num(fill_vals, nan=0.0)
                
                # 执行填充
                feat[node_id][node_miss_mask] = fill_vals[node_miss_mask]
        
        return feat
    
    # 执行填充（消除NaN）
    print(f"\n开始填充缺失值，填充策略：{fill_strategy}")
    if len(normal_node_indices) > 0 and normal_miss_ratio > 0:
        modified_features = fill_normal_node_missing(modified_features, normal_node_indices, miss_mask, fill_strategy)
    if len(anomaly_node_indices) > 0 and anomaly_miss_ratio > 0:
        modified_features = fill_anomaly_node_missing(modified_features, anomaly_node_indices, miss_mask, fill_strategy)

    # ========== 终极兜底修复：全局强制NaN清理 双重保险！ ==========
    # 哪怕前面所有逻辑都有疏漏，这一行能彻底清除所有NaN，绝对无残留
    modified_features = np.nan_to_num(modified_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 步骤6：验证填充效果（无NaN，缺失比例达标）
    def calc_actual_miss_ratio(miss_mask, node_indices):
        if len(node_indices) == 0: return 0.0
        node_miss_mask = miss_mask[node_indices]
        total_elem_num = node_miss_mask.size
        miss_elem_num = node_miss_mask.sum()
        return miss_elem_num / total_elem_num if total_elem_num > 0 else 0.0
    
    normal_actual_miss = calc_actual_miss_ratio(miss_mask, normal_node_indices)
    anomaly_actual_miss = calc_actual_miss_ratio(miss_mask, anomaly_node_indices)
    overall_actual_miss = calc_actual_miss_ratio(miss_mask, np.arange(len(modified_features)))
    has_nan = np.isnan(modified_features).any() # 此时一定是False
    
    # 打印验证结果
    print(f"\n属性缺失+填充验证：")
    print(f"  - 正常节点：目标缺失{normal_miss_ratio:.4f}，实际缺失{normal_actual_miss:.4f}")
    print(f"  - 异常节点：目标缺失{anomaly_miss_ratio:.4f}，实际缺失{anomaly_actual_miss:.4f}")
    print(f"  - 整体数据集：实际缺失{overall_actual_miss:.4f}，仍含NaN：{has_nan}") # 修复后必为False
    print(f"\n其他核心属性验证（应与原始一致）：")
    print(f"  - 特征维度：{modified_features.shape[1]}（原始：{feat_dim}）")
    print(f"  - 节点数：{len(modified_features)}（原始：{original_info['total_node_num']}）")
    
    return modified_features, miss_mask

# ===================== 3. 保存改造后数据（无NaN，兼容模型训练） =====================
def save_filled_data(G, modified_features, labels, miss_mask, original_info, save_path):
    """保存填充后的数据（无NaN），附带缺失掩码，兼容各类图模型训练"""
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
    print(f"\n改造后数据已保存至：{save_path}")
    print(f"最终数据信息：")
    print(f"  - 节点数：{pyg_data.num_nodes}, 边数：{pyg_data.edge_index.shape[1]}")
    print(f"  - 特征维度：{pyg_data.x.shape[1]}, 含NaN：{torch.isnan(pyg_data.x).any().item()}") # 必为False
    print(f"  - 异常比例：{pyg_data.y.sum().item() / pyg_data.num_nodes:.4f}（与原始一致）")
    print(f"  - 包含缺失掩码：{pyg_data.attr_miss_mask is not None}")

# ===================== 主流程（缺失+填充，消除NaN影响） =====================
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
        print("\n✅ 分类型属性缺失+填充改造完成！无NaN，可直接用于模型训练。")
    except Exception as e:
        print(f"\n❌ 执行出错：{type(e).__name__} - {e}")
        raise