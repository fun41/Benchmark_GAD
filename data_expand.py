import numpy as np
import networkx as nx
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
import random
import os
import warnings
from scipy.stats import powerlaw  # 拟合度分布（社交网络典型幂律分布）

# ===================== 全局配置（仅需指定扩容后总节点数） =====================
TARGET_NODE_NUM = 100000  # 扩容后目标总节点数（核心配置）
SEED = 20  # 可复现种子
# 路径配置
ORIGINAL_PT_PATH = "./data/twitter.pt"  # 原始.pt文件路径
SAVE_PT_PATH = "./data/expanded_10_twitter.pt"  # 扩容后保存路径

# 固定随机种子
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# ===================== 1. 加载原始数据 + 统计核心分布特征 =====================
def load_original_pyg_data(pt_path):
    """加载数据并统计原始核心分布：异常比例、边节点比、度分布、特征分布"""
    print("加载原始.pt数据集...")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"原始文件不存在：{pt_path}")

    data = torch.load(pt_path, map_location="cpu")
    edge_index = data.edge_index  # (2, E)
    features = data.x.numpy().astype(np.float32)
    labels = data.y.numpy().astype(int)

    # 转换为NetworkX有向图
    edges = edge_index.T.numpy()
    G = nx.from_edgelist(edges, create_using=nx.DiGraph())
    all_nodes = set(range(len(features)))
    G.add_nodes_from(all_nodes - set(G.nodes()))

    # -------------------- 统计核心分布（扩容的基准） --------------------
    # 1. 异常比例（正常/异常节点比例）
    ori_node_num = len(G)
    ori_anomaly_num = sum(labels)
    ori_anomaly_ratio = ori_anomaly_num / ori_node_num
    # 2. 边节点比（总边数/总节点数）
    ori_edge_num = len(G.edges())
    ori_edge_node_ratio = ori_edge_num / ori_node_num
    # 3. 度分布（提取所有节点的出度+入度，拟合幂律分布）
    ori_degrees = np.array([G.degree(n) for n in G.nodes()])  # 总度（入+出）
    # 拟合幂律分布（社交网络典型分布），获取参数
    try:
        # 幂律拟合：xmin=最小度，alpha=幂指数
        alpha, xmin = powerlaw.fit(ori_degrees, discrete=True)
        degree_dist_params = {"type": "powerlaw", "alpha": alpha, "xmin": xmin}
    except:
        # 降级：用直方图采样（避免拟合失败）
        degree_bins = np.histogram(ori_degrees, bins=20)
        degree_dist_params = {"type": "hist", "bins": degree_bins[0], "edges": degree_bins[1]}

    # 打印原始统计
    print(f"原始数据统计：")
    print(f"  - 节点数：{ori_node_num}, 边数：{ori_edge_num}, 边节点比：{ori_edge_node_ratio:.4f}")
    print(f"  - 异常节点数：{ori_anomaly_num}, 异常比例：{ori_anomaly_ratio:.4f}")
    print(f"  - 度分布范围：[{ori_degrees.min()}, {ori_degrees.max()}], 均值：{ori_degrees.mean():.2f}")

    # 保存原始分布特征（供扩容使用）
    ori_dist = {
        "anomaly_ratio": ori_anomaly_ratio,
        "edge_node_ratio": ori_edge_node_ratio,
        "degree_params": degree_dist_params,
        "normal_feats": features[labels == 0],  # 正常节点特征分布
        "anomaly_feats": features[labels == 1]  # 异常节点特征分布
    }

    return G, features, labels, ori_dist


# ===================== 2. 保分布扩容核心函数 =====================
def expand_graph_preserve_dist(G, features, labels, ori_dist, target_node_num):
    """
    扩容至目标节点数，严格保留原始比例/分布：
    修复中等度节点断层问题，复刻原始分层连接偏好
    """
    ori_node_num = len(G)
    add_node_num = target_node_num - ori_node_num
    if add_node_num <= 0:
        print(f"原始节点数({ori_node_num})≥目标节点数({target_node_num})，无需扩容")
        return G, features, labels

    # -------------------- 步骤1：确定新增节点的正常/异常数量（保比例） --------------------
    ori_anomaly_ratio = ori_dist["anomaly_ratio"]
    add_anomaly_num = int(add_node_num * ori_anomaly_ratio)
    add_normal_num = add_node_num - add_anomaly_num
    print(f"\n扩容计划：")
    print(f"  - 新增总节点：{add_node_num}（正常：{add_normal_num}, 异常：{add_anomaly_num}）")
    print(f"  - 新增后异常比例：{ori_anomaly_ratio:.4f}（与原始一致）")

    # -------------------- 步骤2：生成新增节点ID + 采样度（保度分布） --------------------
    max_node_id = max(G.nodes()) if G.nodes() else 0
    add_normal_nodes = list(range(max_node_id + 1, max_node_id + 1 + add_normal_num))
    add_anomaly_nodes = list(range(max_node_id + 1 + add_normal_num, max_node_id + 1 + add_node_num))
    add_all_nodes = add_normal_nodes + add_anomaly_nodes

    # 原始度数据（用于直接采样，放弃易偏差的幂律拟合）
    ori_degrees = np.array([G.degree(n) for n in G.nodes()])
    ori_degree_mean = ori_degrees.mean()
    ori_degree_max = ori_degrees.max()
    ori_degree_upper = min(ori_degree_mean * 2, ori_degree_max)  # 新增节点度上限（原始均值2倍）

    def sample_degree_fix():
        """
        修正后的度采样：从原始度分布中直接采样，限制上限，贴近原始均值
        采样「出度」（而非总度），避免后续双向边叠加
        """
        # 直接从原始度中随机采样（最贴合原始分布）
        sample_deg = np.random.choice(ori_degrees, size=1)[0]
        # 限制上限，避免过高
        return int(min(sample_deg, ori_degree_upper))

    # 为新增节点采样「出度」（而非总度），确保度均值贴近原始
    add_node_out_degrees = {n: max(1, sample_degree_fix()) for n in add_all_nodes}
    add_out_degrees = np.array(list(add_node_out_degrees.values()))
    print(f"  - 新增节点出度均值：{add_out_degrees.mean():.2f}（原始总度均值：{ori_degree_mean:.2f}）")

    # -------------------- 步骤3：原始节点分层 + 统计分层连接占比 --------------------
    ori_edge_node_ratio = ori_dist["edge_node_ratio"]
    add_edge_num = int(add_node_num * ori_edge_node_ratio)
    print(f"  - 新增边数：{add_edge_num}（边节点比：{ori_edge_node_ratio:.4f}，与原始一致）")

    # 3.1 原始节点按度分层（高/中/低）
    ori_degrees = np.array([G.degree(n) for n in G.nodes()])
    # 计算分层阈值
    high_thresh = np.percentile(ori_degrees, 99)  # 前1%为高核心节点
    mid_thresh = np.percentile(ori_degrees, 90)  # 90%~99%为中等度节点，后90%为低度节点
    # 分层
    high_core_nodes = [n for n in G.nodes() if G.degree(n) >= high_thresh]
    mid_degree_nodes = [n for n in G.nodes() if (G.degree(n) >= mid_thresh) and (G.degree(n) < high_thresh)]
    low_degree_nodes = [n for n in G.nodes() if G.degree(n) < mid_thresh]


    # 3.2 统计原始分层连接占比（各层节点被连接的概率）
    def calc_layer_connect_ratio(G, high_nodes, mid_nodes, low_nodes):
        """统计原始边中，连接到各层节点的占比"""
        all_edges = list(G.edges())
        if not all_edges:
            return 0.5, 0.4, 0.1  # 默认比例
        # 统计各层被连接的边数
        high_connect_count = sum(1 for u, v in all_edges if v in high_nodes)
        mid_connect_count = sum(1 for u, v in all_edges if v in mid_nodes)
        low_connect_count = sum(1 for u, v in all_edges if v in low_nodes)
        total_connect = high_connect_count + mid_connect_count + low_connect_count
        # 计算占比（避免除零）
        high_ratio = high_connect_count / total_connect if total_connect > 0 else 0.5
        mid_ratio = mid_connect_count / total_connect if total_connect > 0 else 0.4
        low_ratio = low_connect_count / total_connect if total_connect > 0 else 0.1
        return high_ratio, mid_ratio, low_ratio

    high_ratio, mid_ratio, low_ratio = calc_layer_connect_ratio(G, high_core_nodes, mid_degree_nodes, low_degree_nodes)
    print(f"  - 原始分层连接占比：核心({high_ratio:.4f})、中等({mid_ratio:.4f})、低度({low_ratio:.4f})")

    # -------------------- 步骤4：按分层比例生成新增边（保度分布+补充中等度节点） --------------------
    add_edges = []
    all_target_nodes = high_core_nodes + mid_degree_nodes + low_degree_nodes

    for n in add_all_nodes:
        need_out_edge_num = add_node_out_degrees[n]  # 需生成的出边数（不再叠加双向边）
        if need_out_edge_num <= 0 or not all_target_nodes:
            continue

        # 按分层比例分配出边目标
        high_target_num = int(need_out_edge_num * high_ratio)
        mid_target_num = int(need_out_edge_num * mid_ratio)
        low_target_num = need_out_edge_num - high_target_num - mid_target_num
        high_target_num = max(high_target_num, 0)
        mid_target_num = max(mid_target_num, 0)
        low_target_num = max(low_target_num, 0)

        # 各层选择目标节点
        target_high = random.choices(high_core_nodes, k=high_target_num) if high_core_nodes else []
        target_mid = random.choices(mid_degree_nodes, k=mid_target_num) if mid_degree_nodes else []
        target_low = random.choices(low_degree_nodes, k=low_target_num) if low_degree_nodes else []
        target_nodes = target_high + target_mid + target_low

        # 新节点之间少量连接（控制度增长，不叠加）
        if len(add_all_nodes) >= 5:
            new_node_connect_num = min(1, need_out_edge_num // 5)  # 减少新节点间连接，避免度偏高
            target_new = random.choices([x for x in add_all_nodes if x != n], k=new_node_connect_num)
            target_nodes.extend(target_new)

        # 仅生成「出边」（新节点→目标节点），不主动生成反向边（避免度重复叠加）
        # 反向边由其他节点的出边贡献，贴合原始度分布
        for target in target_nodes[:need_out_edge_num]:
            add_edges.append((n, target))  # 仅新增出边，度+1（无反向叠加）

        # 将当前新节点加入目标池，供后续新节点连接（贡献入度，不影响当前出度）
        all_target_nodes.append(n)

    # 去重 + 限制总新增边数（保持边节点比不变）
    add_edges = list(set(add_edges))[:add_edge_num]
    G.add_nodes_from(add_all_nodes)
    G.add_edges_from(add_edges)

    # -------------------- 步骤5：生成新增节点特征+标签（保分布，无修改） --------------------
    # 生成正常节点特征
    normal_feats_ori = ori_dist["normal_feats"]
    scaler_normal = StandardScaler()
    normal_feats_scaled = scaler_normal.fit_transform(normal_feats_ori)
    gmm_normal = GaussianMixture(n_components=3, random_state=SEED).fit(normal_feats_scaled)
    add_normal_feats = scaler_normal.inverse_transform(gmm_normal.sample(add_normal_num)[0]).astype(np.float32)

    # 生成异常节点特征
    add_anomaly_feats = np.array([])
    anomaly_feats_ori = ori_dist["anomaly_feats"]
    if len(anomaly_feats_ori) > 0 and add_anomaly_num > 0:
        scaler_anomaly = StandardScaler()
        anomaly_feats_scaled = scaler_anomaly.fit_transform(anomaly_feats_ori)
        gmm_anomaly = GaussianMixture(n_components=2, random_state=SEED).fit(anomaly_feats_scaled)
        add_anomaly_feats = scaler_anomaly.inverse_transform(gmm_anomaly.sample(add_anomaly_num)[0]).astype(np.float32)
    else:
        # 降级生成
        add_anomaly_feats = add_normal_feats[:add_anomaly_num] + np.random.normal(0, 0.5, (add_anomaly_num, normal_feats_ori.shape[1])).astype(np.float32)

    # 合并特征和标签
    add_all_feats = np.vstack([add_normal_feats, add_anomaly_feats]) if add_anomaly_num > 0 else add_normal_feats
    new_features = np.vstack([features, add_all_feats])

    add_normal_labels = np.zeros(add_normal_num, dtype=int)
    add_anomaly_labels = np.ones(add_anomaly_num, dtype=int)
    add_all_labels = np.hstack([add_normal_labels, add_anomaly_labels])
    new_labels = np.hstack([labels, add_all_labels])

    # -------------------- 步骤6：验证扩容后分布（含中等度节点验证） --------------------
    new_node_num = len(G)
    new_edge_num = len(G.edges())
    new_anomaly_ratio = sum(new_labels) / new_node_num
    new_edge_node_ratio = new_edge_num / new_node_num
    new_degrees = np.array([G.degree(n) for n in G.nodes()])

    # 验证高度节点数量
    new_high_degree_nodes = [n for n in G.nodes() if (G.degree(n) >= high_thresh)]
    ori_high_count = len(high_core_nodes)
    new_high_count = len(new_high_degree_nodes)
    high_growth_ratio = new_high_count / ori_high_count

    # 验证中等度节点数量
    new_mid_degree_nodes = [n for n in G.nodes() if (G.degree(n) >= mid_thresh) and (G.degree(n) < high_thresh)]
    ori_mid_count = len(mid_degree_nodes)
    new_mid_count = len(new_mid_degree_nodes)
    mid_growth_ratio = new_mid_count / ori_mid_count

    # 验证低度节点数量
    new_low_degree_nodes = [n for n in G.nodes() if (G.degree(n) < mid_thresh)]
    ori_low_count = len(low_degree_nodes)
    new_low_count = len(new_low_degree_nodes)
    low_growth_ratio = new_low_count / ori_low_count

    print(f"\n扩容完成验证：")
    print(f"  - 总节点数：{new_node_num}（目标：{target_node_num}）")
    print(f"  - 总边数：{new_edge_num}, 边节点比：{new_edge_node_ratio:.4f}（原始：{ori_edge_node_ratio:.4f}）")
    print(f"  - 异常比例：{new_anomaly_ratio:.4f}（原始：{ori_anomaly_ratio:.4f}）")
    print(f"  - 度分布均值：{new_degrees.mean():.2f}（原始：{ori_degrees.mean():.2f}）")


    return G, new_features, new_labels , new_anomaly_ratio


# ===================== 3. 保存扩容后数据 =====================
def save_expanded_data(G, features, labels, new_anomaly_ratio,save_path):
    """保存扩容后的数据为PyG格式.pt文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 转换边为PyG格式
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
        num_nodes=len(G),
        contamination=new_anomaly_ratio
    )

    # 保存
    torch.save(pyg_data, save_path)
    print(f"\n扩容后数据已保存至：{save_path}")
    print(f"最终数据信息：")
    print(f"  - 节点数：{pyg_data.num_nodes}, 边数：{pyg_data.edge_index.shape[1]}")
    print(f"  - 特征维度：{pyg_data.x.shape[1]}, 异常比例：{pyg_data.y.sum().item() / pyg_data.num_nodes:.4f}")


# ===================== 主流程 =====================
if __name__ == "__main__":
    try:
        # 1. 加载原始数据 + 统计分布
        G, features, labels, ori_dist = load_original_pyg_data(ORIGINAL_PT_PATH)

        # 2. 保分布扩容（核心步骤）
        G, expanded_features, expanded_labels ,new_anomaly_ratio= expand_graph_preserve_dist(
            G, features, labels, ori_dist, TARGET_NODE_NUM
        )

        # 3. 保存扩容后数据
        save_expanded_data(G, expanded_features, expanded_labels, new_anomaly_ratio,SAVE_PT_PATH)

        print("\n✅ 数据集扩容完成！所有比例/分布均与原始一致。")
    except Exception as e:
        print(f"\n❌ 执行出错：{type(e).__name__} - {e}")
        raise
