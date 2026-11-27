"""高阶网络的全局指标

"""

from hypergraphx.generation import configuration_model
from hypergraphx import Hypergraph
from scipy.stats import rankdata
from . import *
from .nodemetric import clustering_coefficient
from hypergraphx.measures.reducibility import reducibility
import numpy as np
from scipy.stats import rankdata, pearsonr


def calc_reducibility(H_hnx, no_layer=True, weighted=False) -> float:
    """计算超图的可约性

    Parameters
    ----------
    hg : Hypergraph
        输入的超图对象，包含节点和边的信息

    no_layer : bool, optional
        是否只计算单层超图的可约性，默认为True

    Returns
    -------
    float
        超图的可约性值
    """
    # 如果不为hypergraphx的超图对象
    if isinstance(H_hnx, hnx.Hypergraph):
        # 将HyperNetX的超图对象转换为HyperGraphX的超图对象
        H_hnx = convert_hypernetx_to_hypergraphx(H_hnx)
    # 计算可约性
    if weighted:
        reducibility_value = sub_reducibility(
            H_hnx, optimization='greedy')
        if no_layer:
            reducibility_value = reducibility_value[0]
    else:
        if no_layer:
            reducibility_value = reducibility(H_hnx, optimization='greedy')[0]
        else:
            reducibility_value = reducibility(H_hnx, optimization='greedy')
    return reducibility_value

import math

import numpy as np

from hypergraphx import Hypergraph
from hypergraphx.exceptions import InvalidParameterError
from hypergraphx.measures.reducibility import (
    _get_entropies_count,
    _get_entropies_project,
    _get_layers,
    _powerset,
)


def _layer_weight_costs(hg: Hypergraph, layers):
    """
    计算每个层的「权重异质性成本」 W_l。

    输入
    ----
    hg     : Hypergraph
    layers : {size: set(edge_keys)}（来自 _get_layers）

    输出
    ----
    dict
        {size: W_l}，W_l = E_l * [ln E_l - H(p)] >= 0，仅当层内权重全相等时取 0。
        采用自然对数，与 _get_entropies_* 内部的 ln 单位一致。
    """
    costs = {}
    for size, edges in layers.items():
        edges = list(edges)
        E = len(edges)
        if E <= 1:                       # 0/1 条边无从谈异质性
            costs[size] = 0.0
            continue
        weights = [float(hg.get_weight(e)) for e in edges]
        total = sum(weights)
        if total <= 0.0:
            costs[size] = 0.0
            continue
        p = [w / total for w in weights]          # 层内归一化权重
        H = -sum(pi * math.log(pi) for pi in p if pi > 0.0)  # 权重分布熵
        costs[size] = E * (math.log(E) - H)       # 最大熵与实际熵之差 × 边数
    return costs


def sub_reducibility(
    hg: Hypergraph, partition=None, optimization="exact", entropy_method="count"
):
    """
    计算（加权）超图的可约性及代表层集合（签名与官方一致）。

    输入
    ----
    hg : Hypergraph（加权则权重自动参与；无权则权重全 1，退化为官方结果）
    partition : list / dict / None（节点分区，多尺度可约性）
    optimization : {"exact", "greedy"}
    entropy_method : {"count", "project"}

    输出
    ----
    tuple
        (reducibility, representative_layers)，可约性 ∈ [0, 1]。

    参考文献
    --------
    Kirkley, A., Felippe, H., & Battiston, F. PRL 135, 247401 (2025).
    """
    if entropy_method not in {"count", "project"}:
        raise InvalidParameterError('entropy_method must be either "count" or "project".')
    if optimization not in {"exact", "greedy"}:
        raise InvalidParameterError('optimization must be either "exact" or "greedy".')

    # 1) 复用官方无权熵（层自身成本 Q 与层间条件成本 M）
    if entropy_method == "project":
        M, Q = _get_entropies_project(hg, partition)
    else:
        M, Q = _get_entropies_count(hg, partition)

    # 2) 加权修正：把各层权重异质性成本 W_l 加法叠加到 Q 与 M 上
    wcosts = _layer_weight_costs(hg, _get_layers(hg))
    for l in Q:
        wl = wcosts.get(l, 0.0)
        Q[l] = Q[l] + wl
        for k in M:                      # M[k][l] 仅对 k > l 存在
            if k > l:
                M[k][l] += wl

    max_layer = max(list(Q.keys()))

    # 3) 最小化总信息成本寻找代表层集合（与官方逻辑一致）
    if optimization == "exact":
        DLs = {}
        for reps in _powerset(list(Q.keys())):
            reps = set(reps)
            if max_layer not in reps:
                continue
            H = sum(Q[r] for r in reps)
            for layer in set(Q.keys()) - reps:
                higher = [r for r in reps if r > layer]
                H += M[higher[int(np.argmin([M[r][layer] for r in higher]))]][layer]
            DLs[tuple(sorted(reps))] = H
        Rstar = min(DLs, key=DLs.get)
        Hstar = DLs[Rstar]
    else:
        Rs, Hs = [], []
        reps = {max_layer}
        remaining = set(Q.keys()) - reps
        Rs.append(list(reps))
        Hs.append(Q[max_layer] + sum(M[max_layer][l] for l in remaining))
        while remaining:
            cand_layers, cand_scores = [], []
            for candidate in remaining:
                reps_tmp = reps | {candidate}
                non_reps = remaining - {candidate}
                H_tmp = sum(Q[r] for r in reps_tmp)
                for layer in non_reps:
                    higher = [r for r in reps_tmp if r > layer]
                    H_tmp += M[higher[int(np.argmin([M[r][layer] for r in higher]))]][layer]
                cand_layers.append(candidate)
                cand_scores.append(H_tmp)
            best_index = int(np.argmin(cand_scores))
            reps.add(cand_layers[best_index])
            remaining.remove(cand_layers[best_index])
            Rs.append(list(reps))
            Hs.append(cand_scores[best_index])
        best_num_reps = int(np.argmin(Hs))
        Rstar = tuple(sorted(Rs[best_num_reps]))
        Hstar = Hs[best_num_reps]

    # 4) 归一化为 [0,1]
    H0 = sum(Q[l] for l in Q)
    if H0 == Q[max_layer]:
        return 1.0, Rstar
    return (H0 - Hstar) / (H0 - Q[max_layer]), Rstar


def calc_entropy(H_hnx: hnx.Hypergraph) -> float:
    """计算高阶网络的熵

    Parameters
    ----------
    g: hypergraph
        通过honbuild接口生成的单个超图对象

    Returns
    -------
    float
        超图的熵值
    """
    degree = np.array(hnx.dist_stats(H_hnx)['node degree list'])
    total = sum(degree)
    return 0 if total == 0 else -sum([p * np.log(p) for p in degree / total])


# def calc_assortativity(H_hnx):
#     """计算超图的同配系数
#     """
#     if not isinstance(H_hnx, hgx.Hypergraph):
#         H_hnx = convert_hypernetx_to_hypergraphx(H_hnx)
#     # 计算同配性（基于超图的度相关性）
#     nodes = list(H_hnx.nodes)
#     if len(nodes) < 2:
#         return 0.0

#     # 获取所有节点的度
#     degrees = {node: H_hnx.degree(node) for node in nodes}

#     # 计算平均度
#     avg_degree = sum(degrees.values()) / len(degrees)

#     # 获取所有边
#     all_edges = H_hnx.get_edges()

#     # 计算度-度相关性（使用邻居平均度方法）
#     total_weight = 0.0
#     total_degree_product = 0.0
#     total_degree_squared = 0.0

#     for node in nodes:
#         neighbors = set()
#         # 获取节点所在的所有边
#         incident_edges = H_hnx.get_incident_edges(node)
#         for edge in incident_edges:
#             # 将边中的其他节点添加到邻居集合
#             for neighbor in edge:
#                 if neighbor != node:
#                     neighbors.add(neighbor)

#         # 计算邻居的平均度
#         if len(neighbors) > 0:
#             neighbor_avg_degree = sum(degrees[neighbor]
#                                       for neighbor in neighbors) / len(neighbors)
#             total_weight += degrees[node]
#             total_degree_product += degrees[node] * neighbor_avg_degree
#             total_degree_squared += degrees[node] ** 2

#     if total_weight > 0 and total_degree_squared > 0:
#         # 计算同配性系数
#         numerator = total_degree_product - total_weight * avg_degree
#         denominator = total_degree_squared - total_weight * avg_degree

#         if denominator != 0:
#             assortativity_value = numerator / denominator
#         else:
#             assortativity_value = 0.0
#     else:
#         assortativity_value = 0.0

#     return assortativity_value


def _assortativity(H):
    """
    按公式 (5) 计算单个超图（时间片 t）的高阶同配系数 R_t。

    定义：
        a_i^t = rank_avg(k_i^t)，即节点 i 在超图 t 内超度的平均秩（全局排序，并列取平均秩）。
        对每条超边 r ∈ E^t：w_top(r) 取超度最大的节点，w_bottom(r) 取超度第二大的节点，
        若并列取节点编号最小者。则
            R_t = Cov(a_{w_top}, a_{w_bottom}) / (sd(a_{w_top}) · sd(a_{w_bottom}))
        即对所有超边取的皮尔逊相关系数；分母为 0 时约定 R_t = 0。

    Parameters
    ----------
    H : Hypergraph
        时间片 t 的超图。

    Returns
    -------
    float
        同配系数 R_t。
    """
    nodes = H.get_nodes()                                   # 该时间片全部节点
    deg = {v: H.degree(v) for v in nodes}                   # 超度 k_i^t
    # 平均秩 a_i^t
    rank = dict(zip(nodes, rankdata([deg[v]
                for v in nodes], method='average')))
    x, y = [], []                                           # w_top / w_bottom 对应的秩序列
    for e in H.get_edges():
        if len(e) < 2:                                      # 单节点边无“第二大”，跳过
            continue
        u, v = sorted(e, key=lambda w: (-deg[w], w))[:2]    # 超度最大/第二大，并列取编号最小
        x.append(rank[u])
        y.append(rank[v])
    if len(x) < 2 or np.std(x, ddof=0) == 0 or np.std(y, ddof=0) == 0:
        return 0.0                                          # 分母为 0 时 R_t = 0
    return float(np.corrcoef(x, y)[0, 1])


def calc_assortativity(H_hnx, n_rand=1000, n_steps=1000, label='edge', seed=42):
    """
    计算某个时间片超图 H 的高阶同配系数 R_t 及其相对配置模型零模型的 z 分数。

    Parameters
    ----------
    H : hypergraphx.Hypergraph
        时间片 t 的超图。
    n_rand : int
        零模型抽样次数（文献 ref33 的随机化次数）。
    n_steps : int
        每个零模型的配置模型 MCMC 步数。
    label : {'edge', 'vertex'}
        配置模型标签空间，'edge' 较快（推荐）。
    seed : int, optional
        随机种子，用于结果复现。

    Returns
    -------
    (r, z) : (float, float)
        r 为同配系数 R_t；z = (r - mu_null) / sigma_null 为相对零模型的 z 分数，
        零分布退化（标准差为 0）时 z 返回 nan。
    """
    if not isinstance(H_hnx, hgx.Hypergraph):
        H = convert_hypernetx_to_hypergraphx(H_hnx)
    else:
        H = H_hnx
    r_obs = _assortativity(H)
    rng = np.random.default_rng(seed)                       # 从种子派生各零模型种子
    nulls = np.asarray([
        _assortativity(
            configuration_model(H, n_steps=n_steps, label=label,
                                seed=int(rng.integers(0, 2**32 - 1)))
        ) for _ in range(n_rand)
    ], dtype=float)
    mu, sigma = nulls.mean(), nulls.std(ddof=0)             # 零分布均值与标准差
    z = float('nan') if sigma == 0 else float((r_obs - mu) / sigma)
    
    # ---------- 第 3 步：双侧经验 p 值 → 显著性判定 ----------
    n = nulls.size                                          # 零分布样本数
    alpha = 0.01                                              # 显著性水平
    # 用 (计数+1)/(n+1) 平滑，规避 p=0 与有限样本偏小的问题
    p_ge = (np.sum(nulls >= r_obs) + 1.0) / (n + 1.0)       # P(零分布 ≥ 观测值) 的估计
    p_le = (np.sum(nulls <= r_obs) + 1.0) / (n + 1.0)       # P(零分布 ≤ 观测值) 的估计
    p = float(min(1.0, 2.0 * min(p_ge, p_le)))              # 双侧 p 值，上限为 1

    significant = bool(p < alpha)      
    return float(r_obs), p


def calc_clustering_coefficient_g(H_hnx: hnx.Hypergraph) -> float:
    """计算超图的全局聚类系数
    """
    # 对每个节点求聚类系数然后取平均
    clustering_dict = clustering_coefficient(H_hnx)
    if len(clustering_dict) == 0:
        return 0.0
    return np.mean(list(clustering_dict.values()))


def calc_natural_connectivity(H_hnx: hnx.Hypergraph) -> float:
    """
    计算自然连通度

    Parameters
    ----------
    hg : hnx.Hypergraph
        超图对象

    Returns
    -------
    float
        自然连通度值
    """
    # 检查超图是否为空
    if not H_hnx.nodes:
        return 0.0
    eigvals = np.linalg.eigvalsh(H_hnx.adjacency_matrix().todense())
    return float(np.log(np.mean(np.exp(eigvals))))
