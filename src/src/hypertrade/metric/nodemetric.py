"""高阶网络的节点指标

包括聚类系数、余平均度、特征向量中心性、介数中心性。
"""
from . import *


def weighted_degree(H_hnx: hnx.Hypergraph):
    """计算超图中每个节点的加权度。

    Parameters
    ----------
    H_hnx : hypernetx.Hypergraph
        输入的超图对象

    Returns
    -------
    dict
        字典，键为节点名称，值为对应的加权度
    """
    # 1. 获取关联矩阵 (行=节点, 列=边，自动包含构建时 cell_weight_col 的权重)
    B, node_names, _ = H_hnx.incidence_matrix(
        index=True, weights=True)  # 返回稀疏矩阵，行=节点，列=边，值为权重

    # 2. 对行(axis=1)求和，兼容 scipy.sparse 稀疏矩阵和 numpy 稠密矩阵
    weights = np.asarray(B.sum(axis=1)).flatten()

    # 3. 组合成字典返回
    return dict(zip(node_names, weights))


def clustering_coefficient(H_hnx: hnx.Hypergraph):
    """计算超图中每个节点的聚类系数。

    Parameters
    ----------
    H_hnx : hypernetx.Hypergraph
        输入的超图对象

    Returns
    -------
    dict
        字典，键为节点名称，值为对应的聚类系数
    """
    import xgi.algorithms.clustering as xgi_clustering

    # 转换超图
    H_xgi, idx_to_node = convert_hypernetx_to_xgi(H_hnx)

    # 使用 XGI 算法模块计算聚类系数
    clustering_dict = xgi_clustering.clustering_coefficient(H_xgi)

    # 将索引转换回原始节点名称
    result = {}
    for idx, c in clustering_dict.items():
        node_name = idx_to_node[idx]
        result[node_name] = c

    return result


def excess_average_degree(H_hnx):
    r"""计算超图中所有节点的余平均度，定义为节点的邻居节点的平均超度，反映节点邻居节点的情况。

    计算公式如下

    .. math::

        \langle dk_{nn}\rangle_i = \frac{1}{|\tau(i)|}\sum_{j\in\tau(i)}HD(v_j)

    其中 tau(i) 为节点 i 的邻居节点集合，HD(v_j) 为邻居节点 j 的超度。

    Parameters
    ----------
    H_hnx : Hypergraph
        输入的超图对象

    Returns
    -------
    dict
        字典，键为节点名称，值为对应的余平均度值
    """
    result = {}
    nodes = list(H_hnx.nodes)

    incidence_dict = getattr(H_hnx, 'incidence_dict', None)
    if callable(incidence_dict):
        incidence_dict = incidence_dict()

    if not isinstance(incidence_dict, dict):
        raise TypeError(
            'excess_average_degree 需要支持 incidence_dict 的超图对象')

    for node in nodes:
        neighbors = set()
        # 获取节点关联的所有超边
        for edge_nodes in incidence_dict.values():
            if node in edge_nodes:
                # 将超边中除当前节点外的其他节点加入邻居集合
                for n in edge_nodes:
                    if n != node:
                        neighbors.add(n)

        if len(neighbors) == 0:
            result[node] = 0.0
        else:
            total_hyperdegree = 0
            for neighbor in neighbors:
                total_hyperdegree += H_hnx.degree(neighbor)

            result[node] = total_hyperdegree / len(neighbors)

    return result


def eigenvector_centrality(H):
    """计算超图的特征向量中心性（基于归一化拉普拉斯矩阵的Fiedler向量）

    Parameters
    ----------
    H : hnx.Hypergraph
        输入的超图对象

    Returns
    -------
    dict
        字典，键为节点标识，值为对应的特征向量中心性值
    """
    # 获取归一化拉普拉斯矩阵
    # norm_lap 返回一个元组 (稀疏矩阵, 节点标签数组)
    L, nodes = hnx.norm_lap(H)

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(L.todense())

    # 特征向量中心性对应于第二小特征值的特征向量（Fiedler vector）
    sorted_indices = np.argsort(eigenvalues)
    fiedler_vector = np.asarray(
        eigenvectors[:, sorted_indices[1]]).flatten().real

    # 构建结果字典
    centrality_dict = {node: float(
        fiedler_vector[i]) for i, node in enumerate(nodes)}

    return centrality_dict


def betweenness_centrality(H):
    """计算超图的介数中心性。

    Parameters
    ----------
    H : hnx.Hypergraph
        输入的超图对象

    Returns
    -------
    dict
        字典，键为节点标识，值为对应的介数中心性值
    """
    # 使用 NetworkX 计算二部图的介数中心性
    import networkx as nx
    betweenness_dict = nx.betweenness_centrality(hnx.bipartite(H))

    # 仅保留原始节点的介数中心性
    original_nodes = set(H.nodes)
    result = {node: betweenness_dict[node]
              for node in original_nodes if node in betweenness_dict}

    return result
