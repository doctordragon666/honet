"""高阶网络社区划分

不仅支持划分社区，更提供可视化社区和比较社团划分的差异
"""

from enum import Enum, unique, auto
from .third import np, hnx
from hypernetx.algorithms.hypergraph_modularity import kumar, last_step, modularity, part2dict
from hypergraphx.communities import get_num_hyperlink_communties
from hypergraphx.measures.edge_similarity import jaccard_distance as jaccard
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import distance as ssd
from sklearn.decomposition import NMF
from .core import convert_hypernetx_to_hypergraphx


@unique
class Method(Enum):
    """社区划分方法枚举类

    Attributes
    ----------
    KUMAR : auto()
        Kumar方法
    NMF : auto()
        非负矩阵分解方法
    Random : auto()
        随机方法
    LAST_STEP : auto()
        Last Step方法，基于贪心算法的优化算法
    RECURSIVE : auto()
        递归方法
    Basic : auto()
        转化为基本的图来检测社区
    """
    KUMAR = auto()
    NMF = auto()
    Random = auto()
    LAST_STEP = auto()
    RECURSIVE = auto()
    Basic = auto()
    Hierarchy = auto()


def Random_Method(hnx_g: hnx.Hypergraph, n_components=5):
    """使用随机方法进行高阶网络的社区划分
    """
    node_list = list(hnx_g.nodes)
    np.random.shuffle(node_list)  # 随机打乱节点顺序
    partition = [[] for _ in range(n_components)]
    for idx, node in enumerate(node_list):
        partition[idx % n_components].append(node)  # 将节点均匀分配到社区中
    return partition


def Basic_Method(hnx_g: hnx.Hypergraph) -> list:
    """将高阶网络转化为基本的图，用louvain方法来检测社区
    """
    node_list = list(hnx_g.nodes)
    adj = hnx_g.auxiliary_matrix(s=1, index=False).toarray()
    import networkx as nx
    G = nx.from_numpy_array(adj)
    from networkx.algorithms.community import louvain
    communities = louvain.louvain_communities(G, seed=42)
    # 将 NetworkX 的整数节点编号映射回 HyperNetX 的原始节点标签。
    return [{node_list[idx] for idx in community} for community in communities]


def NMF_Method(hypergraph):
    """
    使用NMF算法进行高阶网络的社区划分
    """
    best_partition = None
    best_qh = -float('inf')
    for n_components in range(3, min(7, len(hypergraph.nodes))):  # 尝试不同的社区数量
        # 将hypergraph转换为邻接矩阵
        adjacency_matrix = hypergraph.incidence_matrix(index=False).toarray()
        # 使用NMF进行社区划分
        model = NMF(n_components=n_components, init='random', random_state=42)
        W = model.fit_transform(np.asarray(adjacency_matrix))
        # 根据W矩阵的最大值索引来确定每个节点所属的社区（返回节点对应的社区标签）
        labels = [int(W[i].argmax()) for i in range(W.shape[0])]
        # 将标签转换为 hypernetx / hmod 期望的分区格式：列表的列表，每个子列表为该社区包含的节点
        node_list = list(hypergraph.nodes)
        # 创建社区容器
        partition = [[] for _ in range(n_components)]
        for idx, lbl in enumerate(labels):
            # 如果节点数>n_components，lbl 仍在 0..n_components-1
            try:
                partition[lbl].append(node_list[idx])
            except IndexError:
                # 如果出现标签越界（极少），将该节点放入最后一个社区
                partition[-1].append(node_list[idx])
        # 删除空社区以避免传递空列表（可选）
        partition = [p for p in partition if p]
        qh = modularity(hypergraph, partition)
        if qh > best_qh:
            best_qh = qh
            best_partition = partition
    return best_partition


def Recursive_Method(hnx_g: hnx.Hypergraph):
    """自定义递归划分算法未实现

    大致思路：如果当前划分的模块度不高，则对每个社区进行递归划分，直到模块度达到一定阈值或无法进一步划分为止。
    原来思路：根据前一年的划分结果，尝试将其作为下一年的初始划分，进行优化。保持社团连续性，如果不能的话，就重新执行last_step算法。

    Parameters
    ----------
    hnx_g : hnx.Hypergraph
        高阶网络
    """
    pass


def Hierarchy_Method(hnx_g: hnx.Hypergraph):
    hgx_g = convert_hypernetx_to_hypergraphx(hnx_g)
    # 获取超边聚类树状图
    lcc = hgx_g.largest_component()
    hgx_g = hgx_g.subhypergraph(lcc)
    h = hgx_g.get_edges()

    adj = {}
    edge_to_id = {}

    cont = 0
    for e in h:
        e = tuple(sorted(e))
        edge_to_id[e] = cont
        for n in e:
            if n not in adj:
                adj[n] = []
            adj[n].append(e)
        cont += 1

    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from([i for i in range(len(h))])

    vis = {}
    c = 0
    for n in adj:
        for i in range(len(adj[n]) - 1):
            for j in range(i + 1, len(adj[n])):
                k = tuple(
                    sorted((edge_to_id[adj[n][i]], edge_to_id[adj[n][j]])))
                e_i = set(adj[n][i])
                e_j = set(adj[n][j])
                if k not in vis:
                    w = jaccard(e_i, e_j)
                    if w > 0:
                        G.add_edge(
                            edge_to_id[adj[n][i]
                                        ], edge_to_id[adj[n][j]], weight=w
                        )
                    vis[k] = True
        c += 1
    X = nx.to_numpy_array(G, weight="weight", nonedge=1.0)
    np.fill_diagonal(X, 0.0)
    dist_matrix = ssd.squareform(X)
    Z = linkage(dist_matrix, method="average")
    # 自动估计社团数量
    num_communities = get_num_hyperlink_communties(Z, len(h))
    # 根据估计的社团数量进行划分
    labels = fcluster(Z, num_communities, criterion='maxclust')
    # 将标签转换为 hypernetx / hmod 期望的分区格式
    node_list = list(hnx_g.nodes)
    partition = [[] for _ in range(num_communities)]
    for idx, lbl in enumerate(labels):
        partition[lbl - 1].append(node_list[idx])  # fcluster返回的标签从1开始
    return partition


def check_partition(hypergraph, prev_partition):
    """检查社区划分结果能否用于该网络

    Parameters
    ----------
    hypergraph : Hypergraph
        当前的高阶网络对象
    prev_partition : dict
        之前的社区划分结果，格式为 {node: community}

    Returns
    -------
    list
        返回不在当前网络中的节点列表，如果不为空，则表示这些节点在当前网络中不存在，无法用于社区划分
    """
    # 获取当前网络中的节点
    current_node_list = list(hypergraph.nodes)
    current_nodes = set(current_node_list)
    # 过滤掉不在当前网络中的节点
    if isinstance(prev_partition, dict):
        filtered_partition = {
            node: community
            for node, community in prev_partition.items()
            if node in current_nodes
        }

        missing_nodes = [
            node for node in current_node_list if node not in filtered_partition]
        return missing_nodes
    print("Warning: prev_partition is not a dictionary. Returning empty list.")
    return []  # 如果prev_partition不是字典，返回空列表


def extract_community_partition(hnx_g: hnx.Hypergraph, method=Method.KUMAR, only_qh=False) -> dict | float:
    """计算高阶网络的社区划分结果

    Parameters
    ----------
    H : Hypergraph
        输入的超图对象
    method : Method, optional
        使用的社区划分方法，默认为Kumar方法
    only_qh : bool, optional
        是否只返回模块度，默认为False

    Returns
    -------
    dict or float
        如果only_qh为True，则返回模块度；否则返回一个包含模块度、社区划分和节点映射的字典

    Raises
    ------
    NotImplementedError
        如果指定的方法未实现，则抛出此异常
    """
    if method == Method.KUMAR:
        result = kumar(hnx_g)
    elif method == Method.LAST_STEP:
        result = last_step(hnx_g, kumar(hnx_g))
    elif method == Method.Random:
        result = Random_Method(hnx_g, n_components=5)
    elif method == Method.Basic:
        result = Basic_Method(hnx_g)
    elif method == Method.NMF:
        result = NMF_Method(hnx_g)
    elif method == Method.RECURSIVE:
        result = Recursive_Method(hnx_g)
    elif method == Method.Hierarchy:
        result = Hierarchy_Method(hnx_g)
    else:
        raise NotImplementedError('Method not implemented')
    qh = modularity(hnx_g, result)
    if only_qh:
        return qh
    return {
        'qh': qh,
        'communities': result,
        'nodemap': part2dict(result)
    }


def compare_methods(hnx_g: hnx.Hypergraph, methods=list[Method], compare_fun=None) -> dict:
    """比较不同社区划分方法的优劣
    """
    result_dict = {}
    if compare_fun is None:
        # 默认比较模块度
        for method in methods:
            result_dict[method] = extract_community_partition(
                hnx_g, method=method, only_qh=True)
        return result_dict
    else:
        for method in methods:
            result_dict[method] = compare_fun(hnx_g, method)
        return result_dict


def compare_graphs(H_lst, method=Method.KUMAR) -> dict:
    """比较不同高阶网络的社区划分结果

    Parameters
    ----------
    H_lst : list
        高阶网络列表
    method : Method, optional
        使用的社区划分方法，默认为Kumar方法
    """
    result_dict = {}
    for H in H_lst:
        result_dict[H] = extract_community_partition(H, method=method)
    return result_dict
