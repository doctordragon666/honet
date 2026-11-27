"""构造高阶网络

根据数据构造不同的高阶网络，包括单年份、多年份、加权、不加权等
"""

from hypernetx.reports.descriptive_stats import info
from .core import merge_ew
from . import hnx
import random
from .third import np


def build_hypernet(e, w=None, i=None) -> hnx.Hypergraph:
    """构建高阶网络的接口函数，输入年份和是否加权，输出对应的超图对象。

    Parameters
    ----------
    e : pd.Dataframe
        边的Dataframe
    w : _type_, optional
        权重, by default None
    i : _type_, optional
        e和w都有的时间, by default None

    Returns
    -------
    hnx.Hypergraph
        构建的超图对象
    """
    if i is not None:
        e = e[e['year'] == int(i)]
        edge_depth_dict = e.drop_duplicates('agreement').set_index('agreement')[
            'depth'].to_dict()
        edge_properties_dict = {k: {'depth': v} for k, v in edge_depth_dict.items()}
        if w is not None:
            w = w[w['period'] == int(i)]
            e = merge_ew(e, w)
        H_temp = hnx.Hypergraph(e, edge_col='agreement',
                                node_col='country', cell_weight_col='weight' if w is not None else None,
                                edge_properties=edge_properties_dict, edge_weight_prop_col='depth')
        return H_temp
    else:
        if w is not None:
            e = merge_ew(e, w)
        # 单个图构建
        return hnx.Hypergraph(e, edge_col='agreement', node_col='country', cell_weight_col='weight' if w is not None else None, edge_weight_prop_col='depth')


def build_hypernets(e, index, w=None):
    """构建多个高阶网络的接口函数，输入年份和是否加权，输出对应的超图对象。

    Parameters
    ----------
    e : pd.Dataframe
        边的Dataframe
    index : list
        年份列表
    w : pd.Dataframe, optional
        权重的Dataframe，by default None

    Returns
    -------
    dict
        返回一个字典，键为年份，值为对应的超图对象
    """
    return {i: build_hypernet(e, w, i) for i in index}


def generate_random_hypernet(n_nodes: int = 100, n_edges: int = 200, max_edge_size: int = 4, seed: int = 0) -> hnx.Hypergraph:
    """生成随机超图，用于测试和验证算法的性能。

    Parameters
    ----------
    n_nodes : int
        随机超图的节点数
    n_edges : int
        随机超图的边数
    max_edge_size : int
        随机超图的最大边大小
    seed : int, optional
        随机种子, by default 0

    Returns
    -------
    hnx.Hypergraph
        随机超图
    """
    if seed is not None:
        import random
        random.seed(seed)

    hyperedge_list = []
    for _ in range(n_edges):
        edge_size = random.randint(2, max_edge_size)
        edge = random.sample(range(n_nodes), k=edge_size)
        hyperedge_list.append(edge)

    return hnx.Hypergraph(hyperedge_list)


def scale_free_hypergraph(n, k, m, gamma=2.5):
    """
    Scale-free hypergraph via preferential attachment.
    Nodes with higher degree are more likely to appear in new hyperedges.
    """
    edge_list = []
    node_degrees = np.zeros(n)

    # Seed with a few random edges so every node has a chance
    num_init = min(5, m)
    for _ in range(num_init):
        tail = random.sample(range(n), k)
        remaining = list(set(range(n)) - set(tail))
        hsize = random.randint(1, min(k - 1, len(remaining)))
        head = random.sample(remaining, hsize)
        edge_list.append({"head": head, "tail": tail})
        for t in tail:
            node_degrees[t] += 1
        for h in head:
            node_degrees[h] += 1

    while len(edge_list) < m:
        probs = (node_degrees + 1) ** gamma
        probs /= probs.sum()
        tail = list(np.random.choice(n, size=k, replace=False, p=probs))

        remaining = list(set(range(n)) - set(tail))
        if len(remaining) == 0:
            continue
        hsize = random.randint(1, min(k - 1, len(remaining)))
        rem_probs = (node_degrees[remaining] + 1) ** gamma
        rem_probs /= rem_probs.sum()
        head = list(np.random.choice(
            remaining, size=hsize, replace=False, p=rem_probs))

        edge_list.append({"head": head, "tail": tail})
        for t in tail:
            node_degrees[t] += 1
        for h in head:
            node_degrees[h] += 1

    return edge_list


def clustered_hypergraph(n, k, m, num_clusters=5):
    """
    Clustered hypergraph with community structure.
    80 % of edges are intra-cluster; 20 % are inter-cluster bridges.
    """
    edge_list = []
    cluster_size = n // num_clusters
    clusters = [
        list(range(i * cluster_size, min((i + 1) * cluster_size, n)))
        for i in range(num_clusters)
    ]
    # Assign leftover nodes to the last cluster
    for i in range(num_clusters * cluster_size, n):
        clusters[-1].append(i)

    intra_prob = 0.8

    for _ in range(m):
        if random.random() < intra_prob:
            cluster = random.choice(clusters)
            if len(cluster) >= k + 1:
                tail = random.sample(cluster, k)
                remaining = list(set(cluster) - set(tail))
                hsize = random.randint(1, min(k - 1, len(remaining)))
                head = random.sample(remaining, hsize)
            else:                          # cluster too small → fall back
                tail = random.sample(range(n), k)
                remaining = list(set(range(n)) - set(tail))
                hsize = random.randint(1, min(k - 1, len(remaining)))
                head = random.sample(remaining, hsize)
        else:                              # inter-cluster edge
            tail = random.sample(range(n), k)
            remaining = list(set(range(n)) - set(tail))
            hsize = random.randint(1, min(k - 1, len(remaining)))
            head = random.sample(remaining, hsize)

        edge_list.append({"head": head, "tail": tail})

    return edge_list


def small_world_hypergraph(n, k, m, rewire_prob=0.1):
    """
    Small-world hypergraph: ring-lattice base with probabilistic rewiring.
    Each node connects to its k nearest neighbours on the ring; with
    probability *rewire_prob* one tail node is replaced by a random node,
    creating long-range shortcuts.
    """
    edge_list = []

    # Walk around the ring, emitting edges until we have m
    total_edges = 0
    while total_edges < m:
        for center in range(n):
            # k nearest neighbours on the ring form the tail
            neighbors = [(center + offset) % n for offset in range(1, k + 1)]
            tail = list(neighbors)                          # copy
            remaining = list(set(range(n)) - set(tail) - {center})

            if len(remaining) == 0:
                continue

            # Head: choose a nearby node, but rewire with probability rewire_prob
            if random.random() < rewire_prob:
                # long-range shortcut
                head = [random.choice(remaining)]
            else:
                # Prefer nodes close to center on the ring
                nearby = [(center + off) %
                          n for off in range(-2 * k, 2 * k + 1)]
                nearby_remaining = [x for x in nearby if x in remaining]
                if nearby_remaining:
                    head = [random.choice(nearby_remaining)]
                else:
                    head = [random.choice(remaining)]

            edge_list.append({"head": head, "tail": tail})
            total_edges += 1
            if total_edges >= m:
                break

    return edge_list
