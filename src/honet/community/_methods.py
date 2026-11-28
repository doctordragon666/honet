"""社区划分各方法的实现（私有，不对外导出）。

本文件集中存放全部公开函数的算法主体。公开模块（``random_method.py`` /
``extract_community_partition.py`` 等）只做带类型签名的转发，彼此之间**不**
互相 import。

这样分层的原因是 Python 的导入语义：若公开模块 ``extract_community_partition.py``
在顶层 import 兄弟公开模块 ``honet.community.random_method``，import 机制会把
``honet.community`` 的包属性 ``random_method`` 写成**模块对象**，遮蔽
``__init__.py`` 的惰性导出，于是 ``honet.community.random_method`` 从函数变成
模块，调用时报 ``TypeError: 'module' object is not callable``。把实现下沉到
``_methods`` （模块名不与任何导出名冲突）即可彻底避免这种遮蔽。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from honet._deps import require
from honet._hnx import to_hypergraph
from honet.community._hmod import kumar, last_step, modularity, part2dict
from honet.community._subhypergraph import induced_edges
from honet.community.method import Method


def random_method(hnx_g: Any, n_components: int = 5, seed: int | None = None) -> list[list[Any]]:
    """随机打乱结点后轮转分配到 ``n_components`` 个社区。"""
    if n_components < 1:
        msg = f"n_components 必须为正整数，收到 {n_components}"
        raise ValueError(msg)

    np = require("numpy")
    node_list = list(hnx_g.nodes)
    # 取下标做置换而非直接 shuffle 结点列表，避免依赖 numpy 对任意可哈希对象
    # 数组的原地打乱行为
    order = np.random.default_rng(seed).permutation(len(node_list))

    partition: list[list[Any]] = [[] for _ in range(n_components)]
    for position, index in enumerate(order):
        partition[position % n_components].append(node_list[int(index)])
    return partition


def basic_method(hnx_g: Any, seed: int = 42) -> list[set[Any]]:
    """在 s=1 辅助矩阵（团投影）上跑 Louvain。"""
    nx = require("networkx")
    louvain_communities = require("networkx.algorithms.community", attr="louvain_communities")

    # index=True 同时返回矩阵与结点索引数组，索引数组元素即结点标签本身
    matrix, node_index = hnx_g.auxiliary_matrix(s=1, index=True)
    graph = nx.from_numpy_array(matrix.toarray())
    communities = louvain_communities(graph, seed=seed)
    # Louvain 返回 0..n-1 的整数下标，需映射回原始结点标签
    return [{node_index[index] for index in community} for community in communities]


def nmf_method(
    hypergraph: Any,
    min_components: int = 2,
    max_components: int = 6,
    seed: int = 42,
) -> list[list[Any]] | None:
    """对关联矩阵做非负矩阵分解，按系数矩阵最大分量打标签。"""
    np = require("numpy")
    NMF = require("sklearn.decomposition", attr="NMF")

    node_list = list(hypergraph.nodes)
    # 关联矩阵行是结点、列是超边；NMF 要求非负，关联矩阵天然满足
    incidence = hypergraph.incidence_matrix(index=False).toarray().astype(float)

    best_partition: list[list[Any]] | None = None
    best_qh = -float("inf")
    for n_components in range(min_components, max_components):
        if n_components < 2 or n_components > len(node_list):
            continue
        model = NMF(n_components=n_components, init="random", random_state=seed)
        weights = model.fit_transform(np.asarray(incidence))
        # 每个结点归到系数最大的那个分量
        labels = [int(row.argmax()) for row in weights]

        partition: list[list[Any]] = [[] for _ in range(n_components)]
        for index, label in enumerate(labels):
            partition[label].append(node_list[index])
        # 空社区不参与模块度计算，否则会拉低分数
        partition = [group for group in partition if group]

        qh = modularity(hypergraph, partition)
        if qh > best_qh:
            best_qh = qh
            best_partition = partition

    return best_partition


def recursive_method(hnx_g: Any, max_depth: int = 10) -> list[set[Any]]:
    """以 Kumar 为起点，对模块度仍可提升的社区递归拆分。"""
    partition = [set(community) for community in kumar(hnx_g)]
    current_qh = modularity(hnx_g, partition)

    for _ in range(max_depth):
        improved = False
        next_partition: list[set[Any]] = []
        for community in partition:
            edges = induced_edges(hnx_g, community)
            # 社区内不足两条合格超边时不可能再拆分出有意义的子结构
            if len(edges) < 2:
                next_partition.append(community)
                continue

            candidate = [set(group) for group in kumar(to_hypergraph(edges))]
            if len(candidate) < 2:
                next_partition.append(community)
                continue

            trial = [c for c in partition if c is not community] + candidate
            trial_qh = modularity(hnx_g, trial)
            if trial_qh > current_qh:
                next_partition.extend(candidate)
                current_qh = trial_qh
                improved = True
            else:
                next_partition.append(community)
        partition = next_partition
        if not improved:
            break

    return partition


def _to_hypergraphx(hnx_g: Any) -> Any:
    """把 hypernetx 超图转成 hypergraphx 超图。

    延迟到调用时 import，避免 ``honet.community`` 在包导入阶段就拉入转换实现。
    这里从**实现模块**导入而非 ``honet.convert`` 的包属性：包属性是惰性导出的，
    一旦有代码直接 import 过该子模块，包属性就会被写成模块对象。导入后顺手把
    ``honet.convert`` 的包属性修复回函数对象，避免影响该子包的正常使用。
    """
    import sys

    from honet.convert.convert_hypernetx_to_hypergraphx import (
        convert_hypernetx_to_hypergraphx,
    )

    convert_package = sys.modules.get("honet.convert")
    if convert_package is not None:
        shadowed = convert_package.__dict__.get("convert_hypernetx_to_hypergraphx")
        if shadowed is not None and not callable(shadowed):
            convert_package.__dict__["convert_hypernetx_to_hypergraphx"] = (
                convert_hypernetx_to_hypergraphx
            )

    return convert_hypernetx_to_hypergraphx(hnx_g)


def hierarchy_method(hnx_g: Any, cut_height: float = 0.5) -> list[list[Any]]:
    """对超边做层次聚类，再用多数票把超边簇投影成结点社区。"""
    np = require("numpy")
    nx = require("networkx")
    to_numpy_array = require("networkx", attr="to_numpy_array")
    get_num_communities = require("hypergraphx.communities", attr="get_num_hyperlink_communties")
    jaccard_distance = require("hypergraphx.measures.edge_similarity", attr="jaccard_distance")
    fcluster = require("scipy.cluster.hierarchy", attr="fcluster")
    linkage = require("scipy.cluster.hierarchy", attr="linkage")
    squareform = require("scipy.spatial.distance", attr="squareform")

    # 1. 取最大连通分量：孤立结点无超边可聚类，稍后单独成簇
    hgx_graph = _to_hypergraphx(hnx_g)
    subgraph = hgx_graph.subhypergraph(hgx_graph.largest_component())
    # 统一成员顺序，保证同一超边在不同来源下得到相同的元组
    edges = [tuple(sorted(edge)) for edge in subgraph.get_edges()]

    all_nodes = list(hnx_g.nodes)
    if len(edges) < 2:
        # 少于两条超边时无法构成距离矩阵，整张图视为一个社区
        return [sorted(all_nodes)]

    # 2. 构建超边相似度图：结点 → 关联的超边下标
    incident: dict[Any, list[int]] = {}
    for edge_index, edge in enumerate(edges):
        for node in edge:
            incident.setdefault(node, []).append(edge_index)

    similarity = nx.Graph()
    similarity.add_nodes_from(range(len(edges)))
    seen: set[tuple[int, int]] = set()
    for edge_indexes in incident.values():
        for i in range(len(edge_indexes) - 1):
            for j in range(i + 1, len(edge_indexes)):
                pair = (edge_indexes[i], edge_indexes[j])
                if pair in seen:
                    continue
                seen.add(pair)
                weight = jaccard_distance(set(edges[pair[0]]), set(edges[pair[1]]))
                # 完全不共享成员的超边之间不连边（距离保持非边默认值 1.0）
                if weight > 0:
                    similarity.add_edge(pair[0], pair[1], weight=weight)

    # 3. 距离矩阵：非边距离取 1.0（Jaccard 距离上界），对角线置 0
    distances = to_numpy_array(similarity, weight="weight", nonedge=1.0)
    np.fill_diagonal(distances, 0.0)
    dendrogram = linkage(squareform(distances), method="average")

    # 4. 按距离阈值切树，得到超边簇
    n_communities = get_num_communities(dendrogram, cut_height)
    edge_labels = fcluster(dendrogram, n_communities, criterion="maxclust")

    # 5. 多数票：结点归属它所在超边中最常见的簇
    votes: dict[Any, dict[int, int]] = {}
    for edge_index, label in enumerate(edge_labels):
        for node in edges[edge_index]:
            tally = votes.setdefault(node, {})
            tally[int(label)] = tally.get(int(label), 0) + 1

    buckets: dict[Any, set[Any]] = {}
    for node, tally in votes.items():
        # 票数相同时取较小的簇号，保证结果确定
        best = max(sorted(tally), key=lambda label: tally[label])
        buckets.setdefault(("edge-cluster", best), set()).add(node)

    # 未被任何超边覆盖的结点各自成簇
    for node in all_nodes:
        if node not in votes:
            buckets.setdefault(("isolated", node), set()).add(node)

    return sorted((sorted(group) for group in buckets.values()), key=lambda group: group[0])


def check_partition(hypergraph: Any, prev_partition: Any) -> list[Any]:
    """返回当前网络中未被历史划分覆盖的结点。"""
    current_nodes = list(hypergraph.nodes)
    if not isinstance(prev_partition, dict):
        return []
    current_node_set = set(current_nodes)
    covered = {node for node in prev_partition if node in current_node_set}
    return [node for node in current_nodes if node not in covered]


def extract_community_partition(
    hnx_g: Any,
    method: Method = Method.KUMAR,
    only_qh: bool = False,
) -> dict[str, Any] | float:
    """按 ``method`` 划分超图并返回模块度与分区。"""
    if method is Method.KUMAR:
        partition = kumar(hnx_g)
    elif method is Method.LAST_STEP:
        partition = last_step(hnx_g, kumar(hnx_g))
    elif method is Method.Random:
        partition = random_method(hnx_g, n_components=5)
    elif method is Method.Basic:
        partition = basic_method(hnx_g)
    elif method is Method.NMF:
        partition = nmf_method(hnx_g)
    elif method is Method.RECURSIVE:
        partition = recursive_method(hnx_g)
    elif method is Method.Hierarchy:
        partition = hierarchy_method(hnx_g)
    else:
        msg = f"未实现的社区划分方法：{method!r}"
        raise ValueError(msg)

    if partition is None:
        msg = f"方法 {method.name} 没有产出任何社区划分（输入可能过于退化）"
        raise RuntimeError(msg)

    qh = modularity(hnx_g, partition)
    if only_qh:
        return qh
    return {
        "qh": qh,
        "communities": partition,
        "nodemap": part2dict(partition),
    }


def compare_methods(
    hnx_g: Any,
    methods: Iterable[Method] | None = None,
    compare_fun: Callable[[Any, Method], Any] | None = None,
) -> dict[Method, Any]:
    """对同一张超图依次施加多种方法并收集评价指标。"""
    selected = list(Method) if methods is None else list(methods)
    result: dict[Method, Any] = {}
    for method in selected:
        try:
            if compare_fun is None:
                result[method] = extract_community_partition(hnx_g, method=method, only_qh=True)
            else:
                result[method] = compare_fun(hnx_g, method)
        except Exception:
            # 单个方法失败不应影响其他方法的比较结果
            result[method] = None
    return result


def compare_graphs(
    H_lst: Iterable[Any],
    method: Method = Method.KUMAR,
) -> dict[Any, dict[str, Any] | float]:
    """对多张超图逐一施加同一种划分方法。"""
    if isinstance(H_lst, dict):
        items = list(H_lst.items())
    else:
        items = list(enumerate(H_lst))

    return {label: extract_community_partition(graph, method=method) for label, graph in items}
