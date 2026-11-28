"""结点级指标：每个结点返回一个分值。

本模块把原先「一个函数一个文件」的结点级指标合并到一处：

* :func:`weighted_degree` —— 结点参与的交互强度之和；
* :func:`clustering_coefficient` —— 邻居之间是否也彼此相连；
* :func:`excess_average_degree` —— 邻居的平均超度；
* :func:`eigenvector_centrality` —— 归一化拉普拉斯 Fiedler 向量；
* :func:`betweenness_centrality` —— 结点—超边二部图上的最短路介数；
* :func:`temporal_closeness_centrality` —— 时序网络上的时序接近中心性。

除 :func:`temporal_closeness_centrality` 接收 :class:`honet.TemporalNetwork`
之外，其余函数都接收 **hypernetx 超图**，并尽量复用 :mod:`honet._topology` 的
一次性拓扑索引与 :mod:`honet._cache` 的结构指纹缓存，避免各自重复遍历超边。
"""

from __future__ import annotations

from typing import Any

from honet import _cache
from honet._deps import require
from honet._topology import Topology, topology
from honet.temporal import TemporalNetwork


def _compute_weighted_degree(H: Any) -> dict[Any, float]:
    """计算加权度：有权重时走关联矩阵，否则直接用度值。

    权重检测不能依赖 ``H.edges.weights``——hypernetx 2.4.3 没有这个属性
    （带 ``cell_weight_col`` 构建的超图也照样没有），据此判断会把加权超图
    误当成无权超图，得出错误的度值。这里直接看关联矩阵：只有当矩阵里所有
    非零元都是 1 时才走快速路径（此时加权度恒等于度）。
    """
    np = require("numpy")
    B, node_names, _ = H.incidence_matrix(index=True, weights=True)

    data = B.data if hasattr(B, "data") else np.asarray(B).ravel()
    if data.size and not np.all(data == 1):
        values = np.asarray(B.sum(axis=1)).flatten()
        return {node: float(w) for node, w in zip(node_names, values, strict=True)}

    # 无权（或权重全为 1）：加权度等于度，用索引里的度值即可
    degrees = topology(H).degrees
    return {node: float(degrees.get(node, 0)) for node in node_names}


def weighted_degree(H: Any) -> dict[Any, float]:
    """计算超图中每个结点的加权度。

    加权度定义为结点所在全部超边的权重之和，等价于关联矩阵按行求和。
    无权超图上它退化为普通的度。

    参数:
        H: hypernetx 超图对象（构建时可用 ``cell_weight_col`` 携带权重）。

    返回:
        ``{结点: 加权度}``，值为 ``float``。

    说明:
        结果按超图结构指纹缓存；结构未变时重复调用直接返回上次结果。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"]})
        >>> weighted_degree(H)["a"]
        1.0
    """
    return _cache.cached(H, "weighted_degree:v2", lambda: _compute_weighted_degree(H))


def _compute_clustering(top: Topology) -> dict[Any, float]:
    """在拓扑索引上计算全部结点的聚类系数。"""
    # 每条超边内部的结点对都「共享超边」；同一对可能被多条超边覆盖，只计一次。
    # 这里只记录结点对本身，不按端点索引——因为判断聚类系数时要求「两端都在
    # 邻居集合内」，而邻居集合不含结点自身，按端点索引会把含自身的对也带进来。
    shared_pairs: set[frozenset[Any]] = set()
    for members in top.members.values():
        ordered = sorted(members, key=repr)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                shared_pairs.add(frozenset((ordered[i], ordered[j])))

    result: dict[Any, float] = {}
    for node in top.nodes:
        neighbors = top.neighbors_of(node)
        k = len(neighbors)
        if k < 2:
            result[node] = 0.0
            continue
        # 邻居之间共享超边的对数：两端都必须在该结点的邻居集合内
        count = sum(1 for pair in shared_pairs if pair <= neighbors)
        result[node] = count / (k * (k - 1) / 2)
    return result


def clustering_coefficient(H: Any) -> dict[Any, float]:
    """计算超图中每个结点的聚类系数。

    对结点 ``i`` ，取邻居集合 ``N(i)`` （与它共享至少一条超边的结点），
    统计 ``N(i)`` 中两两共享超边的比例：

    .. math::

        C_i = \\frac{|\\{(j,k) : j,k \\in N(i),\\ \\exists e,\\ j,k \\in e\\}|}
                    {\\binom{|N(i)|}{2}}

    参数:
        H: hypernetx 超图对象。

    返回:
        ``{结点: 聚类系数}``，邻居少于 2 个的结点取 ``0.0``。

    说明:
        结果按超图结构指纹缓存；结构未变时重复调用直接返回上次结果。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"]})
        >>> clustering_coefficient(H)["a"]
        1.0
    """
    return _cache.cached(H, "clustering_coefficient:v2", lambda: _compute_clustering(topology(H)))


def _compute_excess(top: Topology) -> dict[Any, float]:
    """在拓扑索引上计算全部结点的余平均度。"""
    result: dict[Any, float] = {}
    for node in top.nodes:
        neighbors = top.neighbors_of(node)
        if not neighbors:
            result[node] = 0.0
            continue
        total = sum(top.degrees.get(neighbor, 0) for neighbor in neighbors)
        result[node] = total / len(neighbors)
    return result


def excess_average_degree(H: Any) -> dict[Any, float]:
    r"""计算每个结点的余平均度，即邻居结点超度的平均值。

    刻画「我的邻居们平均有多活跃」：邻居的平均超度越高，说明该结点越可能
    处在高阶交互密集的区域。

    .. math::

        \langle dk_{nn}\rangle_i = \frac{1}{|\tau(i)|}\sum_{j\in\tau(i)} HD(v_j)

    其中 :math:`\tau(i)` 为结点 i 的邻居集合，:math:`HD(v_j)` 为邻居 j 的超度。

    参数:
        H: hypernetx 超图对象。

    返回:
        ``{结点: 余平均度}``；无邻居的结点取 ``0.0``。

    说明:
        结果按超图结构指纹缓存；结构未变时重复调用直接返回上次结果。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> round(excess_average_degree(H)["a"], 3)
        2.0
    """
    return _cache.cached(H, "excess_average_degree:v2", lambda: _compute_excess(topology(H)))


def eigenvector_centrality(H: Any) -> dict[Any, float]:
    """基于归一化拉普拉斯的 Fiedler 向量计算结点中心性。

    以归一化拉普拉斯矩阵的第二小特征值对应的特征向量（Fiedler 向量）度量
    结点在高阶连通结构中的位置，绝对值越大表示越靠近结构的「分界」。

    参数:
        H: hypernetx 超图对象。

    返回:
        ``{结点: 中心性}``，按拉普拉斯矩阵的行列顺序对齐到结点标签。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> set(eigenvector_centrality(H)) == set(H.nodes)
        True
    """
    np = require("numpy")
    hnx = require("hypernetx")

    L, nodes = hnx.norm_lap(H)
    dense = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
    eigenvalues, eigenvectors = np.linalg.eig(dense)

    order = np.argsort(eigenvalues)
    # 取第二小特征值对应的特征向量；单结点图没有第二小，退回第一列
    index = order[1] if len(order) > 1 else order[0]
    fiedler = np.asarray(eigenvectors[:, index]).flatten().real

    return {node: float(fiedler[i]) for i, node in enumerate(nodes)}


def _build_bipartite(top: Topology) -> Any:
    """由拓扑索引构建「结点 ↔ 超边」二部图。"""
    nx = require("networkx")

    graph = nx.Graph()
    for edge, members in top.members.items():
        # 超边结点加前缀元组，避免与原结点同名冲突
        edge_node = ("__edge__", edge)
        graph.add_node(edge_node, bipartite=1)
        for node in members:
            graph.add_node(node, bipartite=0)
            graph.add_edge(node, edge_node)
    return graph


def _compute_betweenness(H: Any, samples: int | None) -> dict[Any, float]:
    """在二部图上算介数，只保留原超图结点。"""
    nx = require("networkx")

    graph = _build_bipartite(topology(H))
    kwargs = {"k": samples} if samples else {}
    scores = nx.betweenness_centrality(graph, **kwargs)
    return {
        node: float(scores[node])
        for node in graph.nodes
        if node in scores and not (isinstance(node, tuple) and node and node[0] == "__edge__")
    }


def betweenness_centrality(H: Any, samples: int | None = None) -> dict[Any, float]:
    """计算超图结点的介数中心性（基于结点—超边二部图）。

    在超图的二部图表示（结点 ↔ 超边）上计算最短路介数，再只保留原结点。
    hypernetx 2.x 移除了 ``hnx.bipartite``，这里直接构造 NetworkX 二部图。

    参数:
        H: hypernetx 超图对象。
        samples: 为 ``None`` 时精确计算；给正整数时只从这么多源点出发做近似
            （对应 NetworkX 的 ``k`` 参数），大图上可显著降低耗时。

    返回:
        ``{结点: 介数中心性}``，只含原超图的结点，不含超边结点。

    说明:
        介数是本模块里最贵的指标（全源最短路）。精确结果（``samples=None``）
        按结构指纹缓存，结构未变时重复调用不再重算；近似结果依赖采样，不缓存。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> set(betweenness_centrality(H)) == set(H.nodes)
        True
    """
    if samples:
        return _compute_betweenness(H, samples)
    return _cache.cached(H, "betweenness_centrality:v2", lambda: _compute_betweenness(H, None))


def _temporal_distances(
    network: TemporalNetwork,
    source: Any,
    *,
    max_hops: int,
) -> dict[Any, int]:
    """从 ``source`` 出发，求沿时间递增路径可达各结点所需的最少事件跳数。

    这就是 Kim & Anderson（PRE 85, 026107, 2012）「时间有序图」上以跳数为距离
    的最短路。一条时间尊重路径是事件序列 ``e_1, e_2, ..., e_k``，满足

    * 时间严格递增：``t(e_1) < t(e_2) < ... < t(e_k)``；
    * 相邻事件有公共结点（换乘点），且 ``source ∈ e_1``、``u ∈ e_k``。

    跳数记作 k。与论文按绝对时间（秒）计时的写法不同，本实现按**跳数**计数：
    跳数与时间戳的物理单位无关，不需要用户先决定尺度（年 / 秒 / 帧）。

    实现要点：逐层推进，同时记录「到达某结点的最早时刻」。只保留最早时刻是
    正确的——更早到达绝不会让后续路径变差（可搭上的事件只会更多），因此每层
    只需保留最早到达时刻，不必保留全部帕累托对。

    Args:
        network: 时序网络。
        source: 起点结点。
        max_hops: 跳数上限（含）。

    Returns:
        ``{结点: 最少跳数}``，不含 ``source`` 自身；不可达的结点不出现。
    """
    # 事件按时间升序（TemporalNetwork 始终维护该不变量）
    timeline = list(zip(network.times, network.edges, strict=True))

    # arrival[node] = 用当前跳数到达该结点的最早时刻
    arrival: dict[Any, float] = {source: float("-inf")}
    distances: dict[Any, int] = {}

    for hops in range(1, max_hops + 1):
        reached: dict[Any, float] = {}
        # 按时间升序扫描：同一结点第一次被触达时即是最早到达时刻
        for timestamp, members in timeline:
            # 该事件能否被「上一跳已到达的某个结点」搭上
            if not any(node in arrival and arrival[node] < timestamp for node in members):
                continue
            for node in members:
                if node not in distances and node not in reached:
                    reached[node] = timestamp
        if not reached:
            break
        for node, timestamp in reached.items():
            if node == source:
                # source 回到自己不算「到达」：那会把它自己的距离算成 >0
                continue
            distances[node] = hops
            # 只登记首次到达时刻；已到达的结点保持更早的时刻，
            # 否则会把 source 的 -inf 覆盖成更晚的时刻，漏掉本该可搭的事件
            arrival.setdefault(node, timestamp)
        if len(distances) >= len(network.nodes) - 1:
            break
    return distances


def _compute_temporal_closeness(
    network: TemporalNetwork,
    *,
    max_hops: int | None,
) -> dict[Any, float]:
    """在时序网络上计算全部结点的时序接近中心性。"""
    nodes = network.nodes
    total = len(nodes)
    if total < 2:
        return {node: 0.0 for node in nodes}

    # 不设上限时，最长的有意义跳数不会超过事件数
    limit = max_hops if max_hops is not None else max(network.num_edges, 1)

    result: dict[Any, float] = {}
    for source in nodes:
        distances = _temporal_distances(network, source, max_hops=limit)
        if not distances:
            result[source] = 0.0
            continue
        # 调和形式：对距离取倒数再求和。相比「平均距离的倒数」，它对不可达
        # 结点天然给出 0 贡献，不需要人为填补距离，也不会因为一个孤立结点
        # 把整张网络的接近中心性全部拉成 0。
        result[source] = sum(1.0 / hops for hops in distances.values()) / (total - 1)
    return result


def temporal_closeness_centrality(
    network: TemporalNetwork,
    *,
    max_hops: int | None = None,
) -> dict[Any, float]:
    """计算时序网络结点的时序接近中心性（time-respecting closeness）。

    静态网络的接近中心性假设信息可以沿任意边传播；时序网络里这并不成立——
    只有当后续事件的时间戳**严格更晚**时，信息才可能从一条交互传到下一条。
    本函数只承认这种「时间尊重路径」，用其跳数计算调和接近中心性：

    .. math::

        C(v) = \\frac{1}{n-1}\\sum_{u \\neq v} \\frac{1}{d_t(v, u)}

    其中 :math:`d_t(v,u)` 是 v 到 u 的时间尊重路径的最少跳数；不可达时该项为 0。

    参数:
        network: :class:`honet.TemporalNetwork` 或其加权 / 有向变体
            （``TemporalNetworkW`` / ``TemporalNetworkD`` / ``TemporalNetworkWD``）。
        max_hops: 路径跳数上限；``None`` （默认）表示不限制。
            给一个正整数可以把注意力限制在「几跳以内」的传播范围，
            这对应 pathpy 等库里的 ``delta`` 时间窗思想。

    返回:
        ``{结点: 时序接近中心性}``，取值落在 ``[0, 1]``。
        没有出向时间路径的结点取 ``0.0``。

    说明:
        本指标按**跳数**而非绝对时间计时，因此结果与时间戳的物理单位无关。
        实现是逐源的按时间推进 BFS，复杂度 ``O(|V| · |E|)``。

    示例:
        >>> from honet import TemporalNetwork
        >>> tn = TemporalNetwork(events=[(1, ["a", "b"]), (2, ["b", "c"])])
        >>> scores = temporal_closeness_centrality(tn)
        >>> scores["b"]        # b 一个跳就能到 a 和 c
        1.0
        >>> scores["c"] < scores["a"]   # c 只能沿时间到达 b
        True
    """
    if not isinstance(network, TemporalNetwork):
        raise TypeError(
            "temporal_closeness_centrality 只接受 honet.TemporalNetwork 及其变体，"
            f"收到 {type(network).__name__}；"
            "一次性超图请改用 betweenness_centrality / eigenvector_centrality"
        )
    if max_hops is not None:
        if isinstance(max_hops, bool) or not isinstance(max_hops, int):
            raise ValueError(f"max_hops 必须是正整数或 None：{max_hops!r}")
        if max_hops < 1:
            raise ValueError(f"max_hops 必须至少为 1：{max_hops!r}")
    return _cache.cached(
        network,
        f"temporal_closeness_centrality:v1:{max_hops}",
        lambda: _compute_temporal_closeness(network, max_hops=max_hops),
    )
