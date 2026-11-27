"""结点的介数中心性。

在超图的二部图表示（结点 ↔ 超边）上计算最短路介数，再只保留原结点。
hypernetx 2.x 移除了 ``hnx.bipartite``，这里直接构造 NetworkX 二部图。

性能说明：
* 二部图由拓扑索引一次性构建（不重复遍历 hypernetx 对象）；
* 结果按结构指纹缓存——介数是本模块里最贵的指标（全源最短路），
  结构未变时重复调用不再重算；
* ``samples`` 参数可启用 NetworkX 的采样近似，用少量源点换取大幅提速。
"""

from __future__ import annotations

from typing import Any

from honet import _cache
from honet._deps import require
from honet._topology import Topology, topology


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


def _compute(H: Any, samples: int | None) -> dict[Any, float]:
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

    参数:
        H: hypernetx 超图对象。
        samples: 为 ``None`` 时精确计算；给正整数时只从这么多源点出发做近似
            （对应 NetworkX 的 ``k`` 参数），大图上可显著降低耗时。

    返回:
        ``{结点: 介数中心性}``，只含原超图的结点，不含超边结点。

    说明:
        精确结果（``samples=None``）按结构指纹缓存；近似结果依赖采样，
        不缓存。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> set(betweenness_centrality(H)) == set(H.nodes)
        True
    """
    if samples:
        return _compute(H, samples)
    return _cache.cached(H, "betweenness_centrality:v2", lambda: _compute(H, None))
