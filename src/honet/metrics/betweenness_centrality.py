"""结点的介数中心性。

在超图的二部图表示（结点 ↔ 超边）上计算最短路介数，再只保留原结点。
hypernetx 2.x 移除了 ``hnx.bipartite``，这里直接构造 NetworkX 二部图，
避免依赖已删除的接口。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require
from honet._hnx import edge_members


def betweenness_centrality(H: Any) -> dict[Any, float]:
    """计算超图结点的介数中心性（基于结点—超边二部图）。

    参数:
        H: hypernetx 超图对象。

    返回:
        ``{结点: 介数中心性}``，只含原超图的结点，不含超边结点。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> set(betweenness_centrality(H)) == set(H.nodes)
        True
    """
    nx = require("networkx")

    graph = nx.Graph()
    for edge, nodes in edge_members(H).items():
        # 超边结点加前缀，避免与原结点同名冲突
        edge_node = ("__edge__", edge)
        graph.add_node(edge_node, bipartite=1)
        for node in nodes:
            graph.add_node(node, bipartite=0)
            graph.add_edge(node, edge_node)

    scores = nx.betweenness_centrality(graph)
    original = set(H.nodes)
    return {node: float(scores[node]) for node in original if node in scores}
