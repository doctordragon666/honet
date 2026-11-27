"""hypernetx 超图 → NetworkX 成对图。

超边内部两两连边（团投影）得到成对图：成员数小于 2 的超边无法产生边，
自然被忽略。同一对结点被多条超边共享时只保留一条无向边。
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from honet._deps import require


def convert_hypernetx_to_nxgraph(hnx_hypergraph: Any) -> Any:
    """把 hypernetx 超图转换为 NetworkX 无向图（团投影）。

    对每条超边取两两组合作为图的边，即 ``(u, v)`` 存在当且仅当 ``u``、``v``
    至少共享一条超边。孤立结点不会出现在结果图中（旧实现也不补结点）。

    参数:
        hnx_hypergraph: 输入的 hypernetx 超图对象；类型不符时抛 ``TypeError``。

    返回:
        ``networkx.Graph``（无向、无权）。重复的结点对由 NetworkX 自动去重，
        因此**超边权重与重数信息会丢失**。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["c", "d"]})
        >>> G = convert_hypernetx_to_nxgraph(H)
        >>> G.number_of_edges()
        4
    """
    hnx = require("hypernetx")
    nx = require("networkx")

    if not isinstance(hnx_hypergraph, hnx.Hypergraph):
        raise TypeError("输入必须是 hypernetx.Hypergraph 对象")

    # 1. 每条超边展开成完全图的边集合（成员数 < 2 时 combinations 为空，自动跳过）
    edges: list[tuple[Any, Any]] = []
    for edge in hnx_hypergraph.edges:
        nodes = list(hnx_hypergraph.edges[edge])
        edges.extend(combinations(nodes, 2))

    # 2. 交给 NetworkX 建无向图（自动去重）
    return nx.Graph(edges)
