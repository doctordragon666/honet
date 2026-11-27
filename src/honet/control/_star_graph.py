"""HAT 有向超图的星形图（二分图）展开。

星形图把超图展开成二分图：``n<i>`` 是系统结点，``e<j>`` 是超边结点。有向超边
按「尾结点 → 边结点 → 头结点」连边，因此一条边对头结点的影响要等尾结点全部
可达之后才会发生；无向超边则让边结点与全部成员相连。

结点名带前缀是后续匹配算法的关键：``n`` 前缀标识系统结点，``e`` 前缀标识超边，
``u`` 前缀标识人为补上的驱动结点（见 ``_dilation``）。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require
from honet.control._hat_edges import edge_head_tail


def build_star_graph(HG: Any) -> Any:
    """构造星形图。

    参数:
        HG: HAT 超图对象（``directed`` 决定返回有向图还是无向图）。

    返回:
        有向超图返回 ``networkx.DiGraph``，无向超图返回 ``networkx.Graph``；
        系统结点带属性 ``bipartite=0``，超边结点带属性 ``bipartite=1``。

    注意:
        超边结点的名字用边表的行号（即 ``HG.edges.index``，HAT 里与 ``Edges``
        列相同），与旧实现一致。
    """
    nx = require("networkx")
    G = nx.DiGraph() if HG.directed else nx.Graph()
    G.add_nodes_from([f"n{i}" for i in range(HG.nnodes)], bipartite=0)

    for edge_id, (heads, tails) in zip(HG.edges.index, edge_head_tail(HG), strict=True):
        e_node = f"e{edge_id}"
        G.add_node(e_node, bipartite=1)
        if HG.directed:
            for tail in tails:
                G.add_edge(f"n{tail}", e_node)
            for head in heads:
                G.add_edge(e_node, f"n{head}")
        else:
            # 无向边表没有 head/tail 列，edge_head_tail 已把成员同时当作头与尾
            for member in heads:
                G.add_edge(e_node, f"n{member}")

    return G
