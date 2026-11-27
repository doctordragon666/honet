"""结点的聚类系数。

采用「邻居对共享超边」的局部聚类系数口径：结点 i 的邻居之间有多少对
至少共享一条超边。纯 Python 实现，不引入额外第三方包。
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from honet._hnx import edge_members, node_neighbors


def clustering_coefficient(H: Any) -> dict[Any, float]:
    """计算超图中每个结点的聚类系数。

    对结点 ``i``，取邻居集合 ``N(i)``（与它共享至少一条超边的结点），
    统计 ``N(i)`` 中两两共享超边的比例：

    .. math::

        C_i = \\frac{|\\{(j,k) : j,k \\in N(i),\\ \\exists e,\\ j,k \\in e\\}|}
                    {\\binom{|N(i)|}{2}}

    参数:
        H: hypernetx 超图对象。

    返回:
        ``{结点: 聚类系数}``，邻居少于 2 个的结点取 ``0.0``。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"]})
        >>> clustering_coefficient(H)["a"]
        1.0
    """
    members = edge_members(H)
    result: dict[Any, float] = {}

    for node in H.nodes:
        neighbors = set(node_neighbors(H, node))
        k = len(neighbors)
        if k < 2:
            result[node] = 0.0
            continue

        # 收集邻居之间的连边对；同一对邻居被多条超边共享时只计一次
        linked: set[frozenset[Any]] = set()
        for nodes in members.values():
            inside = neighbors.intersection(nodes)
            if len(inside) < 2:
                continue
            for pair in combinations(sorted(inside, key=repr), 2):
                linked.add(frozenset(pair))

        result[node] = len(linked) / (k * (k - 1) / 2)

    return result
