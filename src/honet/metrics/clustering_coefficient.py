"""结点的聚类系数。

采用「邻居对共享超边」的局部聚类系数口径：结点 i 的邻居之间有多少对
至少共享一条超边。

实现用一次性构建的拓扑索引（:mod:`honet._topology`）替代「逐结点扫描全部
超边」的朴素做法：共享超边的结点对只需统计一次，随后按结点归并即可。
复杂度从 O(结点数 × 超边数) 降到 O(Σ 超边成员数²)。
"""

from __future__ import annotations

from typing import Any

from honet import _cache
from honet._topology import Topology, topology


def _compute(top: Topology) -> dict[Any, float]:
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

    对结点 ``i``，取邻居集合 ``N(i)``（与它共享至少一条超边的结点），
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
    return _cache.cached(H, "clustering_coefficient:v2", lambda: _compute(topology(H)))
