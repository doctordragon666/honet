"""结点的余平均度（excess average degree）。

刻画「我的邻居们平均有多活跃」：邻居的平均超度越高，说明该结点越可能
处在高阶交互密集的区域。复用一次性拓扑索引，避免逐结点重新遍历超边。
"""

from __future__ import annotations

from typing import Any

from honet import _cache
from honet._topology import Topology, topology


def _compute(top: Topology) -> dict[Any, float]:
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
    return _cache.cached(H, "excess_average_degree:v2", lambda: _compute(topology(H)))
