"""honet.utils.degree_sequence —— 高阶度序列。"""

from __future__ import annotations

from honet._types import Node
from honet.core import Hypergraph


def degree_sequence(hg: Hypergraph) -> list[int]:
    """返回高阶度序列，按降序排列。

    度定义为结点参与的超边条数：``k_i = |{e ∈ E : i ∈ e}|``。
    该序列可用于绘制度分布、计算异质性等。

    Args:
        hg: 待统计的超图。

    Returns:
        长度等于 ``num_nodes`` 的整数列表，降序。空图返回 ``[]``。
    """
    # 一次遍历所有超边累加各结点度，复杂度 O(Σ|e|)
    degrees: dict[Node, int] = {node: 0 for node in hg.nodes}
    for edge in hg.edges:
        for node in edge.members:
            degrees[node] += 1
    return sorted(degrees.values(), reverse=True)


__all__ = ["degree_sequence"]
