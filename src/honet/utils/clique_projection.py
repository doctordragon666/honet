"""honet.utils.clique_projection —— 团投影（高阶 → 成对网络）。"""

from __future__ import annotations

from honet._types import Node
from honet.core import Hypergraph


def clique_projection(hg: Hypergraph) -> dict[Node, frozenset[Node]]:
    """团投影：把超图展开为成对（pairwise）邻接映射。

    每条大小为 ``k`` 的超边在其成员之间生成一个 ``k`` 团，因此投影后
    结点 ``i`` 的邻居就是所有与它共边的结点，即 ``N(i) = (∪_{e ∋ i} e) \\ {i}``。

    Args:
        hg: 待投影的超图。

    Returns:
        字典，键为每个结点（含孤立结点），值为其邻居的 ``frozenset``。
        返回值覆盖全部结点，可直接用于构图。

    用法::

        proj = clique_projection(hg)
        proj["a"]        # frozenset({'b', 'c'})
    """
    projection: dict[Node, set[Node]] = {node: set() for node in hg.nodes}
    for edge in hg.edges:
        # 超边内部任意两点相连：对每个成员并入“该超边成员集合 - 自身”
        for node in edge.members:
            projection[node] |= edge.members - {node}
    return {node: frozenset(neigh) for node, neigh in projection.items()}


__all__ = ["clique_projection"]
