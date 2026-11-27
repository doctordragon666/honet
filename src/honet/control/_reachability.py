"""有向超图上的可达性传播。

传播规则（与旧实现一致）：一条超边的**全部**尾结点都可达时，它的头结点才变为
可达。重复扫描直到不再新增结点，因此时间复杂度是 ``O(nnodes * nedges)``，
但常数极小、不依赖任何第三方包。
"""

from __future__ import annotations

from typing import Any

from honet.control._hat_edges import edge_head_tail, node_id


def hypergraph_reachable(HG_local: Any, start_nodes: Any) -> set[Any]:
    """计算从起始结点集合出发、通过超边传播可达的结点集合。

    参数:
        HG_local: HAT 有向超图。
        start_nodes: 起始结点下标集合（可迭代）。

    返回:
        可达结点下标集合，包含 ``start_nodes`` 本身。
    """
    ends = edge_head_tail(HG_local)
    accessible = {node_id(n) for n in start_nodes}
    changed = True
    while changed:
        changed = False
        for heads, tails in ends:
            if set(tails).issubset(accessible):
                before = len(accessible)
                accessible.update(heads)
                if len(accessible) > before:
                    changed = True
    return accessible
