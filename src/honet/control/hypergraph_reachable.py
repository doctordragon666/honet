"""有向超图上的可达性传播。"""

from __future__ import annotations

from typing import Any

from honet.control._reachability import hypergraph_reachable as _impl


def hypergraph_reachable(HG_local: Any, start_nodes: Any) -> set[Any]:
    """计算从起始结点集合出发、通过超边传播可达的结点集合。

    传播规则：一条超边的全部尾结点都可达时，它的头结点变为可达，反复迭代直到
    不再新增结点。

    参数:
        HG_local: HAT 有向超图。
        start_nodes: 起始结点下标集合（任意可迭代对象）。

    返回:
        可达结点下标集合，包含 ``start_nodes`` 本身。

    示例:
        >>> hypergraph_reachable(HG, {0})  # doctest: +SKIP
        {0, 1, 2}
    """
    return _impl(HG_local, start_nodes)
