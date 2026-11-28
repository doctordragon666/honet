"""hypernetx 版本兼容层。

旧实现（hypertrade）基于 hypernetx 1.x 的接口编写，而 hypernetx 2.x 移除了
若干属性，直接照搬会 ``AttributeError``。本模块把这些差异集中到一处，
所有分析子包只调用这里的函数，不直接依赖 hypernetx 的内部属性。

实测差异（hypernetx 2.4.3）：

============================ ===========================================
1.x 写法                      2.4.3 中的替代
============================ ===========================================
``H.nodes.degree``             ``H.degree()`` （标量或 dict）
``H.edges.members()``          ``H.edges.incidence_dict`` / ``H.edges[e]``
``H.edges.size``               用 ``len(H.edges[e])`` 逐个算
``H.nodes.neighbors``          经关联关系自行展开
``H.nodes.memberships``        仍可用（返回 dict）
============================ ===========================================

性能说明：``edge_members`` / ``node_degrees`` / ``edge_sizes`` 都优先复用
:mod:`honet._topology` 的一次性索引。索引本身带结构指纹缓存，因此同一超图上
的多次调用不会重复遍历；只有无法索引的非标准对象才走逐边访问的兜底路径。
"""

from __future__ import annotations

from typing import Any


def node_degrees(H: Any) -> dict[Any, int]:
    """返回 ``{结点: 度}``。

    优先用共享拓扑索引（一次构建、带缓存），不可用时退回 ``H.degree()``
    批量形式，再退回逐个结点计算，兼容 1.x 与 2.x。
    """
    try:
        from honet._topology import topology

        return dict(topology(H).degrees)
    except Exception:  # noqa: BLE001 - 非标准对象退回逐个求度
        pass

    try:
        degrees = H.degree()
    except TypeError:
        degrees = None
    if isinstance(degrees, dict):
        return dict(degrees)

    result: dict[Any, int] = {}
    for node in H.nodes:
        try:
            result[node] = int(H.degree(node))
        except Exception:  # noqa: BLE001 - 兼容层需要容忍实现差异
            result[node] = len(H.nodes.memberships[node])
    return result


def edge_members(H: Any) -> dict[Any, list[Any]]:
    """返回 ``{超边: 成员列表}``，顺序尽量保持插入顺序。

    优先走 ``H.edges.incidence_dict`` （一次性字典，近 0ms）；逐边访问
    ``H.edges[e]`` 每次都会重建成员列表，只在没有 incidence_dict 时使用。
    """
    try:
        raw = H.edges.incidence_dict
        if not isinstance(raw, dict):
            raw = raw()
        return {edge: list(members) for edge, members in raw.items()}
    except Exception:  # noqa: BLE001
        return {e: list(H.edges[e]) for e in H.edges}


def edge_sizes(H: Any) -> dict[Any, int]:
    """返回 ``{超边: 成员数}``。"""
    try:
        from honet._topology import topology

        return dict(topology(H).sizes)
    except Exception:  # noqa: BLE001
        return {e: len(members) for e, members in edge_members(H).items()}


def node_memberships(H: Any) -> dict[Any, list[Any]]:
    """返回 ``{结点: 所属超边列表}``。"""
    try:
        from honet._topology import topology

        return {n: list(edges) for n, edges in topology(H).memberships.items()}
    except Exception:  # noqa: BLE001
        return {n: list(edges) for n, edges in H.nodes.memberships.items()}


def incidence(H: Any, *, weights: bool = False) -> tuple[Any, list[Any], list[Any]]:
    """返回 ``(矩阵, 行索引=结点, 列索引=超边)``。"""
    return H.incidence_matrix(index=True, weights=weights)


def node_neighbors(H: Any, node: Any) -> set[Any]:
    """返回与 ``node`` 至少共享一条超边的其他结点。"""
    try:
        from honet._topology import topology

        return set(topology(H).neighbors_of(node))
    except Exception:  # noqa: BLE001
        neighbors: set[Any] = set()
        for edge in H.nodes.memberships[node]:
            neighbors.update(H.edges[edge])
        neighbors.discard(node)
        return neighbors


def edge_neighbors(H: Any, edge: Any) -> set[Any]:
    """返回与 ``edge`` 至少共享一个结点的其他超边。"""
    neighbors: set[Any] = set()
    for node in H.edges[edge]:
        neighbors.update(H.nodes.memberships[node])
    neighbors.discard(edge)
    return neighbors


def to_hypergraph(edges: Any, *, directed: bool = False) -> Any:
    """由 ``{边名: 成员}`` 或边列表构造 hypernetx 超图。"""
    import hypernetx as hnx

    return hnx.Hypergraph(edges, directed=directed)
