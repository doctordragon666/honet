"""贪心可达性：用最少的额外驱动结点让全系统可达。

先用初始驱动结点做一次传播，之后每轮在「剩余不可达结点」里挑一个能带来最多
新增可达结点的候选，直到全部结点可达。是启发式算法，不保证全局最优。
"""

from __future__ import annotations

from typing import Any

from honet.control._fast_propagate import _fast_propagate
from honet.control._hat_edges import edge_head_tail, node_edge_index


def greedy_accessibility(HG_local: Any, initial_drivers: Any = None) -> set[Any]:
    """在初始驱动结点基础上，贪心地补齐最少的额外驱动结点使全系统可达。

    参数:
        HG_local: HAT 有向超图。
        initial_drivers: 初始驱动结点下标集合；``None`` 表示从空集开始。

    返回:
        最终的驱动结点下标集合（含 ``initial_drivers``）。

    注意:
        结点到超边的映射优先取 ``nodes['edges']`` 列（:func:`honet.control.add_node_fields`
        写入），该列缺失时按边表现算，因此本函数不强制先调用 ``add_node_fields``
        （旧实现会抛 ``KeyError``）。
    """
    all_sys_nodes = set(range(HG_local.nnodes))
    ends = edge_head_tail(HG_local)
    edge_tails = [set(tails) for _, tails in ends]
    edge_heads = [set(heads) for heads, _ in ends]
    node_edges = node_edge_index(HG_local)
    node_edges_list = [node_edges.get(node, []) for node in range(HG_local.nnodes)]

    edge_missing = [set(tail) for tail in edge_tails]
    current_drivers = set(initial_drivers) if initial_drivers is not None else set()
    accessible = set(current_drivers)
    active_edges: set[int] = set()

    # 初始驱动结点先把各自参与的边「点亮」
    for node in list(accessible):
        for edge_idx in node_edges_list[node]:
            edge_missing[edge_idx].discard(node)
            if not edge_missing[edge_idx] and edge_idx not in active_edges:
                active_edges.add(edge_idx)
                accessible.update(edge_heads[edge_idx])

    remaining = all_sys_nodes - (accessible & all_sys_nodes)
    while remaining:
        best_node = None
        best_size = len(remaining)
        best_new_accessible = None

        # 优先从「只差自己就能点亮一条边」的结点里挑候选，没有则退回全部剩余结点
        candidate_pool = set()
        for v in remaining:
            for edge_idx in node_edges_list[v]:
                if edge_idx not in active_edges and edge_missing[edge_idx] == {v}:
                    candidate_pool.add(v)
                    break
        if not candidate_pool:
            candidate_pool = remaining

        for cand in candidate_pool:
            new_accessible = accessible | {cand}
            newly_activated = [
                edge_idx
                for edge_idx in node_edges_list[cand]
                if edge_idx not in active_edges and edge_missing[edge_idx] <= new_accessible
            ]
            if newly_activated:
                new_accessible = _fast_propagate(
                    new_accessible,
                    newly_activated,
                    active_edges,
                    edge_missing,
                    edge_heads,
                    node_edges_list,
                )
            new_remaining_size = len(all_sys_nodes - (new_accessible & all_sys_nodes))
            if new_remaining_size < best_size:
                best_node = cand
                best_size = new_remaining_size
                best_new_accessible = new_accessible

        if best_node is None:  # pragma: no cover - 剩余结点总能让自己变得可达
            # 旧实现此处会 current_drivers.add(None) 并陷入死循环，这里改为退出
            break

        current_drivers.add(best_node)
        accessible = best_new_accessible
        for edge_idx in node_edges_list[best_node]:
            edge_missing[edge_idx].discard(best_node)
            if not edge_missing[edge_idx] and edge_idx not in active_edges:
                active_edges.add(edge_idx)
        remaining = all_sys_nodes - (accessible & all_sys_nodes)

    return current_drivers
