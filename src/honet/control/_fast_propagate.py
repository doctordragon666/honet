"""快速传播：从新激活的超边出发递归激活更多超边与结点。

这是贪心可达性算法的内层循环。与逐个超边扫描的朴素实现相比，它只检查「刚刚
新增的结点」所在的超边，因此在大超图上快得多，结果完全等价。
"""

from __future__ import annotations

from typing import Any


def _fast_propagate(
    accessible: set[Any],
    newly_activated: list[int],
    already_active: set[int],
    edge_missing: list[set[Any]],
    edge_heads: list[set[Any]],
    node_edges_list: list[list[int]],
) -> set[Any]:
    """从新激活的超边出发递归传播可达结点。

    参数:
        accessible: 当前已可达结点集合。
        newly_activated: 刚刚满足「尾结点全部可达」的超边下标列表。
        already_active: 已经处理过的超边下标集合。
        edge_missing: 每条超边还缺哪些尾结点（就地读取，不会被修改）。
        edge_heads: 每条超边的头结点集合。
        node_edges_list: 每个结点参与的超边下标列表。

    返回:
        传播后的可达结点集合（新集合，不修改 ``accessible``）。

    注意:
        ``already_active`` 只用于去重，本函数**不会**把传播过程中激活的超边写回
        调用方的 ``already_active``（旧实现即如此）；重复处理同一超边是幂等的，
        不影响结果。
    """
    result = set(accessible)
    to_process = list(newly_activated)
    processed = set(already_active)
    i = 0
    while i < len(to_process):
        edge_idx = to_process[i]
        i += 1
        processed.add(edge_idx)
        new_nodes = edge_heads[edge_idx] - result
        result.update(new_nodes)
        for node in new_nodes:
            for check_edge_idx in node_edges_list[node]:
                missing = edge_missing[check_edge_idx]
                if check_edge_idx not in processed and missing.issubset(result):
                    to_process.append(check_edge_idx)
                    processed.add(check_edge_idx)
    return result
