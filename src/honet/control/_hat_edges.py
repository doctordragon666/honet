"""从 HAT 超图的边表读取有向端点。

HAT（HypergraphAnalysisToolbox）用「结点表 + 边表」描述超图：有向超图的边表
含 ``head`` / ``tail`` 两列。注意方向信息**只存在于边表的列里**，``Hypergraph``
对象本身没有 ``head`` / ``tail`` 属性。

本模块把边表统一读成 Python 列表，供 ``_star_graph`` / ``_reachability`` /
``_greedy_accessibility`` 共用，避免各实现模块重复处理 numpy 数组。
"""

from __future__ import annotations

from typing import Any


def node_id(value: Any) -> Any:
    """把 HAT 的结点编号规整成 Python ``int``，非数值编号原样返回。

    HAT 内部会把结点编号成整数，这里只是把 ``numpy.int64`` 转成 ``int``，
    让公开函数的返回值可以直接序列化；对非数值标签保持原样而不是抛异常。
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def edge_head_tail(HG: Any) -> list[tuple[list[Any], list[Any]]]:
    """按边表行序返回 ``[(head 列表, tail 列表), ...]``。

    无向超图的边表只有 ``Nodes`` 列，此时头尾都取该边的全部成员（与
    :func:`honet.control.star_graph` 的无向分支一致）。

    参数:
        HG: HAT 超图对象。

    返回:
        长度等于 ``HG.nedges`` 的列表，每项是 ``(head, tail)`` 两个结点下标列表。
    """
    edges = HG.edges
    if "head" in edges.columns and "tail" in edges.columns:
        return [
            ([node_id(n) for n in head], [node_id(n) for n in tail])
            for head, tail in zip(edges["head"], edges["tail"], strict=True)
        ]
    members = [[node_id(n) for n in row] for row in edges["Nodes"]]
    return [(row, list(row)) for row in members]


def node_edge_index(HG: Any) -> dict[Any, list[int]]:
    """返回 ``{结点: [所属超边的行号, ...]}``。

    优先使用 :func:`honet.control.add_node_fields` 写入的 ``nodes['edges']``
    列；该列不存在时按边表的 ``Nodes`` 列现算，因此贪心算法不强制要求先调用
    ``add_node_fields``（旧实现会因此抛 ``KeyError``）。

    参数:
        HG: HAT 超图对象。

    返回:
        结点下标 → 超边行号列表（行号与 :func:`edge_head_tail` 的顺序一致）。
    """
    nodes = HG.nodes
    if "edges" in nodes.columns:
        return {
            node_id(n): list(edge_ids)
            for n, edge_ids in zip(nodes["Nodes"], nodes["edges"], strict=True)
        }

    index: dict[Any, list[int]] = {node_id(n): [] for n in nodes["Nodes"]}
    for edge_idx, members in enumerate(HG.edges["Nodes"]):
        for n in members:
            index.setdefault(node_id(n), []).append(edge_idx)
    return index
