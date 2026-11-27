"""诱导子超图的私有辅助函数（不对外导出）。

社区划分的多处实现需要「把某个社区单独拎出来再看内部结构」：递归划分要判断
社区能否继续拆分，层次聚类要按社区内超边做多数票。这里的 ``induced_edges``
统一提供该能力。

注意：不能直接用 hypernetx 的 ``H.restrict_to_nodes(...)``——它只裁剪超边的
成员，仍保留「一部分成员在社区外」的超边，导致子超图里出现社区外的结点。
本函数只保留**完全落在**给定结点集内的超边，与图论中诱导子图的定义一致。
"""

from __future__ import annotations

from typing import Any

from honet._hnx import edge_members


def induced_edges(hnx_g: Any, nodes: Any) -> dict[Any, list[Any]]:
    """返回只保留完全落在 ``nodes`` 内的超边的边字典。

    参数:
        hnx_g: hypernetx 超图对象。
        nodes: 结点集合（任意可迭代对象，内部会转成 ``set``）。

    返回:
        ``{超边名: [成员, ...]}``，只含成员数不少于 2 的超边。成员数不足 2 的
        超边无法构成超图意义上的交互，丢弃后与 ``hypergraphx`` 的边语义一致。
        没有合格超边时返回空字典。
    """
    keep = set(nodes)
    result: dict[Any, list[Any]] = {}
    for edge_id, members in edge_members(hnx_g).items():
        inside = [node for node in members if node in keep]
        if len(inside) >= 2:
            result[edge_id] = inside
    return result
