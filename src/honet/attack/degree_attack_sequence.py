"""基于度的攻击序列。

定向攻击的核心思想：先移除「结构上最重要」的结点或超边。度是其中最直接
的排序依据——超度越高的结点参与的多元交互越多，移除它造成的破坏通常越大。
除超度外，本模块还支持按余平均度（邻居的平均超度）与聚类系数排序，
用于考察「处在高阶交互密集区」与「处在三角闭合结构中心」的结点。

所有度量都经 ``honet._hnx`` 的兼容函数取得，不直接使用 hypernetx 1.x 的
``H.edges.elements`` 等已被 2.x 移除的接口。
"""

from __future__ import annotations

from typing import Any

from honet._hnx import edge_sizes, node_degrees
from honet.metrics.clustering_coefficient import clustering_coefficient
from honet.metrics.excess_average_degree import excess_average_degree

#: 支持的排序依据（与 ``target`` 参数取值一致）
_STRATEGIES = ("node", "edge", "excess", "cluster")


def degree_attack_sequence(hg: Any, target: str = "node") -> list[Any]:
    """生成基于度的攻击序列（降序）。

    参数:
        hg: hypernetx 超图对象。
        target: 攻击目标与排序依据，可选值：

            * ``"node"`` —— 按结点超度降序；
            * ``"edge"`` —— 按超边基数（成员数）降序；
            * ``"excess"`` —— 按结点余平均度降序；
            * ``"cluster"`` —— 按结点聚类系数降序。

    返回:
        结点（或超边）标识列表，按对应度量从大到小排列。

    抛出:
        ValueError: ``target`` 不在支持范围内。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["c", "d"]})
        >>> degree_attack_sequence(H)[0]
        'c'
    """
    if target == "node":
        scores = node_degrees(hg)
    elif target == "edge":
        # 超边的「度」取其基数，即成员数量
        scores = edge_sizes(hg)
    elif target == "excess":
        scores = excess_average_degree(hg)
    elif target == "cluster":
        scores = clustering_coefficient(hg)
    else:
        msg = f"target 只能是 {_STRATEGIES} 之一，收到 {target!r}"
        raise ValueError(msg)

    return [item for item, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
