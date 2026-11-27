"""hypernetx 超图 → HyperGraphX 超图。

HyperGraphX 用「成员元组的有序列表 + 权重列表」描述超图，因此每条超边都
要转成元组并配一个权重。旧实现会丢掉成员数不足 2 的退化超边（HGX 的边是
成对以上的交互），这里保持一致。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require
from honet.convert._weights import edge_weight_sums


def convert_hypernetx_to_hypergraphx(hnx_g: Any) -> Any:
    """把 hypernetx 超图转换为 HyperGraphX 超图。

    超边成员数小于 2 的超边会被跳过（HyperGraphX 的边表示至少两个结点的
    交互）；其余超边按 ``H.edges`` 的顺序转成成员元组，权重取关联矩阵列和。

    参数:
        hnx_g: 输入的 hypernetx 超图对象。

    返回:
        ``hypergraphx.Hypergraph``，构造时 ``weighted=True``，权重列表与边
        列表一一对应。**超边名会丢失**：HyperGraphX 用成员元组本身标识边，
        不保留 hypernetx 的边名。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["c", "d"]})
        >>> G = convert_hypernetx_to_hypergraphx(H)
        >>> sorted(tuple(sorted(e)) for e in G.edges)
        [('a', 'b', 'c'), ('c', 'd')]
    """
    hgx = require("hypergraphx")

    # 1. 超边权重（关联矩阵列和）
    weights = edge_weight_sums(hnx_g)

    # 2. 过滤退化超边并收集成员元组 + 对应权重，两个列表必须严格等长
    edge_list: list[tuple[Any, ...]] = []
    valid_weights: list[float] = []
    for edge_id in hnx_g.edges:
        edge_nodes = tuple(hnx_g.edges[edge_id])
        if len(edge_nodes) >= 2:
            edge_list.append(edge_nodes)
            valid_weights.append(float(weights.get(edge_id, 1.0)))

    return hgx.Hypergraph(edge_list, weighted=True, weights=valid_weights)
