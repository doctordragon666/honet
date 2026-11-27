"""hypernetx 超图 → XGI 超图。

XGI 的结点用整数下标标识，因此转换时把结点名按 ``H.nodes`` 的顺序编号成
``0..n-1``，同时返回「下标 → 结点名」的映射，供调用方还原结果。超边权重
沿用 :func:`honet.convert.get_hyperedge_weights` 的口径（关联矩阵列和）。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require
from honet.convert._weights import edge_weight_sums


def convert_hypernetx_to_xgi(hnx_g: Any) -> tuple[Any, dict[int, Any]]:
    """把 hypernetx 超图转换为 XGI 超图。

    结点按 ``list(hnx_g.nodes)`` 的顺序编号为整数，每条超边转换成一个
    ``(结点下标..., 权重)`` 元组，交给 ``xgi.Hypergraph.add_weighted_edges_from``。

    参数:
        hnx_g: 输入的 hypernetx 超图对象。

    返回:
        ``(xgi 超图, {结点下标: 原结点名})``。XGI 超边的编号是 ``0..m-1``，
        成员以集合存储，因此**结点顺序信息会丢失**（XGI 本身不保留超边内顺序）。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"]})
        >>> X, idx_to_node = convert_hypernetx_to_xgi(H)
        >>> sorted(idx_to_node.values())
        ['a', 'b', 'c']
    """
    xgi = require("xgi")

    # 1. 结点编号：XGI 要求结点是整数下标，这里保持 hypernetx 的结点顺序
    nodes = list(hnx_g.nodes)
    node_to_idx = {node: index for index, node in enumerate(nodes)}
    idx_to_node = dict(enumerate(nodes))

    # 2. 超边权重（关联矩阵列和）；取不到时按 XGI 的默认权重 1.0 处理
    edge_weights = edge_weight_sums(hnx_g)

    # 3. 每条超边转成 (结点下标..., 权重)，权重放在元组末尾
    xgi_edges: list[tuple[int, ...]] = []
    for edge in hnx_g.edges:
        edge_indices = [node_to_idx[node] for node in hnx_g.edges[edge]]
        edge_indices.append(edge_weights.get(edge, 1.0))
        xgi_edges.append(tuple(edge_indices))

    # 4. 逐条带权重加入 XGI 超图
    H_xgi = xgi.Hypergraph()
    H_xgi.add_weighted_edges_from(xgi_edges)

    return H_xgi, idx_to_node
