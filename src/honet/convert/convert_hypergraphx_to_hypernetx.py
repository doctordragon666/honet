"""HyperGraphX 超图 → hypernetx 超图。

HyperGraphX 的边以成员元组标识，没有边名；转换时按出现顺序重新编号为
``e1``、``e2``…（沿用旧实现的编号方式）。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require


def convert_hypergraphx_to_hypernetx(hgx_hypergraph: Any) -> Any:
    """把 HyperGraphX 超图转换为 hypernetx 超图。

    参数:
        hgx_hypergraph: 输入的 ``hypergraphx.Hypergraph`` 对象；类型不符时
            抛 ``TypeError`` （与旧实现一致）。

    返回:
        hypernetx 超图对象，边名为 ``e1`` 、``e2``…，成员顺序与 HyperGraphX
        边元组一致。**边权重不迁移** （旧实现同样只迁移结构）。

    示例:
        >>> import hypergraphx as hgx
        >>> G = hgx.Hypergraph([("a", "b", "c")], weighted=False)
        >>> H = convert_hypergraphx_to_hypernetx(G)
        >>> list(H.edges)
        ['e1']
    """
    hgx = require("hypergraphx")
    hnx = require("hypernetx")

    if not isinstance(hgx_hypergraph, hgx.Hypergraph):
        raise TypeError("输入必须是 hypergraphx.Hypergraph 对象")

    # 按出现顺序重新编号：HyperGraphX 没有边名，用 e1/e2… 生成
    edge_dict: dict[str, list[Any]] = {}
    for edge in hgx_hypergraph.edges:
        edge_id = f"e{len(edge_dict) + 1}"
        edge_dict[edge_id] = list(edge)

    return hnx.Hypergraph(edge_dict)
