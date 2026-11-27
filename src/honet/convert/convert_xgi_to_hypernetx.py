"""XGI 超图 → hypernetx 超图。

XGI 的 ``Hypergraph.edges.members`` 在不同版本里形态不一致：实测 xgi 1.1 是
**方法** ``members()``，而旧库 hypertrade 的注释记录过早期版本是直接返回
``{边: 成员}`` 的**属性**。这里用 ``getattr`` 取到后判断是否可调用，把三种
形态（方法 / 属性字典 / 属性列表）收敛成一份 ``{边标识: 成员列表}`` 字典，
再交给 hypernetx 构造超图。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require


def convert_xgi_to_hypernetx(xgi_hg: Any) -> Any:
    """把 XGI 超图转换为 hypernetx 超图。

    边标识直接沿用 XGI 的边 id；取不到 ``members`` 映射时退化为逐条边调用
    ``members(edge_id)``。

    参数:
        xgi_hg: 输入的 XGI 超图对象。

    返回:
        hypernetx 超图对象。**XGI 的边权重不迁移**（旧实现同样只迁移结构）；
        需要权重时请用 ``edge_properties`` 显式挂载。

    兼容性:
        实测 xgi 1.1（``edges.members`` 是方法）。对属性式 ``edges.members``
        也做了兼容：属性值既可以是 ``{边: 成员}`` 字典，也可以是「成员列表的
        列表」（此时按 ``e0``、``e1``… 自动编号）。

    示例:
        >>> import xgi
        >>> X = xgi.Hypergraph()
        >>> X.add_weighted_edges_from([(0, 1, 2, 1.0)])
        >>> H = convert_xgi_to_hypernetx(X)
        >>> H.edges.incidence_dict[0]
        [0, 1, 2]
    """
    hnx = require("hypernetx")

    # 1. 取出 members：可能是方法（xgi 1.x），也可能是属性（早期版本）
    members = getattr(xgi_hg.edges, "members", None)
    edges_obj = members() if callable(members) else members

    # 2. 归一化为 {边标识: 成员列表}
    hyperedges: dict[Any, list[Any]] = {}
    if isinstance(edges_obj, dict):
        # 方法/属性直接给出 {边: 成员} 映射
        hyperedges = {edge_id: list(nodes) for edge_id, nodes in edges_obj.items()}
    elif isinstance(edges_obj, list | tuple):
        # 只有成员列表、没有边标识：按位置生成 e0、e1… 作为边名
        hyperedges = {f"e{index}": list(nodes) for index, nodes in enumerate(edges_obj)}
    else:
        # 兜底：逐条边按 id 查询成员（要求 members 可调用）
        hyperedges = {edge_id: list(members(edge_id)) for edge_id in xgi_hg.edges}

    return hnx.Hypergraph(hyperedges)
