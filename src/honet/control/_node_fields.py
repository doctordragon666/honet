"""在 HAT 超图的结点表上登记每个结点参与的超边及其角色。

旧实现要求边表已经有 ``head`` / ``tail`` 列（即 HAT 有向超图），这里对无向超图
做退化处理：``head`` / ``tail`` 都填该边的全部成员，于是后续算法可以无条件读取
这两列，不必再判分支。
"""

from __future__ import annotations

from typing import Any

from honet.control._hat_edges import node_id


def add_node_fields(HG_local: Any) -> Any:
    """为结点表添加 ``edges`` / ``head`` / ``tail`` 三列（原地修改并返回同一对象）。

    参数:
        HG_local: HAT 超图对象，结点表需含 ``Nodes`` 列。

    返回:
        传入的同一个 ``HG_local``（原地修改，旧实现也是这个语义）。

    注意:
        三列的取值都是「超边行号列表」，与 ``HG_local.edges.index`` 一致。
        旧实现对边表的 ``Edges`` 列做 ``append``，这里统一成边表行号，因此即使
        边表被 ``compress`` 重排过也能对齐。
    """
    nodes = HG_local.nodes
    nodes["edges"] = [[] for _ in range(len(nodes))]
    nodes["head"] = [[] for _ in range(len(nodes))]
    nodes["tail"] = [[] for _ in range(len(nodes))]

    node_to_idx = {node_id(n): idx for n, idx in zip(nodes["Nodes"], nodes.index, strict=True)}
    for edge_idx, edge_row in HG_local.edges.iterrows():
        for n in edge_row["Nodes"]:
            nodes.at[node_to_idx[node_id(n)], "edges"].append(edge_idx)
        if "head" in edge_row:
            for n in edge_row["head"]:
                nodes.at[node_to_idx[node_id(n)], "head"].append(edge_idx)
            for n in edge_row["tail"]:
                nodes.at[node_to_idx[node_id(n)], "tail"].append(edge_idx)
        else:
            # 无向超图没有 head/tail 列：成员同时算作头与尾
            for n in edge_row["Nodes"]:
                nodes.at[node_to_idx[node_id(n)], "head"].append(edge_idx)
                nodes.at[node_to_idx[node_id(n)], "tail"].append(edge_idx)
    return HG_local
