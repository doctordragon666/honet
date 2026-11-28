"""honet.utils.incidence_matrix —— 关联矩阵。"""

from __future__ import annotations

from honet._types import Node
from honet.core import Hypergraph


def incidence_matrix(hg: Hypergraph) -> list[list[int]]:
    """计算超图的关联矩阵 ``B`` （incidence matrix）。

    定义：``B[i][j] = 1`` 当且仅当第 ``i`` 个结点属于第 ``j`` 条超边，否则为 0。
    矩阵形状为 ``num_nodes × num_edges``，行顺序与 :attr:`Hypergraph.nodes` 一致，
    列顺序与 :attr:`Hypergraph.edges` 一致。

    Args:
        hg: 待计算的超图。

    Returns:
        二维列表，``matrix[i][j] ∈ {0, 1}``。空超图返回 ``[]``。

    用法::

        hg = Hypergraph(nodes=["a", "b"], edges=[["a", "b"]])
        incidence_matrix(hg)     # [[1], [1]]
    """
    # 逐列（逐超边）填充，避免对每个格子重复做成员判断
    matrix = [[0] * hg.num_edges for _ in range(hg.num_nodes)]
    # 建立结点到行号的索引：字典查找 O(1)，整体复杂度 O(Σ|e|)
    row_of: dict[Node, int] = {node: i for i, node in enumerate(hg.nodes)}

    for col, edge in enumerate(hg.edges):
        for node in edge.members:
            matrix[row_of[node]][col] = 1
    return matrix


__all__ = ["incidence_matrix"]
