"""结点的加权度。

加权度把结点所在的每条超边的权重累加起来，是普通度在加权超图上的推广。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require


def weighted_degree(H: Any) -> dict[Any, float]:
    """计算超图中每个结点的加权度。

    加权度定义为结点所在全部超边的权重之和，等价于关联矩阵按行求和。

    参数:
        H: hypernetx 超图对象（构建时可用 ``cell_weight_col`` 携带权重）。

    返回:
        ``{结点: 加权度}``，值为 ``float``。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"]})
        >>> weighted_degree(H)["a"]
        1.0
    """
    np = require("numpy")
    B, node_names, _ = H.incidence_matrix(index=True, weights=True)
    weights = np.asarray(B.sum(axis=1)).flatten()
    return {node: float(w) for node, w in zip(node_names, weights, strict=True)}
