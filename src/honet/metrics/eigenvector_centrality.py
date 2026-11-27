"""结点的特征向量中心性。

以归一化拉普拉斯矩阵的第二小特征值对应的特征向量（Fiedler 向量）度量
结点在高阶连通结构中的位置，绝对值越大表示越靠近结构的「分界」。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require


def eigenvector_centrality(H: Any) -> dict[Any, float]:
    """基于归一化拉普拉斯的 Fiedler 向量计算结点中心性。

    参数:
        H: hypernetx 超图对象。

    返回:
        ``{结点: 中心性}``，按拉普拉斯矩阵的行列顺序对齐到结点标签。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> set(eigenvector_centrality(H)) == set(H.nodes)
        True
    """
    np = require("numpy")
    hnx = require("hypernetx")

    L, nodes = hnx.norm_lap(H)
    dense = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
    eigenvalues, eigenvectors = np.linalg.eig(dense)

    order = np.argsort(eigenvalues)
    # 取第二小特征值对应的特征向量；单结点图没有第二小，退回第一列
    index = order[1] if len(order) > 1 else order[0]
    fiedler = np.asarray(eigenvectors[:, index]).flatten().real

    return {node: float(fiedler[i]) for i, node in enumerate(nodes)}
