"""超图的自然连通度。

自然连通度把邻接矩阵的全部特征值聚合成一个标量，值越大表示高阶连通结构
越冗余：去掉少量结点后仍能维持连通。

.. math::

    \\lambda = \\ln\\left(\\frac{1}{N}\\sum_{i=1}^{N} e^{\\lambda_i}\\right)
"""

from __future__ import annotations

from typing import Any

from honet._deps import require


def calc_natural_connectivity(H: Any) -> float:
    """计算超图的自然连通度。

    用 hypernetx 的团展开邻接矩阵（两结点共享超边则连一条边），取全部特征值
    :math:`\\lambda_i` 的指数平均的对数。

    参数:
        H: hypernetx 超图对象。

    返回:
        自然连通度 ``float``；超图没有结点时返回 ``0.0``。

    说明:
        旧实现写 ``H.adjacency_matrix().todense()``。hypernetx 2.4.3 的
        ``adjacency_matrix()`` 直接返回稀疏矩阵（不是 tuple），因此这里用
        ``toarray()`` 取稠密数组，同时兼容返回 tuple 的旧版本（取第一项）。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"]})
        >>> round(calc_natural_connectivity(H), 4)
        0.4338
    """
    np = require("numpy")

    if not H.nodes:
        return 0.0

    adjacency = H.adjacency_matrix()
    # hypernetx 2.x 返回稀疏矩阵；1.x 曾返回 (矩阵, 结点索引, 超边索引) 元组
    if isinstance(adjacency, tuple):
        adjacency = adjacency[0]
    dense = adjacency.toarray() if hasattr(adjacency, "toarray") else np.asarray(adjacency)

    eigenvalues = np.linalg.eigvalsh(dense)
    return float(np.log(np.mean(np.exp(eigenvalues))))
