"""按超边大小分组的结点平均度（私有，不对外导出）。

``honet.metrics`` 子包里已有同口径的公开函数 ``get_avg_degree_by_edge_size``；
这里保留一份内部实现，是为了让 ``honet.contagion`` 在只加载自身时也能完成
稳定性判定，不引入跨子包的模块级依赖。两处口径必须保持一致：
用关联矩阵按列求和得到超边大小，按大小分组对行求和，再对**全部结点**
（含该组内度为 0 的结点）取平均。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require


def avg_degree_by_edge_size(H: Any) -> dict[int, float]:
    """返回 ``{超边大小: 结点平均度}``。

    参数:
        H: hypernetx 超图对象。

    返回:
        键为超边大小 ``int``，值为该大小超边下全部结点的平均度数 ``float``。
    """
    np = require("numpy")

    matrix, _, _ = H.incidence_matrix(index=True)
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    edge_sizes = dense.sum(axis=0)

    result: dict[int, float] = {}
    for size in np.unique(edge_sizes):
        node_degrees = dense[:, edge_sizes == size].sum(axis=1)
        result[int(size)] = float(np.mean(node_degrees))
    return result
