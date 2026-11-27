"""超边指标：按超边规模分组的结点平均度。"""

from __future__ import annotations

from typing import Any

from honet._deps import require


def get_avg_degree_by_edge_size(H: Any) -> dict[int, float]:
    """计算每种超边大小 m 下，全部结点的平均度数。

    用关联矩阵按列求和得到每条超边的大小，再按大小分组对行求和，最后对
    全部结点（含在该组内度为 0 的结点）取平均。

    参数:
        H: hypernetx 超图对象。

    返回:
        ``{超边大小: 结点平均度}``，键为 ``int``，值为 ``float``。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c", "d"]})
        >>> sorted(get_avg_degree_by_edge_size(H))
        [2, 3]
    """
    np = require("numpy")

    M, _, _ = H.incidence_matrix(index=True)
    dense = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    edge_sizes = dense.sum(axis=0)

    result: dict[int, float] = {}
    for size in np.unique(edge_sizes):
        mask = edge_sizes == size
        node_degrees = dense[:, mask].sum(axis=1)
        result[int(size)] = float(np.mean(node_degrees))

    return result
