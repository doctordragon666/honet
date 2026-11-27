"""超边权重（关联矩阵按列求和）。

超边权重定义为「该超边所有结点-超边关联权重之和」，也就是关联矩阵
``B``（行=结点，列=超边）中该列元素之和。没有显式给出关联权重时，每个
结点贡献默认权重 1，此时超边权重就等于超边大小。
"""

from __future__ import annotations

from typing import Any

from honet.convert._weights import edge_weight_sums


def get_hyperedge_weights(H: Any) -> dict[Any, float]:
    """获取超图中每条超边的权重。

    权重取关联矩阵的列和：

    .. math::

        w(e) = \\sum_{v \\in e} B[v, e]

    其中 ``B[v, e]`` 是结点 ``v`` 在超边 ``e`` 上的关联权重（hypernetx 的
    cell weight，构造超图时通过 ``cell_weight_col`` 指定）。

    参数:
        H: hypernetx 超图对象。

    返回:
        ``{超边标识: 权重}``，权重统一为 Python ``float``；超边顺序与
        ``H.incidence_matrix(index=True)`` 的列顺序一致。

    抛出:
        ImportError: 未安装 numpy（需要 ``honet[analysis]``）。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"]})
        >>> get_hyperedge_weights(H)
        {'e1': 3.0}
    """
    return edge_weight_sums(H)
