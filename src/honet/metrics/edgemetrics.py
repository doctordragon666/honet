"""超边级指标：按超边属性分组，返回以分组为键的映射。

目前只有一个指标 :func:`get_avg_degree_by_edge_size`——按超边规模分组的结点
平均度。它把「结点—超边」关联按超边大小分桶，因此不必构造
``结点数 × 超边数`` 的稠密关联矩阵（大图上的主要开销）。
"""

from __future__ import annotations

from typing import Any

from honet import _cache
from honet._topology import Topology, topology


def _compute(top: Topology) -> dict[int, float]:
    """按超边规模分组，求全部结点在该组内的平均度。"""
    nodes = top.nodes
    n_nodes = len(nodes)
    if n_nodes == 0:
        return {}

    # 按超边大小累计每个结点在该组内的度数
    buckets: dict[int, dict[Any, int]] = {}
    for edge, members in top.members.items():
        size = top.sizes[edge]
        bucket = buckets.setdefault(size, {})
        for node in members:
            bucket[node] = bucket.get(node, 0) + 1

    result: dict[int, float] = {}
    for size, counts in buckets.items():
        # 分母是全部结点数：在该组内度为 0 的结点也计入平均
        result[size] = sum(counts.values()) / n_nodes
    return result


def get_avg_degree_by_edge_size(H: Any) -> dict[int, float]:
    """计算每种超边大小 m 下，全部结点的平均度数。

    对每个超边大小分组，统计结点在该组超边中的度，再对**全部结点**取平均
    （在该组内度为 0 的结点也计入分母）。

    参数:
        H: hypernetx 超图对象。

    返回:
        ``{超边大小: 结点平均度}``，键为 ``int``，值为 ``float``。

    说明:
        结果按超图结构指纹缓存；结构未变时重复调用直接返回上次结果。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c", "d"]})
        >>> sorted(get_avg_degree_by_edge_size(H))
        [2, 3]
    """
    return _cache.cached(H, "avg_degree_by_edge_size:v2", lambda: _compute(topology(H)))
