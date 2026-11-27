"""结点的加权度。

加权度把结点所在的每条超边的权重累加起来，是普通度在加权超图上的推广。

性能说明：无权超图上「加权度」恒等于「度」，因此直接复用拓扑索引里的度值，
不必构造整个关联矩阵再按行求和（后者要为每个结点 × 每条超边分配矩阵元素）。
只有检测到真实权重时才走关联矩阵路径。
"""

from __future__ import annotations

from typing import Any

from honet import _cache
from honet._deps import require
from honet._topology import topology


def _compute(H: Any) -> dict[Any, float]:
    """计算加权度：有权重时走关联矩阵，否则直接用度值。

    权重检测不能依赖 ``H.edges.weights``——hypernetx 2.4.3 没有这个属性
    （带 ``cell_weight_col`` 构建的超图也照样没有），据此判断会把加权超图
    误当成无权超图，得出错误的度值。这里直接看关联矩阵：只有当矩阵里所有
    非零元都是 1 时才走快速路径（此时加权度恒等于度）。
    """
    np = require("numpy")
    B, node_names, _ = H.incidence_matrix(index=True, weights=True)

    data = B.data if hasattr(B, "data") else np.asarray(B).ravel()
    if data.size and not np.all(data == 1):
        values = np.asarray(B.sum(axis=1)).flatten()
        return {node: float(w) for node, w in zip(node_names, values, strict=True)}

    # 无权（或权重全为 1）：加权度等于度，用索引里的度值即可
    degrees = topology(H).degrees
    return {node: float(degrees.get(node, 0)) for node in node_names}


def weighted_degree(H: Any) -> dict[Any, float]:
    """计算超图中每个结点的加权度。

    加权度定义为结点所在全部超边的权重之和，等价于关联矩阵按行求和。
    无权超图上它退化为普通的度。

    参数:
        H: hypernetx 超图对象（构建时可用 ``cell_weight_col`` 携带权重）。

    返回:
        ``{结点: 加权度}``，值为 ``float``。

    说明:
        结果按超图结构指纹缓存；结构未变时重复调用直接返回上次结果。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"]})
        >>> weighted_degree(H)["a"]
        1.0
    """
    return _cache.cached(H, "weighted_degree:v2", lambda: _compute(H))
