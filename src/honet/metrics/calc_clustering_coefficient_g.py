"""超图的全局聚类系数。

全局值是全部结点局部聚类系数的算术平均，因此口径完全由
:func:`honet.metrics.clustering_coefficient` 决定：邻居对是否至少共享一条
超边。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require
from honet.metrics._reuse import package_function


def calc_clustering_coefficient_g(H: Any) -> float:
    """计算超图的全局聚类系数。

    参数:
        H: hypernetx 超图对象。

    返回:
        全部结点局部聚类系数的算术平均；超图没有结点时返回 ``0.0``。

    说明:
        局部值直接复用 :func:`honet.metrics.clustering_coefficient`（纯 Python
        实现），因此本函数只额外依赖 numpy 求均值。旧实现调用 XGI 的
        ``clustering_coefficient``，其口径是「邻居对共享超边的平均比例」
        （按共享超边数加权），与这里的「邻居对是否连通」口径在超边重叠时
        数值不同；新口径与 honet 既有局部实现保持一致。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"]})
        >>> calc_clustering_coefficient_g(H)
        1.0
    """
    np = require("numpy")

    local = package_function("honet.metrics", "clustering_coefficient")(H)
    if not local:
        return 0.0
    return float(np.mean(list(local.values())))
