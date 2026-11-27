"""按结点超度分组的余平均度，用于判断是否存在度相关性。

做法：把结点按超度 ``k`` 分组，看每组内的余平均度（邻居的平均超度）是否随
``k`` 变化。若余平均度对 ``k`` 是常数，说明超度与邻居超度无关（无度相关性）；
若单调变化，则存在同配（随 k 上升）或异配（随 k 下降）。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from honet._deps import require
from honet._topology import topology
from honet.metrics._reuse import package_function


def assortativity_test(H: Any) -> dict[int, float]:
    """返回每个超度取值下结点的平均余平均度。

    参数:
        H: hypernetx 超图对象。

    返回:
        ``{超度: 该度全部结点的余平均度均值}``，键按超度升序排列。

    说明:
        余平均度直接复用 :func:`honet.metrics.excess_average_degree`，因此与
        全局余平均度口径完全一致；度值取自共享拓扑索引
        （:mod:`honet._topology`），不再逐点求度。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> sorted(assortativity_test(H))
        [1, 2]
    """
    np = require("numpy")

    # 度取自共享拓扑索引；余平均度复用局部指标（其结果同样带缓存）
    degrees = topology(H).degrees
    excess = package_function("honet.metrics", "excess_average_degree")(H)

    # 1) 按超度分桶收集余平均度
    buckets: dict[int, list[float]] = defaultdict(list)
    for node, degree in degrees.items():
        buckets[int(degree)].append(excess.get(node, 0.0))

    # 2) 每个超度取平均；键升序，便于直接观察随 k 的变化趋势
    return {degree: float(np.mean(values)) for degree, values in sorted(buckets.items())}
