"""高阶网络的度分布熵。

熵衡量「超度分布有多不均匀」：全部结点超度相同（规则超图）时熵最大，少数
结点垄断全部关联时熵接近 0。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require
from honet._hnx import node_degrees


def calc_entropy(H: Any) -> float:
    r"""计算超图的度分布香农熵。

    以结点超度 :math:`k_i` 为分布，取

    .. math::

        S = -\sum_i p_i \ln p_i, \qquad p_i = \frac{k_i}{\sum_j k_j}

    参数:
        H: hypernetx 超图对象。

    返回:
        熵值 ``float``；没有任何关联（度之和为 0）时返回 ``0.0``。

    说明:
        旧实现调用 ``hypernetx.dist_stats(H)['node degree list']`` 取度序列，
        但 ``dist_stats`` 不属于 hypernetx 的稳定接口（2.x 各小版本字段名
        有变动），因此这里改用 :func:`honet._hnx.node_degrees` 直接求度。
        两者在 2.4.3 上给出同一组度值，熵是顺序无关量，结果完全一致。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> round(calc_entropy(H), 4)
        1.0397
    """
    np = require("numpy")

    degrees = np.asarray(list(node_degrees(H).values()), dtype=float)
    total = degrees.sum()
    if total <= 0.0:
        return 0.0

    p = degrees / total
    if p.size <= 1:
        # 只有一个结点（或只有一条关联）时分布退化，熵为 0；单独返回避免 -0.0
        return 0.0
    # 度为 0 的结点 p=0，0·ln0 约定为 0，用掩码排除避免 log(0) 警告
    positive = p[p > 0.0]
    return float(-np.sum(positive * np.log(positive)))
