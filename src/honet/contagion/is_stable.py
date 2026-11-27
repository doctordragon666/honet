"""SIS 传播模型的稳定性判定（基本再生数条件）。"""

from __future__ import annotations

from typing import Any

from honet.contagion._stability import stability_check


def is_stable(H: Any, tau: dict[int, float], gamma: float) -> bool:
    """判断 SIS 传播是否稳定（感染会衰减而非暴发）。

    判定式沿用旧实现：把等式右侧视为基本再生数的倒数形式，

    .. math::

        \\sum_{m \\ge 2} \\frac{(m-1)\\beta_m \\langle k \\rangle_m}{\\langle k \\rangle}
        < \\frac{\\gamma \\langle k \\rangle}{\\langle k \\rangle_2}

    其中 ``<k>`` 是全体结点的平均（超边）度，``<k>_m`` 是「仅计大小为 ``m``
    的超边」时全体结点的平均度，``beta_m`` 是大小为 ``m`` 的超边的感染率。

    参数:
        H: hypernetx 超图对象。
        tau: ``{超边大小: 感染率}``。
        gamma: 恢复率。

    返回:
        ``True`` 表示稳定（``LHS < RHS``）；没有结点、或图中没有大小为 2 的
        超边（``<k>_2 == 0``）时一律返回 ``True``，与旧实现一致。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c", "d"]})
        >>> is_stable(H, {2: 0.05, 3: 0.05}, gamma=0.5)
        True
    """
    return stability_check(H, tau, gamma)
