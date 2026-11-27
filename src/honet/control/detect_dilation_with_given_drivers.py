"""给定驱动结点集合时判断 dilation 是否仍存在。"""

from __future__ import annotations

from typing import Any

from honet.control._dilation import detect_dilation_with_given_drivers as _impl


def detect_dilation_with_given_drivers(HG_local: Any, drivers: Any) -> bool:
    """在给定驱动结点集合下检测 dilation 是否仍然存在。

    驱动结点相当于一条只指向自己的「虚拟超边」，因此实现上是在星形图里补
    ``u<i> -> n<i>`` 的边后再求最大匹配。

    参数:
        HG_local: HAT 有向超图。
        drivers: 候选驱动结点下标集合（任意可迭代对象）。

    返回:
        ``True`` 表示仍存在 dilation，即该驱动集合无法完全控制系统。

    示例:
        >>> detect_dilation_with_given_drivers(HG, {0})  # doctest: +SKIP
        False
    """
    return _impl(HG_local, drivers)
