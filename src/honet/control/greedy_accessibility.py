"""贪心补齐驱动结点，使全系统可达。"""

from __future__ import annotations

from typing import Any

from honet.control._greedy_accessibility import greedy_accessibility as _impl


def greedy_accessibility(HG_local: Any, initial_drivers: Any = None) -> set[Any]:
    """在初始驱动结点基础上，贪心选择最少的额外驱动结点使全系统可达。

    每轮在剩余不可达结点里挑「能带来最多新增可达结点」的候选，直到全部可达。
    这是启发式算法，结果不一定全局最优，但与旧实现逐轮一致。

    参数:
        HG_local: HAT 有向超图。
        initial_drivers: 初始驱动结点下标集合；``None`` 表示从空集开始。

    返回:
        最终驱动结点下标集合（包含 ``initial_drivers``）。

    示例:
        >>> greedy_accessibility(HG, {0})  # doctest: +SKIP
        {0, 2}
    """
    return _impl(HG_local, initial_drivers)
