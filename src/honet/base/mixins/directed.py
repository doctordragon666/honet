"""honet.base.mixins.directed —— 有向能力 mixin。"""

from __future__ import annotations

from collections.abc import Iterable

from honet._types import Members, Node


class DirectedMixin:
    """有向能力 mixin：``is_directed=True``，成员按“起点 → 终点”顺序保存。

    覆盖 :meth:`_prepare_members`：保留顺序、要求至少两个结点、拒绝重复成员
    （因此自环 ``[a, a]`` 也会被拒绝）。
    """

    is_directed: bool = True

    def _prepare_members(self, members: Iterable[Node]) -> Members:
        """把成员规范化为有序元组并做方向性校验。

        Args:
            members: 成员可迭代对象，顺序即方向（如 ``["a", "b", "c"]`` 表示 a→b→c）。

        Returns:
            有序成员元组。

        Raises:
            ValueError: 成员少于两个，或存在重复结点（含自环）。
        """
        sequence = tuple(members)
        if len(sequence) < 2:
            raise ValueError("有向交互至少需要起点和终点两个结点")
        if len(set(sequence)) != len(sequence):
            raise ValueError(f"有向交互的成员不能重复（含自环）：{sequence!r}")
        return sequence


__all__ = ["DirectedMixin"]
