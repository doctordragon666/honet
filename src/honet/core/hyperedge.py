"""honet.core.hyperedge —— 不可变超边类型。

一条超边表示“任意数量的结点构成的一个整体”，用于刻画高阶交互
（例如一篇论文的合作者集合、一次多体碰撞）。

被以下文件使用：``src/honet/core/hypergraph.py`` （容器存储）、
``src/honet/core/__init__.py``（re-export）、``src/honet/utils/*``。
"""

from __future__ import annotations

from dataclasses import dataclass

from honet._types import Node


@dataclass(frozen=True, slots=True)
class Hyperedge:
    """一条超边，表示若干结点之间的一次高阶交互。

    Attributes:
        members: 超边包含的结点集合，构造时由调用方传入的可迭代对象转换为
            ``frozenset`` （自动去重、无序）。不可为空。
        label: 可选的超边标签，用于区分成员相同但语义不同的高阶交互
            （例如“同一批作者的 2023 年论文”与“2024 年论文”）。
            为 ``None`` 表示无标签；传入空字符串或纯空白字符串会报错。
        weight: 超边权重，默认 ``1.0``，必须为正的有限实数。用于表达高阶交互的
            强度／频次（如同一组作者合作了 3 篇论文 → ``weight=3``）。
            权重不参与超边身份判定，只参与结构相等比较。

    用法::

        edge = Hyperedge(frozenset({"a", "b", "c"}), label="paper-1", weight=2.0)
        len(edge)          # 3
        "b" in edge        # True
        edge.weight        # 2.0

    Raises:
        ValueError: ``members`` 为空、``label`` 为空白字符串，或 ``weight`` 非正/非有限。
    """

    members: frozenset[Node]
    label: str | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        """构造后校验：超边非空、标签非空白、权重为正的有限数。

        dataclass 在 ``__init__`` 之后自动调用本方法，因此校验对直接构造
        ``Hyperedge(...)`` 与经由 ``Hypergraph.add_edge`` 两条路径都生效。
        """
        # 空超边没有物理意义，且会让 ``is_uniform`` 等推断出现歧义
        if not self.members:
            raise ValueError("超边至少需要包含一个结点")

        # 标签要么不设置，要么必须是有效文本，避免出现 "  " 这种脏数据
        if self.label is not None and not self.label.strip():
            raise ValueError("超边标签不能是空字符串或纯空白字符串")

        # 权重必须在 (0, +∞) 内且非 NaN：``w != w`` 是 NaN 的判据，
        # 用 ``not (w > 0)`` 同时挡住 0、负数与 NaN
        if not (isinstance(self.weight, (int, float)) and self.weight == self.weight):
            raise ValueError(f"超边权重必须是实数：{self.weight!r}")
        if not self.weight > 0:
            raise ValueError(f"超边权重必须为正数：{self.weight!r}")

    @property
    def size(self) -> int:
        """超边的大小，即成员结点个数 ``|e|``。"""
        return len(self.members)

    def __contains__(self, node: object) -> bool:
        """支持 ``node in edge`` 语法，判断结点是否属于该超边。"""
        return node in self.members

    def __len__(self) -> int:
        """``len(edge)`` 返回成员个数，与 :attr:`size` 等价。"""
        return len(self.members)


__all__ = ["Hyperedge"]
