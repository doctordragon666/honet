"""honet.temporal.network —— 时序高阶网络及其加权 / 有向变体。

时序网络把交互存成 ``(时间戳, 超边)`` 事件流，因此同一组结点可以在不同时间
重复出现——这是一次性超图无法表达的信息。

本模块同时定义四个类（同文件内先基类、后变体，天然无循环导入）：

* :class:`TemporalNetwork` —— 无权无向；
* :class:`TemporalNetworkW` —— 加权；
* :class:`TemporalNetworkD` —— 有向；
* :class:`TemporalNetworkWD` —— 加权有向。
"""

from __future__ import annotations

from collections.abc import Iterable

from honet._types import Members, Node
from honet.base import HigherOrderNetwork
from honet.base.mixins import DirectedMixin, WeightedMixin


class TemporalNetwork(HigherOrderNetwork):
    """时序高阶网络：交互以 ``(时间戳, 超边)`` 事件流存储，始终按时间升序。

    同一组结点可以在不同时间重复出现（重复判定只看 ``(时间, 成员)``），
    这正是一次性超图无法表达的信息。

    Args:
        nodes: 孤立结点。
        events: 初始事件，元素为 ``(时间戳, 成员)`` 二元组。

    用法::

        tn = TemporalNetwork(events=[(1.0, ["a", "b"]), (3.0, ["a", "b", "c"])])
        tn.times                  # (1.0, 3.0)
        tn.active_nodes(3.0)      # frozenset({'a', 'b', 'c'})
    """

    def __init__(
        self,
        nodes: Iterable[Node] = (),
        events: Iterable[tuple[float, Iterable[Node]]] = (),
    ) -> None:
        self._nodes: dict[Node, None] = {}
        # 事件列表：元素为 (时间, 成员, 权重)，始终按时间升序
        self._events: list[tuple[float, Members, float]] = []

        for node in nodes:
            self._nodes[node] = None
        for timestamp, members in events:
            self.add_event(members, time=timestamp)

    @staticmethod
    def _check_time(time: float) -> float:
        """校验时间戳：必须是实数（``bool`` 除外）。"""
        if isinstance(time, bool) or not isinstance(time, (int, float)) or time != time:
            raise ValueError(f"时间戳必须是实数：{time!r}")
        return float(time)

    def add_event(
        self,
        members: Iterable[Node],
        time: float,
        weight: float = 1.0,
    ) -> Members:
        """加入一个事件，插入后重新按时间排序（稳定排序，同一时刻保持加入顺序）。

        Args:
            members: 事件涉及的结点。
            time: 时间戳（实数，可为整数年/秒等）。
            weight: 仅加权变体可用，默认 1.0。

        Returns:
            规范化后的成员表示。

        Raises:
            ValueError: 时间戳非法、成员非法，或同一时刻已存在相同成员的事件。
            TypeError: 无权变体传入非 1.0 权重。
        """
        ordered = self._ordered_unique(members)
        # 用原始序列做方向校验（保留重复项，便于识别自环）
        event_members = self._prepare_members(tuple(members))
        prepared_weight = self._prepare_weight(weight)
        timestamp = self._check_time(time)

        if any(t == timestamp and members_ == event_members for t, members_, _ in self._events):
            raise ValueError(f"时间 {timestamp} 已存在相同事件：{event_members!r}")

        for node in ordered:
            self._register(node)
        self._events.append((timestamp, event_members, prepared_weight))
        # 稳定排序：同一时间戳的事件保持加入顺序
        self._events.sort(key=lambda item: item[0])
        return event_members

    @property
    def times(self) -> tuple[float, ...]:
        """全部事件时间戳（升序，含重复时刻）。"""
        return tuple(timestamp for timestamp, _, _ in self._events)

    def events_between(self, start: float, end: float) -> tuple[Members, ...]:
        """取 ``start <= t <= end`` 的事件成员。

        Args:
            start: 起始时间（含）。
            end: 结束时间（含）。

        Returns:
            命中的事件成员序列。
        """
        return tuple(members for timestamp, members, _ in self._events if start <= timestamp <= end)

    def active_nodes(self, time: float) -> frozenset[Node]:
        """某一时刻活跃（参与事件）的结点集合。

        Args:
            time: 目标时间戳（精确匹配）。

        Returns:
            该时刻全部事件成员构成的结点集合；无事件时为空集。
        """
        result: set[Node] = set()
        for timestamp, members, _ in self._events:
            if timestamp == time:
                result |= set(members)
        return frozenset(result)

    @property
    def nodes(self) -> tuple[Node, ...]:
        """全部结点。"""
        return tuple(self._nodes)

    @property
    def edges(self) -> tuple[Members, ...]:
        """全部事件成员（按时间升序）。"""
        return tuple(members for _, members, _ in self._events)

    @property
    def num_nodes(self) -> int:
        """结点数。"""
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        """事件数。"""
        return len(self._events)

    def degree(self, node: Node) -> int:
        """结点参与的事件数。

        Raises:
            KeyError: 结点不存在。
        """
        if node not in self._nodes:
            raise KeyError(f"结点不存在：{node!r}")
        return sum(1 for _, members, _ in self._events if node in members)

    def incidence_matrix(self) -> list[list[int]]:
        """结点 × 事件 的关联矩阵，列顺序为时间升序。"""
        return [
            [1 if node in members else 0 for _, members, _ in self._events] for node in self._nodes
        ]


class TemporalNetworkW(WeightedMixin, TemporalNetwork):
    """加权时序网络：每个事件带权重（默认 1.0）。"""


class TemporalNetworkD(DirectedMixin, TemporalNetwork):
    """有向时序网络：每个事件的成员有序一次传播方向。"""


class TemporalNetworkWD(WeightedMixin, DirectedMixin, TemporalNetwork):
    """加权有向时序网络。"""


__all__ = [
    "TemporalNetwork",
    "TemporalNetworkD",
    "TemporalNetworkW",
    "TemporalNetworkWD",
]
