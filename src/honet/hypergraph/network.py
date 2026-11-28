"""honet.hypergraph.network —— 超图网络及其加权 / 有向变体。

超图以**超边**为存储单位，大小任意且不要求向下封闭：加入 ``{a, b, c}``
不会自动产生 ``{a, b}`` 等子边。注意与轻量容器 :class:`honet.Hypergraph` 的区别：
后者是通用数据结构，本模块是类层次中“形态明确”的超图网络。

本模块同时定义四个类（同文件内先基类、后变体，天然无循环导入）：

* :class:`HypergraphNetwork` —— 无权无向；
* :class:`HypergraphNetworkW` —— 加权；
* :class:`HypergraphNetworkD` —— 有向；
* :class:`HypergraphNetworkWD` —— 加权有向。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from honet._frame import group_rows
from honet._types import Members, Node
from honet.base import HigherOrderNetwork
from honet.base.mixins import DirectedMixin, WeightedMixin

if TYPE_CHECKING:
    import pandas as pd


class HypergraphNetwork(HigherOrderNetwork):
    """超图网络：以**超边**为存储单位，大小任意且不要求向下封闭。

    与 ``SimplicialComplexNetwork`` 的差别：加入 ``{a, b, c}`` **不会** 自动产生
    ``{a, b}`` 等子边；只有显式加入的边才存在。

    Args:
        nodes: 孤立结点。
        edges: 初始超边。

    用法::

        hg = HypergraphNetwork(edges=[["a", "b", "c"], ["c", "d"]])
        hg.num_edges      # 2
        hg.edge_sizes     # (3, 2)
    """

    def __init__(
        self,
        nodes: Iterable[Node] = (),
        edges: Iterable[Iterable[Node]] = (),
    ) -> None:
        self._nodes: dict[Node, None] = {}
        # 超边列表：元素为 (成员, 权重)
        self._edges: list[tuple[Members, float]] = []

        for node in nodes:
            self._nodes[node] = None
        for edge in edges:
            self.add_edge(edge)

    @classmethod
    def from_dataframe(
        cls,
        frame: pd.DataFrame,
        *,
        edge_col: str = "edge",
        node_col: str = "node",
        weight_col: str | None = None,
    ) -> HypergraphNetwork:
        """由 pandas DataFrame 构造超图网络（长表：一行 = 结点属于某条超边）。

        ``edge_col`` 取值相同的行合并为一条超边；不会向下封闭出子边。

        Args:
            frame: 输入表。
            edge_col: 超边标识列。
            node_col: 结点列。
            weight_col: 可选权重列，同一超边内必须一致，缺省为 1.0。

        Returns:
            新的超图网络，超边按首次出现顺序排列。

        Raises:
            KeyError: 缺少指定列。
            ValueError: 标识或结点为空、同组权重冲突、超边重复。
        """
        graph = cls()
        groups = group_rows(
            frame,
            key_cols=(edge_col,),
            node_col=node_col,
            weight_col=weight_col,
        )
        for _, members, weight, _ in groups:
            graph.add_edge(members, weight=weight)
        return graph

    def add_edge(self, members: Iterable[Node], weight: float = 1.0) -> Members:
        """加入一条超边；成员不存在时自动补建结点。

        Args:
            members: 超边成员。
            weight: 仅加权变体可用，默认 1.0。

        Returns:
            规范化后的超边成员表示。

        Raises:
            ValueError: 成员为空（无向）/不足两个或重复（有向）/重复超边。
            TypeError: 无权变体传入非 1.0 权重。
        """
        ordered = self._ordered_unique(members)
        # 用原始序列做方向校验（保留重复项，便于识别自环）
        edge = self._prepare_members(tuple(members))
        prepared_weight = self._prepare_weight(weight)

        if any(existing == edge for existing, _ in self._edges):
            raise ValueError(f"重复的超边：{edge!r}")

        for node in ordered:
            self._register(node)
        self._edges.append((edge, prepared_weight))
        return edge

    @property
    def edge_sizes(self) -> tuple[int, ...]:
        """各超边大小，顺序与 :attr:`edges` 一致。"""
        return tuple(len(members) for members, _ in self._edges)

    @property
    def is_uniform(self) -> bool:
        """是否均匀超图（无超边时视为均匀）。"""
        return len({len(members) for members, _ in self._edges}) <= 1

    @property
    def weights(self) -> tuple[float, ...]:
        """各超边权重（无权变体恒为 1.0）。"""
        return tuple(weight for _, weight in self._edges)

    @property
    def nodes(self) -> tuple[Node, ...]:
        """全部结点。"""
        return tuple(self._nodes)

    @property
    def edges(self) -> tuple[Members, ...]:
        """全部超边。"""
        return tuple(members for members, _ in self._edges)

    @property
    def num_nodes(self) -> int:
        """结点数。"""
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        """超边数。"""
        return len(self._edges)

    def degree(self, node: Node) -> int:
        """高阶度：包含该结点的超边条数。

        Raises:
            KeyError: 结点不存在。
        """
        if node not in self._nodes:
            raise KeyError(f"结点不存在：{node!r}")
        return sum(1 for members, _ in self._edges if node in members)

    def incidence_matrix(self) -> list[list[int]]:
        """结点 × 超边 的关联矩阵。"""
        return [[1 if node in members else 0 for members, _ in self._edges] for node in self._nodes]


class HypergraphNetworkW(WeightedMixin, HypergraphNetwork):
    """加权超图：每条超边带权重（默认 1.0）。"""


class HypergraphNetworkD(DirectedMixin, HypergraphNetwork):
    """有向超图：超边成员有序（如 a→b→c），拒绝自环与重复成员。"""


class HypergraphNetworkWD(WeightedMixin, DirectedMixin, HypergraphNetwork):
    """加权有向超图。"""


__all__ = [
    "HypergraphNetwork",
    "HypergraphNetworkD",
    "HypergraphNetworkW",
    "HypergraphNetworkWD",
]
