"""honet.multilayer.network —— 多层网络及其加权 / 有向变体。

多层网络的结点跨层共享，每层各自维护一组超边，因此“度”有两种口径：
全局度（跨层累加，:meth:`MultilayerNetwork.degree`）与层内度
（:meth:`MultilayerNetwork.layer_degree`）。

本模块同时定义四个类（同文件内先基类、后变体，天然无循环导入）：

* :class:`MultilayerNetwork` —— 无权无向；
* :class:`MultilayerNetworkW` —— 加权；
* :class:`MultilayerNetworkD` —— 有向；
* :class:`MultilayerNetworkWD` —— 加权有向。
"""

from __future__ import annotations

from collections.abc import Iterable

from honet._types import Members, Node
from honet.base import HigherOrderNetwork
from honet.base.mixins import DirectedMixin, WeightedMixin


class MultilayerNetwork(HigherOrderNetwork):
    """多层网络：结点跨层共享，每层各自维护一组超边。

    Args:
        layers: 初始层名集合。
        nodes: 全局结点（可只出现在某些层）。

    用法::

        ml = MultilayerNetwork(layers=["work", "family"], nodes=["a", "b", "c"])
        ml.add_edge(["a", "b"], layer="work")
        ml.add_edge(["a", "c"], layer="family")
        ml.degree("a")                    # 2（跨层累加）
        ml.layer_degree("work", "a")      # 1
    """

    def __init__(
        self,
        layers: Iterable[str] = (),
        nodes: Iterable[Node] = (),
    ) -> None:
        self._nodes: dict[Node, None] = {}
        # 层名 → [(成员, 权重), ...]，用 dict 保持层的注册顺序
        self._layers: dict[str, list[tuple[Members, float]]] = {}

        for layer in layers:
            self.add_layer(layer)
        for node in nodes:
            self._nodes[node] = None

    def add_layer(self, name: str) -> None:
        """注册一个新层（幂等）。

        Args:
            name: 层名，不能为空字符串。

        Raises:
            ValueError: 层名为空白。
        """
        if not name or not name.strip():
            raise ValueError("层名不能为空")
        self._layers.setdefault(name, [])

    @property
    def layers(self) -> tuple[str, ...]:
        """已注册层名，按注册顺序。"""
        return tuple(self._layers)

    def add_edge(self, members: Iterable[Node], layer: str, weight: float = 1.0) -> Members:
        """在指定层加入超边；层未注册会自动创建。

        Args:
            members: 超边成员。
            layer: 层名。
            weight: 仅加权变体可用，默认 1.0。

        Returns:
            规范化后的成员表示。

        Raises:
            ValueError: 成员非法或该层内已存在相同超边。
            TypeError: 无权变体传入非 1.0 权重。
        """
        ordered = self._ordered_unique(members)
        # 用原始序列做方向校验（保留重复项，便于识别自环）
        edge = self._prepare_members(tuple(members))
        prepared_weight = self._prepare_weight(weight)
        self.add_layer(layer)

        bucket = self._layers[layer]
        if any(existing == edge for existing, _ in bucket):
            raise ValueError(f"层 {layer!r} 中已存在相同超边：{edge!r}")

        for node in ordered:
            self._register(node)
        bucket.append((edge, prepared_weight))
        return edge

    def layer_edges(self, layer: str) -> tuple[Members, ...]:
        """指定层的全部超边。

        Raises:
            KeyError: 层不存在。
        """
        if layer not in self._layers:
            raise KeyError(f"层不存在：{layer!r}")
        return tuple(members for members, _ in self._layers[layer])

    def layer_degree(self, layer: str, node: Node) -> int:
        """结点在单层内的度。

        Raises:
            KeyError: 层或结点不存在。
        """
        if layer not in self._layers:
            raise KeyError(f"层不存在：{layer!r}")
        if node not in self._nodes:
            raise KeyError(f"结点不存在：{node!r}")
        return sum(1 for members, _ in self._layers[layer] if node in members)

    @property
    def nodes(self) -> tuple[Node, ...]:
        """全局结点集合。"""
        return tuple(self._nodes)

    @property
    def edges(self) -> tuple[Members, ...]:
        """所有层的超边拼接（先按层顺序，再按层内加入顺序）。"""
        return tuple(members for bucket in self._layers.values() for members, _ in bucket)

    @property
    def num_nodes(self) -> int:
        """全局结点数。"""
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        """所有层超边总数。"""
        return sum(len(bucket) for bucket in self._layers.values())

    def degree(self, node: Node) -> int:
        """结点在各层的度之和。

        Raises:
            KeyError: 结点不存在。
        """
        if node not in self._nodes:
            raise KeyError(f"结点不存在：{node!r}")
        return sum(
            1 for bucket in self._layers.values() for members, _ in bucket if node in members
        )

    def incidence_matrix(self) -> list[list[int]]:
        """结点 × （层, 超边） 的关联矩阵，列顺序与 :attr:`edges` 一致。"""
        return [[1 if node in members else 0 for members in self.edges] for node in self._nodes]


class MultilayerNetworkW(WeightedMixin, MultilayerNetwork):
    """加权多层网络：每条（层内）超边带权重。"""


class MultilayerNetworkD(DirectedMixin, MultilayerNetwork):
    """有向多层网络：各层的超边均为有序成员。"""


class MultilayerNetworkWD(WeightedMixin, DirectedMixin, MultilayerNetwork):
    """加权有向多层网络。"""


__all__ = [
    "MultilayerNetwork",
    "MultilayerNetworkD",
    "MultilayerNetworkW",
    "MultilayerNetworkWD",
]
