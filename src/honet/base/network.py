"""honet.base.network —— 所有高阶网络形态的抽象基类。

本模块只定义 :class:`HigherOrderNetwork`：公共接口、容器协议，
以及供子类复用的成员/权重规范化钩子。加权 / 有向能力由 ``honet.base.mixins``
中的两个 mixin 提供，四种形态分别位于 ``honet.simplicial`` / ``honet.hypergraph`` /
``honet.multilayer`` / ``honet.temporal``。

被以下文件使用：``src/honet/base/__init__.py`` （re-export）、各形态的
``network.py``（继承）、``tests/test_networks.py``。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from honet._types import Members, Node


class HigherOrderNetwork(ABC):
    """所有高阶网络形态的抽象基类：定义公共接口，不能直接实例化。

    子类必须实现六个抽象成员：:attr:`nodes`、:attr:`edges`、:attr:`num_nodes`、
    :attr:`num_edges`、:meth:`degree`、:meth:`incidence_matrix`。

    类属性 :attr:`is_weighted` / :attr:`is_directed` 表示能力开关：
    默认无权无向，由 ``WeightedMixin`` / ``DirectedMixin`` 覆盖为 ``True``。

    用法（子类）::

        hg = HypergraphNetwork(edges=[["a", "b"]])
        hg.degree("a")            # 1
        hg.incidence_matrix()     # [[1], [1]]

    Raises:
        TypeError: 直接实例化 ``HigherOrderNetwork()`` 时抛出（存在抽象方法）。
    """

    # ---- 能力标记：由 mixin 覆盖 ----
    is_weighted: bool = False
    is_directed: bool = False

    # ------------------------------------------------------------------
    # 抽象接口（四个形态基类全部实现）
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def nodes(self) -> tuple[Node, ...]:
        """全部结点，按首次出现顺序返回。"""

    @property
    @abstractmethod
    def edges(self) -> tuple[Members, ...]:
        """全部交互（单形面 / 超边 / 各层超边 / 时序事件），按存储顺序返回。"""

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        """结点数。"""

    @property
    @abstractmethod
    def num_edges(self) -> int:
        """交互条数。"""

    @abstractmethod
    def degree(self, node: Node) -> int:
        """结点度数：包含该结点的交互条数。"""

    @abstractmethod
    def incidence_matrix(self) -> list[list[int]]:
        """关联矩阵：行对应 :attr:`nodes`，列对应 :attr:`edges`。"""

    # ------------------------------------------------------------------
    # 公共容器协议
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """``len(net)`` 等于结点数。"""
        return self.num_nodes

    def __contains__(self, node: object) -> bool:
        """``node in net`` 判断结点是否存在。"""
        return node in self.nodes

    def __repr__(self) -> str:
        """调试表示，例如 ``<HypergraphNetworkW nodes=3 edges=2 weighted=True directed=False>``。"""
        return (
            f"<{type(self).__name__} nodes={self.num_nodes} edges={self.num_edges} "
            f"weighted={self.is_weighted} directed={self.is_directed}>"
        )

    # ------------------------------------------------------------------
    # 供子类复用的成员/权重规范化钩子（MRO 决定实际行为）
    # ------------------------------------------------------------------

    def _prepare_members(self, members: Iterable[Node]) -> Members:
        """无向场景：把成员去重为 ``frozenset``，并拒绝空交互。

        有向变体由 ``DirectedMixin`` 覆盖本方法（保留顺序、校验重复）。

        Args:
            members: 成员可迭代对象（顺序不重要）。

        Returns:
            规范化后的成员集合。

        Raises:
            ValueError: 成员为空。
        """
        member_set = frozenset(members)
        if not member_set:
            raise ValueError("一条交互至少需要包含一个结点")
        return member_set

    def _prepare_weight(self, weight: float) -> float:
        """无权场景：只接受 1.0；加权变体由 ``WeightedMixin`` 覆盖。

        Raises:
            TypeError: 传入非 1.0 的权重（提示改用 ``...W`` / ``...WD`` 变体）。
        """
        if weight != 1.0:
            raise TypeError(
                f"{type(self).__name__} 是无权网络，不接受自定义权重 {weight!r}；"
                "请使用 ...W 或 ...WD 变体"
            )
        return 1.0

    @staticmethod
    def _ordered_unique(members: Iterable[Node]) -> tuple[Node, ...]:
        """按调用方顺序去重，用于“先按序建结点、再规范化成员”。"""
        return tuple(dict.fromkeys(members))

    def _register(self, node: Node, table: dict[Node, None] | None = None) -> None:
        """把结点登记到 ``table``（缺省为本实例的结点表 ``_nodes``）。"""
        (table if table is not None else self._nodes)[node] = None  # type: ignore[attr-defined]


__all__ = ["HigherOrderNetwork"]
