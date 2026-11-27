"""honet.simplicial.network —— 单纯复形网络及其加权 / 有向变体。

单纯复形以**极大单形（facet）**为基本存储单位，并向下封闭：
加入 ``{a, b, c}`` 会自动拥有其全部非空子面。

本模块同时定义四个类（同文件内先基类、后变体，天然无循环导入）：

* :class:`SimplicialComplexNetwork` —— 无权无向；
* :class:`SimplicialComplexNetworkW` —— 加权；
* :class:`SimplicialComplexNetworkD` —— 有向；
* :class:`SimplicialComplexNetworkWD` —— 加权有向。
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


class SimplicialComplexNetwork(HigherOrderNetwork):
    """单纯复形网络：以**极大单形（facet）**为基本存储单位。

    与超图的关键差别是**向下封闭**：加入 ``{a, b, c}`` 这一 2-单形，
    其边界 ``{a,b}``、``{a,c}``、``{b,c}`` 与顶点 ``{a}``、``{b}``、``{c}`` 都自动属于复形。
    :attr:`facets` 只返回用户显式加入的极大面，:meth:`faces` 展开全部子面。

    Args:
        nodes: 孤立结点（不属于任何单形）。
        facets: 初始极大单形，每个元素是结点可迭代对象。

    用法::

        sc = SimplicialComplexNetwork(facets=[["a", "b", "c"]])
        sc.num_facets      # 1
        sc.num_faces       # 7 = C(3,1)+C(3,2)+C(3,3)
        sc.dimension       # 2（最大单形大小 - 1）
    """

    def __init__(
        self,
        nodes: Iterable[Node] = (),
        facets: Iterable[Iterable[Node]] = (),
    ) -> None:
        # 结点表：保序去重
        self._nodes: dict[Node, None] = {}
        # 极大单形列表：元素为 (成员, 权重)，保持加入顺序
        self._facets: list[tuple[Members, float]] = []

        for node in nodes:
            self._nodes[node] = None
        for facet in facets:
            self.add_facet(facet)

    @classmethod
    def from_dataframe(
        cls,
        frame: pd.DataFrame,
        *,
        edge_col: str = "edge",
        node_col: str = "node",
        weight_col: str | None = None,
    ) -> SimplicialComplexNetwork:
        """由 pandas DataFrame 构造单纯复形（长表：一行 = 结点属于某个极大单形）。

        ``edge_col`` 取值相同的行合并为一个极大单形；其全部子面仍由
        :meth:`faces` 按向下封闭规则展开，无需在表中重复罗列。

        Args:
            frame: 输入表。
            edge_col: 极大单形的标识列。
            node_col: 结点列。
            weight_col: 可选权重列，同一单形内必须一致，缺省为 1.0。

        Returns:
            新的复形，极大单形按首次出现顺序排列。

        Raises:
            KeyError: 缺少指定列。
            ValueError: 标识或结点为空、同组权重冲突、单形重复。
        """
        graph = cls()
        groups = group_rows(
            frame,
            key_cols=(edge_col,),
            node_col=node_col,
            weight_col=weight_col,
        )
        for _, members, weight, _ in groups:
            graph.add_facet(members, weight=weight)
        return graph

    # ---- 构造 ----

    def add_facet(self, members: Iterable[Node], weight: float = 1.0) -> Members:
        """加入一个极大单形（面）。

        Args:
            members: 单形顶点，规模 k 即 (k-1)-单形。
            weight: 仅加权变体可用，默认 1.0。

        Returns:
            规范化后的单形成员表示。

        Raises:
            ValueError: 成员为空（无向）/成员不足两个或重复（有向）/重复单形。
            TypeError: 无权变体传入了非 1.0 权重。
        """
        ordered = self._ordered_unique(members)
        # 注意：规范化用的是**原始序列**（含可能的重复项），
        # 这样有向变体才能识别 a→a 这类自环；ordered 仅用于建结点
        simplex = self._prepare_members(tuple(members))
        prepared_weight = self._prepare_weight(weight)

        if any(existing == simplex for existing, _ in self._facets):
            raise ValueError(f"重复的单形：{simplex!r}")

        for node in ordered:
            self._register(node)
        self._facets.append((simplex, prepared_weight))
        return simplex

    # ---- 形态特有属性 ----

    @property
    def facets(self) -> tuple[Members, ...]:
        """极大单形（用户显式加入的面），不含自动展开的子面。"""
        return tuple(members for members, _ in self._facets)

    @property
    def num_facets(self) -> int:
        """极大单形个数。"""
        return len(self._facets)

    @property
    def dimension(self) -> int:
        """复形维数 = 最大单形大小 - 1（空复形为 -1）。"""
        return max((len(m) for m, _ in self._facets), default=0) - 1

    @property
    def weights(self) -> tuple[float, ...]:
        """各极大单形的权重（无权变体恒为 1.0）。"""
        return tuple(weight for _, weight in self._facets)

    def faces(self) -> tuple[frozenset[Node], ...]:
        """展开全部非空子面（去重、按大小升序）。

        规模 k 的单形贡献 ``2**k - 1`` 个非空子面。

        Returns:
            子面集合的元组，元素为 ``frozenset``。
        """
        all_faces: set[frozenset[Node]] = set()
        for members, _ in self._facets:
            vertices = sorted(members, key=str)
            count = len(vertices)
            # 用位掩码枚举全部非空子集，规模小（k ≤ 10 量级）足够
            for mask in range(1, 1 << count):
                all_faces.add(frozenset(vertices[i] for i in range(count) if mask >> i & 1))
        return tuple(sorted(all_faces, key=lambda f: (len(f), sorted(map(str, f)))))

    @property
    def num_faces(self) -> int:
        """全部非空子面的个数。"""
        return len(self.faces())

    # ---- 抽象接口实现 ----

    @property
    def nodes(self) -> tuple[Node, ...]:
        """全部结点（含只出现在子面中的结点）。"""
        return tuple(self._nodes)

    @property
    def edges(self) -> tuple[Members, ...]:
        """等同于 :attr:`facets`：单纯复形以极大面作为“边”。"""
        return self.facets

    @property
    def num_nodes(self) -> int:
        """结点数。"""
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        """极大单形数。"""
        return len(self._facets)

    def degree(self, node: Node) -> int:
        """结点参与的极大单形个数。

        Raises:
            KeyError: 结点不存在。
        """
        if node not in self._nodes:
            raise KeyError(f"结点不存在：{node!r}")
        return sum(1 for members, _ in self._facets if node in members)

    def incidence_matrix(self) -> list[list[int]]:
        """结点 × 极大单形 的关联矩阵。"""
        return [
            [1 if node in members else 0 for members, _ in self._facets] for node in self._nodes
        ]


class SimplicialComplexNetworkW(WeightedMixin, SimplicialComplexNetwork):
    """加权单纯复形：每个极大面带权重 ``weight > 0``（默认 1.0）。"""


class SimplicialComplexNetworkD(DirectedMixin, SimplicialComplexNetwork):
    """有向单纯复形：单形按顶点顺序表示（有向 k-单形），自环与重复顶点被拒绝。"""


class SimplicialComplexNetworkWD(WeightedMixin, DirectedMixin, SimplicialComplexNetwork):
    """加权有向单纯复形：既保留顶点顺序，又携带权重。"""


__all__ = [
    "SimplicialComplexNetwork",
    "SimplicialComplexNetworkD",
    "SimplicialComplexNetworkW",
    "SimplicialComplexNetworkWD",
]
