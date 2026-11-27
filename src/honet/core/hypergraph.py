"""honet.core.hypergraph —— 按插入顺序保存结点与超边的超图容器。

设计约定：

1. 结点必须是可哈希对象（``str``、``int``、``tuple`` 等），以字典保序存储；
2. 超边内成员无序，重复成员自动去重；完全相同的成员集合 + 相同标签视为重复超边，
   由 :meth:`Hypergraph.add_edge` 拒绝，避免隐式覆盖；
3. 所有查询方法的返回值为不可变类型（``tuple`` / ``frozenset``），
   调用方无法通过返回值篡改内部状态。

被以下文件使用：``src/honet/core/__init__.py``（re-export）、
``src/honet/utils/*``（度量函数入参）、``tests/test_core.py``。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from honet._frame import group_rows
from honet._types import Node
from honet.core.hyperedge import Hyperedge

if TYPE_CHECKING:
    import pandas as pd

# 内部用于判重的键：(超边成员集合, 标签)
EdgeKey = tuple[frozenset[Node], str | None]


class Hypergraph:
    """按插入顺序保存结点与超边的超图（高阶网络）。

    内部维护三份状态：

    * ``_nodes``：结点字典，利用 ``dict`` 的插入有序性同时实现“保序”和“O(1) 去重”；
    * ``_edges``：超边列表，保持添加顺序，供 ``edges`` 按序返回；
    * ``_edge_keys``：``(成员集合, 标签)`` 集合，用于 O(1) 判定重复超边。

    Args:
        nodes: 初始结点，可迭代对象，重复项自动去重。
        edges: 初始超边，每个元素是“结点可迭代对象”，内部逐个调用
            :meth:`add_edge`。成员若未在 ``nodes`` 中出现会被自动补建。

    用法::

        hg = Hypergraph(nodes=["a", "b"], edges=[["a", "b"], ["a", "b", "c"]])
        hg.num_nodes, hg.num_edges    # (3, 2)
        hg.degree("a")                # 2
        hg.neighbors("c")             # frozenset({'a', 'b'})

    Raises:
        TypeError: 结点不可哈希。
        ValueError: 初始超边为空，或出现重复超边。
    """

    __slots__ = ("_edge_keys", "_edges", "_nodes")

    def __init__(
        self,
        nodes: Iterable[Node] = (),
        edges: Iterable[Iterable[Node]] = (),
    ) -> None:
        # 结点容器：dict 的键集合即结点集合，value 恒为 None（仅占位）
        self._nodes: dict[Node, None] = {}
        # 超边容器：list 保证顺序，保持“插入顺序即编号顺序”的语义
        self._edges: list[Hyperedge] = []
        # 判重集合：元素为 (frozenset(成员), 标签)
        self._edge_keys: set[EdgeKey] = set()

        # ---- 第一阶段：导入结点 ----
        for node in nodes:
            self.add_node(node)

        # ---- 第二阶段：导入超边（成员结点按需补建）----
        for members in edges:
            self.add_edge(members)

    # ------------------------------------------------------------------
    # 从表格数据构造
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        frame: pd.DataFrame,
        *,
        edge_col: str = "edge",
        node_col: str = "node",
        weight_col: str | None = None,
        label_col: str | None = None,
    ) -> Hypergraph:
        """由 pandas DataFrame 构造超图（长表：一行表示“某结点属于某条超边”）。

        Args:
            frame: 输入表，列名通过下面的参数指定。
            edge_col: 超边标识列，取值相同的行归为同一条超边。
            node_col: 结点列。
            weight_col: 可选权重列；同一超边的各行权重必须一致，缺省为 1.0。
            label_col: 可选标签列；非字符串值用 ``str()`` 转换，缺失视为无标签。

        Returns:
            新的 :class:`Hypergraph`：超边与结点均按首次出现顺序排列，
            同一超边内重复出现的结点自动去重。

        Raises:
            KeyError: DataFrame 缺少指定列。
            ValueError: 标识或结点为空、权重不是实数、同一超边的权重/标签不一致，
                或两条不同超边拥有完全相同的成员与标签。

        用法::

            df = pd.DataFrame({
                "paper": ["p1", "p1", "p2"],
                "author": ["a", "b", "a"],
            })
            hg = Hypergraph.from_dataframe(df, edge_col="paper", node_col="author")
            hg.num_edges, hg.degree("a")    # (2, 2)
        """
        graph = cls()
        groups = group_rows(
            frame,
            key_cols=(edge_col,),
            node_col=node_col,
            weight_col=weight_col,
            label_col=label_col,
        )
        for _, members, weight, label in groups:
            graph.add_edge(members, label=label, weight=weight)
        return graph

    # ------------------------------------------------------------------
    # 构造与修改
    # ------------------------------------------------------------------

    def add_node(self, node: Node) -> None:
        """加入单个结点；若已存在则什么也不做（幂等）。

        Args:
            node: 可哈希的结点对象。

        Raises:
            TypeError: ``node`` 不可哈希（由 ``dict`` 插入抛出）。
        """
        self._nodes[node] = None

    def add_edge(
        self,
        members: Iterable[Node],
        label: str | None = None,
        weight: float = 1.0,
    ) -> Hyperedge:
        """加入一条超边，成员结点若不存在会被自动补建。

        Args:
            members: 超边成员的可迭代对象，重复成员会被去重。
            label: 可选标签，语义见 :class:`Hyperedge`。
            weight: 超边权重（正的有限实数），语义见 :class:`Hyperedge`。

        Returns:
            实际存入超图的 :class:`Hyperedge` 实例。

        Raises:
            ValueError: ``members`` 为空、``weight`` 非正，或与已有超边的成员集合和标签完全相同。
        """
        # 先物化为 tuple：保留调用方给出的成员顺序，用于后续按序补建结点，
        # 保证 ``hg.nodes`` 的顺序与输入顺序一致（frozenset 的迭代顺序随机）
        member_order = tuple(dict.fromkeys(members))
        # 再去重、转成不可变集合，作为超边的正式成员表示
        member_set = frozenset(member_order)
        edge = Hyperedge(member_set, label, weight)

        # 判重键 = 成员集合 + 标签；成员相同但标签不同视为不同超边
        key: EdgeKey = (edge.members, edge.label)
        if key in self._edge_keys:
            raise ValueError(f"重复的超边：{sorted(map(str, edge.members))}（label={label!r}）")

        # ---- 补建成员结点（按调用方给出的顺序，保证 nodes 顺序可预期）----
        for node in member_order:
            self._nodes[node] = None

        # ---- 落库 ----
        self._edges.append(edge)
        self._edge_keys.add(key)
        return edge

    def remove_node(self, node: Node) -> None:
        """删除结点，并把该结点从所有超边中移除。

        若某条超边因此变空，则该超边一并删除（空超边无意义）。
        其余结点为空但仍有成员的超边保持不变。

        Args:
            node: 待删除结点。

        Raises:
            KeyError: 结点不在超图中。
        """
        if node not in self._nodes:
            raise KeyError(f"结点不存在：{node!r}")
        del self._nodes[node]

        # ---- 重建超边列表：剔除目标结点，丢弃变空的超边 ----
        kept: list[Hyperedge] = []
        kept_keys: set[EdgeKey] = set()
        for edge in self._edges:
            if node not in edge.members:
                kept.append(edge)
                kept_keys.add((edge.members, edge.label))
                continue
            shrunk = edge.members - {node}
            if not shrunk:
                continue  # 超边被清空，直接丢弃
            new_edge = Hyperedge(shrunk, edge.label)
            new_key: EdgeKey = (new_edge.members, new_edge.label)
            # 收缩后可能与已有超边重合，此时保留先出现的，避免破坏唯一性约束
            if new_key in kept_keys:
                continue
            kept.append(new_edge)
            kept_keys.add(new_key)

        self._edges = kept
        self._edge_keys = kept_keys

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> tuple[Node, ...]:
        """全部结点，按首次插入顺序返回的只读元组。"""
        return tuple(self._nodes)

    @property
    def node_set(self) -> frozenset[Node]:
        """全部结点构成的集合，适合做集合运算（交、并、差）。"""
        return frozenset(self._nodes)

    @property
    def edges(self) -> tuple[Hyperedge, ...]:
        """全部超边，按添加顺序返回的只读元组。"""
        return tuple(self._edges)

    @property
    def num_nodes(self) -> int:
        """结点数 ``N``。"""
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        """超边数 ``M``。"""
        return len(self._edges)

    @property
    def edge_sizes(self) -> tuple[int, ...]:
        """各超边大小的元组，顺序与 :attr:`edges` 一致。"""
        return tuple(len(edge) for edge in self._edges)

    @property
    def is_uniform(self) -> bool:
        """是否为均匀超图：所有超边大小相同（无超边时视为均匀）。"""
        sizes = {len(edge) for edge in self._edges}
        return len(sizes) <= 1

    def degree(self, node: Node) -> int:
        """结点的高阶度：包含该结点的超边条数 ``k_i = |{e : i ∈ e}|``。

        Args:
            node: 待查询结点。

        Returns:
            超边条数；孤立结点返回 0。

        Raises:
            KeyError: 结点不在超图中。
        """
        if node not in self._nodes:
            raise KeyError(f"结点不存在：{node!r}")
        return sum(1 for edge in self._edges if node in edge.members)

    def neighbors(self, node: Node) -> frozenset[Node]:
        """结点在“共边”意义下的邻居集合。

        只要两个结点同时属于至少一条超边，就互为邻居，即::

            N(i) = (∪_{e ∋ i} e) \\ {i}

        Args:
            node: 待查询结点。

        Returns:
            邻居结点集合（不含自身）；孤立结点返回空集合。

        Raises:
            KeyError: 结点不在超图中。
        """
        if node not in self._nodes:
            raise KeyError(f"结点不存在：{node!r}")
        result: set[Node] = set()
        for edge in self._edges:
            if node in edge.members:
                result |= edge.members
        result.discard(node)
        return frozenset(result)

    def subhypergraph(self, nodes: Iterable[Node]) -> Hypergraph:
        """导出由给定结点诱导的子超图。

        规则：只保留成员全部落在 ``nodes`` 内的超边；``nodes`` 中不存在于原图的结点
        会被忽略（不报错，便于按名字过滤场景）。

        Args:
            nodes: 目标结点集合。

        Returns:
            新的 :class:`Hypergraph` 实例，原图不受影响。

        Raises:
            无。
        """
        selected = frozenset(n for n in nodes if n in self._nodes)
        kept = [edge for edge in self._edges if edge.members <= selected]
        return Hypergraph(
            nodes=[n for n in self._nodes if n in selected],
            edges=[list(edge.members) for edge in kept],
        )

    # ------------------------------------------------------------------
    # 容器协议
    # ------------------------------------------------------------------

    def __contains__(self, node: object) -> bool:
        """支持 ``node in hg``，判断结点是否存在。"""
        return node in self._nodes

    def __len__(self) -> int:
        """``len(hg)`` 返回结点数，与 :attr:`num_nodes` 等价。"""
        return len(self._nodes)

    def __eq__(self, other: object) -> bool:
        """结构相等：结点集合与超边集合（含标签、权重）均相同即相等，与插入顺序无关。"""
        if not isinstance(other, Hypergraph):
            return NotImplemented
        # 逐条比较 Hyperedge（frozen dataclass 可哈希），因此权重差异也会被识别
        return self.node_set == other.node_set and set(self._edges) == set(other._edges)

    def __hash__(self) -> int:
        """可变对象不提供哈希，显式声明以给出明确报错信息。"""
        raise TypeError("Hypergraph 是可变对象，不可哈希")

    def __repr__(self) -> str:
        """调试友好表示，例如 ``<Hypergraph nodes=5 edges=3 uniform=False>``。"""
        return (
            f"<Hypergraph nodes={self.num_nodes} edges={self.num_edges} uniform={self.is_uniform}>"
        )


__all__ = ["EdgeKey", "Hypergraph"]
