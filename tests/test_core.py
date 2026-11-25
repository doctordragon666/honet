"""honet 核心数据结构的单元测试。

覆盖点：构造、自动补建结点、重复/非法超边拒绝、度与邻居查询、
结点删除时的超边收缩、子超图导出、相等性与容器协议。
运行方式：``uv run pytest``。
"""

from __future__ import annotations

import dataclasses

import pytest

from honet import Hyperedge, Hypergraph
from honet.core import Node


class TestHyperedge:
    """Hyperedge 的构造校验与基本属性。"""

    def test_size_and_len(self) -> None:
        """大小属性与 len() 一致。"""
        edge = Hyperedge(frozenset({"a", "b"}))
        assert edge.size == 2
        assert len(edge) == 2

    def test_label_optional(self) -> None:
        """标签缺省为 None，显式传入时原样保留。"""
        assert Hyperedge(frozenset({"a"})).label is None
        assert Hyperedge(frozenset({"a"}), label="t1").label == "t1"

    def test_empty_members_rejected(self) -> None:
        """空超边没有物理意义，必须报 ValueError。"""
        with pytest.raises(ValueError, match="至少需要包含一个结点"):
            Hyperedge(frozenset())

    def test_blank_label_rejected(self) -> None:
        """空白标签属于脏数据，必须报 ValueError。"""
        with pytest.raises(ValueError, match="标签"):
            Hyperedge(frozenset({"a"}), label="   ")

    def test_contains(self) -> None:
        """``node in edge`` 语义为成员判断。"""
        edge = Hyperedge(frozenset({"a", "b"}))
        assert "a" in edge
        assert "z" not in edge

    def test_frozen(self) -> None:
        """frozen dataclass 不允许再赋值，保证内部状态稳定。"""
        edge = Hyperedge(frozenset({"a"}))
        with pytest.raises(dataclasses.FrozenInstanceError):
            edge.members = frozenset({"b"})  # type: ignore[misc]

    def test_weight_default(self) -> None:
        """权重缺省为 1.0，表示“未加权的普通超边”。"""
        assert Hyperedge(frozenset({"a"})).weight == 1.0

    def test_weight_custom(self) -> None:
        """可显式指定正整数/浮点权重，原样保留。"""
        assert Hyperedge(frozenset({"a"}), weight=3).weight == 3
        assert Hyperedge(frozenset({"a"}), weight=0.5).weight == 0.5

    @pytest.mark.parametrize("bad", [0, -1, -0.5, float("nan"), "2"])
    def test_invalid_weight_rejected(self, bad: object) -> None:
        """非正数、NaN、非数值的权重都要报 ValueError。"""
        with pytest.raises(ValueError, match="权重"):
            Hyperedge(frozenset({"a"}), weight=bad)  # type: ignore[arg-type]


class TestHypergraphConstruction:
    """构造阶段的结点/超边导入行为。"""

    def test_initial_nodes_and_edges(self) -> None:
        """构造参数中的结点与超边被完整导入，并保持插入顺序。"""
        hg = Hypergraph(nodes=["a", "b"], edges=[["a", "b"], ["a", "b", "c"]])
        assert hg.nodes == ("a", "b", "c")  # c 由第二条超边自动补建
        assert hg.num_nodes == 3
        assert hg.num_edges == 2

    def test_duplicate_nodes_collapsed(self) -> None:
        """重复结点只保留一份（幂等）。"""
        hg = Hypergraph(nodes=["a", "a", "b"])
        assert hg.nodes == ("a", "b")

    def test_duplicate_edge_rejected(self) -> None:
        """成员集合与标签都相同的超边视为重复，必须报错。"""
        hg = Hypergraph(edges=[["a", "b"]])
        with pytest.raises(ValueError, match="重复的超边"):
            hg.add_edge(["b", "a"])  # 顺序不同但成员集合相同

    def test_same_members_different_label_allowed(self) -> None:
        """成员相同但标签不同的超边是合法的两条边。"""
        hg = Hypergraph()
        hg.add_edge(["a", "b"], label="2023")
        hg.add_edge(["a", "b"], label="2024")
        assert hg.num_edges == 2

    def test_empty_edge_rejected(self) -> None:
        """通过 add_edge 传空集合也要报错。"""
        hg = Hypergraph()
        with pytest.raises(ValueError, match="至少需要包含一个结点"):
            hg.add_edge([])

    def test_unhashable_node_rejected(self) -> None:
        """结点必须可哈希，列表结点应触发 TypeError。"""
        hg = Hypergraph()
        with pytest.raises(TypeError):
            hg.add_node(["not", "hashable"])  # type: ignore[arg-type]

    def test_edge_weight_roundtrip(self) -> None:
        """add_edge 的 weight 参数写入超边并可回读。"""
        hg = Hypergraph()
        edge = hg.add_edge(["a", "b"], weight=2.5)
        assert edge.weight == 2.5
        assert hg.edges[0].weight == 2.5


class TestHypergraphQueries:
    """度、邻居、均匀性等只读查询。"""

    @pytest.fixture()
    def hg(self) -> Hypergraph:
        """构造一个固定拓扑：{a,b,c} 与 {b,c} 两条超边。"""
        return Hypergraph(nodes=["a", "b", "c", "d"], edges=[["a", "b", "c"], ["b", "c"]])

    def test_degree(self, hg: Hypergraph) -> None:
        """高阶度 = 参与的超边条数；孤立结点为 0。"""
        assert hg.degree("b") == 2
        assert hg.degree("a") == 1
        assert hg.degree("d") == 0

    def test_degree_missing_node(self, hg: Hypergraph) -> None:
        """查询不存在的结点抛 KeyError。"""
        with pytest.raises(KeyError):
            hg.degree("zzz")

    def test_neighbors(self, hg: Hypergraph) -> None:
        """邻居为共边结点集合，且不含自身。"""
        assert hg.neighbors("a") == frozenset({"b", "c"})
        assert hg.neighbors("b") == frozenset({"a", "c"})
        assert hg.neighbors("d") == frozenset()

    def test_neighbors_missing_node(self, hg: Hypergraph) -> None:
        """查询不存在结点的邻居抛 KeyError。"""
        with pytest.raises(KeyError):
            hg.neighbors("zzz")

    def test_edge_sizes_and_uniform(self, hg: Hypergraph) -> None:
        """边大小按插入顺序返回；大小不一致时非均匀。"""
        assert hg.edge_sizes == (3, 2)
        assert hg.is_uniform is False

    def test_uniform_true(self) -> None:
        """全部边大小一致 → 均匀超图；空图视为均匀。"""
        hg = Hypergraph(edges=[["a", "b"], ["b", "c"]])
        assert hg.is_uniform is True
        assert Hypergraph().is_uniform is True

    def test_node_set_and_contains(self, hg: Hypergraph) -> None:
        """node_set 可用于集合运算，in 判断结点存在性。"""
        assert hg.node_set == frozenset({"a", "b", "c", "d"})
        assert "a" in hg
        assert "zzz" not in hg

    def test_len(self, hg: Hypergraph) -> None:
        """len(hg) 表示结点数。"""
        assert len(hg) == 4

    def test_repr(self, hg: Hypergraph) -> None:
        """repr 暴露规模信息，便于调试。"""
        assert "nodes=4" in repr(hg)
        assert "edges=2" in repr(hg)


class TestHypergraphMutation:
    """删除结点与子超图导出。"""

    def test_remove_node_shrinks_edges(self) -> None:
        """删除结点后，含该结点的超边收缩，且结点从结点表消失。"""
        hg = Hypergraph(edges=[["a", "b", "c"]])
        hg.remove_node("a")
        assert hg.nodes == ("b", "c")
        assert hg.edges[0].members == frozenset({"b", "c"})

    def test_remove_node_drops_empty_edge(self) -> None:
        """超边被删空后应整体移除。"""
        hg = Hypergraph(edges=[["a", "b"], ["c"]])
        hg.remove_node("c")
        assert hg.num_edges == 1

    def test_remove_missing_node(self) -> None:
        """删除不存在的结点抛 KeyError。"""
        hg = Hypergraph(nodes=["a"])
        with pytest.raises(KeyError):
            hg.remove_node("zzz")

    def test_subhypergraph(self) -> None:
        """子图只保留成员全部落入选择集的超边。"""
        hg = Hypergraph(edges=[["a", "b"], ["a", "b", "c"]])
        sub = hg.subhypergraph(["a", "b"])
        assert sub.nodes == ("a", "b")
        assert sub.num_edges == 1
        assert sub.edges[0].members == frozenset({"a", "b"})
        # 原图不受影响
        assert hg.num_edges == 2

    def test_subhypergraph_ignores_unknown(self) -> None:
        """选择集中不存在的结点被忽略而非报错。"""
        hg = Hypergraph(edges=[["a", "b"]])
        sub = hg.subhypergraph(["a", "b", "ghost"])
        assert sub.nodes == ("a", "b")


class TestEqualityAndHash:
    """相等性与哈希行为。"""

    def test_equality_order_independent(self) -> None:
        """结点的插入顺序不影响结构相等性。"""
        a = Hypergraph(nodes=["x", "y"], edges=[["x", "y"]])
        b = Hypergraph(nodes=["y", "x"], edges=[["y", "x"]])
        assert a == b

    def test_inequality_on_edges(self) -> None:
        """超边不同则不等。"""
        a = Hypergraph(edges=[["x", "y"]])
        b = Hypergraph(edges=[["x", "y", "z"]])
        assert a != b

    def test_compare_with_other_type(self) -> None:
        """与非 Hypergraph 对象比较返回 False 而非异常。"""
        assert Hypergraph() != object()

    def test_inequality_on_weight(self) -> None:
        """成员与标签相同、仅权重不同，也算不相等。"""
        weighted = Hypergraph()
        weighted.add_edge(["x", "y"], weight=2.0)
        assert Hypergraph(edges=[["x", "y"]]) != weighted

    def test_hash_raises(self) -> None:
        """可变容器不可哈希，显式抛 TypeError。"""
        with pytest.raises(TypeError, match="不可哈希"):
            hash(Hypergraph())

    def test_node_type_alias(self) -> None:
        """Node 类型别名可被下游代码导入使用。"""
        node: Node = "a"
        assert isinstance(node, str)
