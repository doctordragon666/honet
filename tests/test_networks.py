"""honet 类层次（``base`` + 四个形态子包）的单元测试。

覆盖点：
* 抽象基类 ``HigherOrderNetwork`` 不可实例化，四个形态基类可实例化；
* 16 个具体类齐全、继承关系与 MRO 正确（``WD`` 两个 mixin 叠加）；
* ``is_weighted`` / ``is_directed`` 组合正确，加权默认 1.0、自定义权重生效、无权类拒收权重；
* 有向类的方向校验（成员顺序保留、自环与重复成员被拒绝）；
* 四种形态各自的数据结构行为（单纯复形展开子面、多层按层统计、时序按时间排序等）。

运行方式：``uv run pytest``。
"""

from __future__ import annotations

import pytest

import honet

# 四种形态的类名前缀
SHAPES = [
    "SimplicialComplexNetwork",
    "HypergraphNetwork",
    "MultilayerNetwork",
    "TemporalNetwork",
]

# 每个形态派生的四个变体（无后缀 / W / D / WD）
VARIANTS = ["", "W", "D", "WD"]

# 16 个具体类的类名
ALL_CONCRETE = [f"{shape}{suffix}" for shape in SHAPES for suffix in VARIANTS]


class TestHierarchy:
    """继承关系、MRO 与可实例化性。"""

    def test_abstract_base_not_instantiable(self) -> None:
        """抽象基类必须抛 TypeError。"""
        with pytest.raises(TypeError):
            honet.HigherOrderNetwork()  # type: ignore[abstract]

    @pytest.mark.parametrize("name", SHAPES)
    def test_shape_bases_instantiable(self, name: str) -> None:
        """四个形态基类实现了全部抽象方法，可以直接实例化。"""
        cls = getattr(honet, name)
        instance = cls()
        assert isinstance(instance, honet.HigherOrderNetwork)
        assert instance.num_nodes == 0
        assert instance.num_edges == 0
        assert instance.is_weighted is False
        assert instance.is_directed is False

    @pytest.mark.parametrize("name", ALL_CONCRETE)
    def test_all_sixteen_exist_and_instantiate(self, name: str) -> None:
        """16 个具体类都存在、可实例化且是抽象基类的子类。"""
        cls = getattr(honet, name, None)
        assert cls is not None, f"缺少类 {name}"
        assert issubclass(cls, honet.HigherOrderNetwork)
        instance = cls()
        assert type(instance).__name__ == name

    @pytest.mark.parametrize("name", ALL_CONCRETE)
    def test_exported_in_all(self, name: str) -> None:
        """全部具体类都出现在 ``honet.__all__`` 中。"""
        assert name in honet.__all__

    @pytest.mark.parametrize("shape", SHAPES)
    def test_flag_combinations(self, shape: str) -> None:
        """四种变体的 is_weighted / is_directed 组合正确。"""
        expectations = {
            "": (False, False),
            "W": (True, False),
            "D": (False, True),
            "WD": (True, True),
        }
        for suffix, (weighted, directed) in expectations.items():
            instance = getattr(honet, f"{shape}{suffix}")()
            assert instance.is_weighted is weighted, f"{shape}{suffix} is_weighted 错误"
            assert instance.is_directed is directed, f"{shape}{suffix} is_directed 错误"

    def test_mro_weighted_directed(self) -> None:
        """WD 变体的 MRO 必须同时包含两个 mixin，且 mixin 在形态基类之前。"""
        mro = honet.HypergraphNetworkWD.__mro__
        assert honet.WeightedMixin in mro
        assert honet.DirectedMixin in mro
        assert honet.HypergraphNetwork in mro
        # mixin 必须排在形态基类之前，其行为才能覆盖默认实现
        assert mro.index(honet.WeightedMixin) < mro.index(honet.HypergraphNetwork)
        assert mro.index(honet.DirectedMixin) < mro.index(honet.HypergraphNetwork)

    def test_mixin_issubclass_relations(self) -> None:
        """W/D/WD 与对应 mixin 的 issubclass 关系正确，且互不越界。"""
        assert issubclass(honet.HypergraphNetworkW, honet.WeightedMixin)
        assert not issubclass(honet.HypergraphNetworkW, honet.DirectedMixin)
        assert issubclass(honet.HypergraphNetworkD, honet.DirectedMixin)
        assert not issubclass(honet.HypergraphNetworkD, honet.WeightedMixin)
        assert issubclass(honet.HypergraphNetworkWD, honet.WeightedMixin)
        assert issubclass(honet.HypergraphNetworkWD, honet.DirectedMixin)

    def test_repr_mentions_capabilities(self) -> None:
        """repr 反映类名与能力标记，便于调试。"""
        text = repr(honet.HypergraphNetworkWD())
        assert "HypergraphNetworkWD" in text
        assert "weighted=True" in text
        assert "directed=True" in text


class TestWeighting:
    """加权变体的默认值、自定义权重与无权类的拒绝行为。"""

    def test_weighted_default_is_one(self) -> None:
        """加权变体不传 weight 时默认为 1.0。"""
        net = honet.HypergraphNetworkW(edges=[["a", "b"]])
        assert net.weights == (1.0,)

    def test_weighted_custom_value(self) -> None:
        """加权变体接受正实数权重并原样保存。"""
        net = honet.HypergraphNetworkW()
        net.add_edge(["a", "b"], weight=2.5)
        net.add_edge(["b", "c"], weight=3)  # int 应被归一化为 float
        assert net.weights == (2.5, 3.0)
        assert all(isinstance(w, float) for w in net.weights)

    @pytest.mark.parametrize("bad", [0, -1, float("nan")])
    def test_weighted_rejects_invalid(self, bad: float) -> None:
        """加权变体拒绝 0、负数与 NaN。"""
        net = honet.HypergraphNetworkW()
        with pytest.raises(ValueError, match="权重"):
            net.add_edge(["a", "b"], weight=bad)

    def test_unweighted_rejects_custom_weight(self) -> None:
        """无权变体传入非 1.0 权重时抛 TypeError，提示改用 W 变体。"""
        net = honet.HypergraphNetwork()
        with pytest.raises(TypeError, match="无权网络"):
            net.add_edge(["a", "b"], weight=2.0)

    def test_unweighted_accepts_explicit_one(self) -> None:
        """无权变体显式传 1.0 是允许的（等价于默认值）。"""
        net = honet.HypergraphNetwork()
        net.add_edge(["a", "b"], weight=1.0)
        assert net.weights == (1.0,)


class TestDirection:
    """有向变体的顺序保存与方向校验。"""

    def test_direction_order_preserved(self) -> None:
        """有向超边的成员顺序即方向，按传入顺序保存。"""
        net = honet.HypergraphNetworkD()
        net.add_edge(["c", "a", "b"])
        assert net.edges == (("c", "a", "b"),)

    def test_order_sensitive_duplicate(self) -> None:
        """顺序不同的同组成员视为不同有向边。"""
        net = honet.HypergraphNetworkD()
        net.add_edge(["a", "b"])
        net.add_edge(["b", "a"])
        assert net.num_edges == 2

    def test_self_loop_rejected(self) -> None:
        """自环 a→a 被拒绝。"""
        net = honet.HypergraphNetworkD()
        with pytest.raises(ValueError, match="重复"):
            net.add_edge(["a", "a"])

    def test_single_member_rejected(self) -> None:
        """有向交互至少需要起点与终点。"""
        net = honet.HypergraphNetworkD()
        with pytest.raises(ValueError, match="至少需要起点和终点"):
            net.add_edge(["a"])

    def test_undirected_dedupes_and_sorts_agnostically(self) -> None:
        """无向超图对成员顺序不敏感，重复成员被去重。"""
        net = honet.HypergraphNetwork()
        net.add_edge(["a", "b", "a"])
        assert net.edges == (frozenset({"a", "b"}),)


class TestSimplicialComplex:
    """单纯复形：按面存储并展开子面。"""

    def test_faces_expansion(self) -> None:
        """一个 3 顶点面贡献 7 个非空子面（C3,1 + C3,2 + C3,3）。"""
        sc = honet.SimplicialComplexNetwork(facets=[["a", "b", "c"]])
        assert sc.num_facets == 1
        assert sc.num_faces == 7
        assert sc.dimension == 2

    def test_facets_are_maximal_only(self) -> None:
        """facets 只含显式加入的极大面，不含自动子面。"""
        sc = honet.SimplicialComplexNetwork(facets=[["a", "b", "c"]])
        assert sc.facets == (frozenset({"a", "b", "c"}),)
        assert len(sc.faces()) == 7

    def test_duplicate_facet_rejected(self) -> None:
        """重复单形被拒绝。"""
        sc = honet.SimplicialComplexNetwork(facets=[["a", "b"]])
        with pytest.raises(ValueError, match="重复的单形"):
            sc.add_facet(["b", "a"])

    def test_degree_counts_facets(self) -> None:
        """度 = 参与的极大面个数。"""
        sc = honet.SimplicialComplexNetwork(facets=[["a", "b"], ["a", "c", "d"]])
        assert sc.degree("a") == 2
        assert sc.degree("b") == 1


class TestHypergraphShape:
    """超图形态：不向下封闭。"""

    def test_no_face_closure(self) -> None:
        """加入三元超边不会自动产生二元子边。"""
        hg = honet.HypergraphNetwork(edges=[["a", "b", "c"]])
        assert hg.num_edges == 1
        assert hg.edge_sizes == (3,)
        assert hg.degree("a") == 1

    def test_incidence_matrix(self) -> None:
        """关联矩阵形状与内容正确。"""
        hg = honet.HypergraphNetwork(nodes=["a", "b"], edges=[["a", "b"], ["a"]])
        assert hg.incidence_matrix() == [[1, 1], [1, 0]]

    def test_uniform_flag(self) -> None:
        """均匀性判定与边大小一致。"""
        assert honet.HypergraphNetwork(edges=[["a", "b"], ["c", "d"]]).is_uniform is True
        assert honet.HypergraphNetwork(edges=[["a", "b"], ["c", "d", "e"]]).is_uniform is False


class TestMultilayer:
    """多层网络：按层隔离存储。"""

    def test_layers_and_edges(self) -> None:
        """每层的超边互不混淆，层内重复被拒绝。"""
        ml = honet.MultilayerNetwork(layers=["work"], nodes=["a", "b", "c"])
        ml.add_edge(["a", "b"], layer="work")
        ml.add_edge(["a", "c"], layer="family")
        assert ml.layers == ("work", "family")
        assert ml.layer_edges("work") == (frozenset({"a", "b"}),)
        assert ml.num_edges == 2
        with pytest.raises(ValueError, match="已存在相同超边"):
            ml.add_edge(["b", "a"], layer="work")

    def test_degree_sums_across_layers(self) -> None:
        """总度跨层累加，单层度只看本层。"""
        ml = honet.MultilayerNetwork(nodes=["a", "b", "c"])
        ml.add_edge(["a", "b"], layer="work")
        ml.add_edge(["a", "c"], layer="family")
        assert ml.degree("a") == 2
        assert ml.layer_degree("work", "a") == 1
        assert ml.layer_degree("family", "a") == 1

    def test_missing_layer_raises(self) -> None:
        """查询不存在的层抛 KeyError。"""
        ml = honet.MultilayerNetwork(nodes=["a"])
        with pytest.raises(KeyError, match="层不存在"):
            ml.layer_edges("ghost")


class TestTemporal:
    """时序网络：按时间戳存储并排序。"""

    def test_events_sorted_by_time(self) -> None:
        """乱序插入后仍按时间升序返回。"""
        tn = honet.TemporalNetwork(events=[(3.0, ["a", "b"]), (1.0, ["a"]), (2.0, ["b", "c"])])
        assert tn.times == (1.0, 2.0, 3.0)
        assert [len(e) for e in tn.edges] == [1, 2, 2]

    def test_events_between(self) -> None:
        """按时间窗口筛选事件（闭区间）。"""
        tn = honet.TemporalNetwork(events=[(1.0, ["a"]), (2.0, ["a", "b"]), (5.0, ["c"])])
        assert tn.events_between(1.0, 2.0) == (frozenset({"a"}), frozenset({"a", "b"}))

    def test_active_nodes(self) -> None:
        """给定时点的活跃结点集合。"""
        tn = honet.TemporalNetwork(events=[(1.0, ["a", "b"]), (2.0, ["c"])])
        assert tn.active_nodes(1.0) == frozenset({"a", "b"})
        assert tn.active_nodes(9.0) == frozenset()

    def test_same_members_different_time_allowed(self) -> None:
        """同一组结点在不同时间可以重复出现，但同一时刻重复会报错。"""
        tn = honet.TemporalNetwork()
        tn.add_event(["a", "b"], time=1.0)
        tn.add_event(["a", "b"], time=2.0)
        assert tn.num_edges == 2
        with pytest.raises(ValueError, match="已存在相同事件"):
            tn.add_event(["b", "a"], time=1.0)

    def test_invalid_time_rejected(self) -> None:
        """时间戳必须是实数。"""
        tn = honet.TemporalNetwork()
        with pytest.raises(ValueError, match="时间戳"):
            tn.add_event(["a"], time="2024")  # type: ignore[arg-type]


class TestCommonInterface:
    """跨形态的公共接口行为。"""

    @pytest.mark.parametrize("name", ["HypergraphNetwork", "MultilayerNetwork", "TemporalNetwork"])
    def test_len_and_contains(self, name: str) -> None:
        """len() 与 in 依据结点表工作。"""
        net = getattr(honet, name)(nodes=["a", "b"])
        assert len(net) == 2
        assert "a" in net
        assert "zzz" not in net

    def test_degree_missing_node(self) -> None:
        """度查询不存在的结点抛 KeyError。"""
        with pytest.raises(KeyError):
            honet.HypergraphNetwork(nodes=["a"]).degree("zzz")

    def test_incidence_matrix_dimensions(self) -> None:
        """关联矩阵为 num_nodes × num_edges。"""
        net = honet.HypergraphNetwork(edges=[["a", "b"], ["b", "c"]])
        matrix = net.incidence_matrix()
        assert len(matrix) == net.num_nodes
        assert all(len(row) == net.num_edges for row in matrix)
