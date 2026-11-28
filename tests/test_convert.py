"""honet.convert 格式互转函数的单元测试。

覆盖点：超边权重（关联矩阵列和）、hypernetx ↔ XGI / HyperGraphX 互转、
hypernetx → NetworkX 团投影、hypernetx → HAT 有向超图，以及类型校验。

测试数据是 5 个结点（a~e）、3 条超边的真实小超图，关联权重显式给出，
因此「列和」与「超边大小」不相等，能真正验证权重口径而非巧合相等。

运行方式：``uv run pytest tests/test_convert.py -q``。
"""

from __future__ import annotations

from importlib.util import find_spec

import pytest

from honet.convert import (
    convert_hypergraphx_to_hypernetx,
    convert_hypernetx_to_honet,
    convert_hypernetx_to_hypergraphx,
    convert_hypernetx_to_nxgraph,
    convert_hypernetx_to_xgi,
    convert_xgi_to_hypernetx,
    get_hyperedge_weights,
    hypernetx_to_hat_directed,
)

# analysis 组的依赖缺失时整个模块无意义（连 fixture 都建不出来），直接跳过。
hnx = pytest.importorskip("hypernetx", reason="需要 hypernetx（honet[analysis]）")
nx = pytest.importorskip("networkx", reason="需要 networkx（honet[analysis]）")
pd = pytest.importorskip("pandas", reason="需要 pandas（honet[analysis]）")


# xgi / hypergraphx / HAT 是更窄的可选依赖，只按需跳过对应测试类，见下方 skipif。
def _available(module: str) -> bool:
    """判断可选依赖是否可导入；find_spec 自身抛错时也视为不可用。"""
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


HAS_XGI = _available("xgi")
HAS_HGX = _available("hypergraphx")
HAS_HAT = _available("HAT")

needs_xgi = pytest.mark.skipif(not HAS_XGI, reason="需要 xgi")
needs_hgx = pytest.mark.skipif(not HAS_HGX, reason="需要 hypergraphx")
needs_hat = pytest.mark.skipif(not HAS_HAT, reason="需要 HypergraphAnalysisToolbox（honet[hat]）")


@pytest.fixture()
def H() -> hnx.Hypergraph:
    """5 结点 3 超图，关联权重刻意不等于 1：e1=6.0、e2=1.0、e3=1.0。"""
    rows = [
        ("e1", "a", 2.0),
        ("e1", "b", 2.0),
        ("e1", "c", 2.0),
        ("e2", "c", 0.5),
        ("e2", "d", 0.5),
        ("e3", "e", 1.0),
    ]
    return hnx.Hypergraph(pd.DataFrame(rows, columns=["edges", "nodes", "weight"]))


class TestGetHyperedgeWeights:
    """超边权重 = 关联矩阵列和。"""

    def test_column_sums(self, H: hnx.Hypergraph) -> None:
        """权重等于该列关联权重之和，而不是超边大小。"""
        assert get_hyperedge_weights(H) == {"e1": 6.0, "e2": 1.0, "e3": 1.0}

    def test_default_weight_equals_edge_size(self) -> None:
        """没有显式关联权重时，权重退化为超边大小。"""
        H = hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["c", "d"]})
        assert get_hyperedge_weights(H) == {"e1": 3.0, "e2": 2.0}

    def test_values_are_python_float(self, H: hnx.Hypergraph) -> None:
        """返回的是 Python float，调用方不必再处理 numpy 标量。"""
        assert all(type(weight) is float for weight in get_hyperedge_weights(H).values())


@needs_xgi
class TestHypernetxToXgi:
    """hypernetx → XGI：结点编号、权重与反查表。"""

    def test_nodes_and_weights(self, H: hnx.Hypergraph) -> None:
        """结点按 hypernetx 顺序编号，超边成员与权重都落到 XGI。"""
        xgi = pytest.importorskip("xgi")
        X, idx_to_node = convert_hypernetx_to_xgi(H)

        assert isinstance(X, xgi.Hypergraph)
        assert sorted(idx_to_node.values()) == ["a", "b", "c", "d", "e"]
        # 权重来自关联矩阵列和，e1 的 6.0 必须保留（而不是超边大小 3）
        assert [X.edges[e]["weight"] for e in X.edges] == [6.0, 1.0, 1.0]

    def test_membership_preserved(self, H: hnx.Hypergraph) -> None:
        """每条超边的成员集合按反查表还原后与原超图一致。"""
        X, idx_to_node = convert_hypernetx_to_xgi(H)
        for edge_id, members in X.edges.members().items():
            original = set(H.edges[list(H.edges)[edge_id]])
            assert {idx_to_node[i] for i in members} == original

    def test_isolated_node_kept(self) -> None:
        """只在退化超边里出现的结点也会编号，不丢结点。"""
        H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["c"]})
        X, idx_to_node = convert_hypernetx_to_xgi(H)
        assert sorted(idx_to_node.values()) == ["a", "b", "c"]
        assert sorted(X.nodes) == [0, 1, 2]


@needs_xgi
class TestXgiToHypernetx:
    """XGI → hypernetx。"""

    def test_members_converted(self) -> None:
        """xgi 1.x 的 ``edges.members()`` 方法形态被正确读取。"""
        xgi = pytest.importorskip("xgi")
        X = xgi.Hypergraph()
        X.add_weighted_edges_from([(0, 1, 2, 2.5), (1, 2, 3.0)])

        H = convert_xgi_to_hypernetx(X)

        assert isinstance(H, hnx.Hypergraph)
        assert H.edges.incidence_dict == {0: [0, 1, 2], 1: [1, 2]}

    def test_property_style_members(self) -> None:
        """早期 xgi 的属性式 ``edges.members``（非 callable）也兼容。"""

        class _Edges:
            # 故意用属性而不是方法，模拟旧版 xgi
            members = {0: {10, 11}, 1: {11, 12}}

        class _FakeXgi:
            edges = _Edges()

        H = convert_xgi_to_hypernetx(_FakeXgi())
        assert H.edges.incidence_dict == {0: [10, 11], 1: [11, 12]}

    def test_roundtrip_structure(self, H: hnx.Hypergraph) -> None:
        """hypernetx → XGI → hypernetx 结构不变（结点名换成整数下标）。"""
        X, idx_to_node = convert_hypernetx_to_xgi(H)
        node_to_idx = {name: idx for idx, name in idx_to_node.items()}
        back = convert_xgi_to_hypernetx(X)

        # 原图超边按同一套编号映射后，成员集合应与回转图逐条一致
        expected = {frozenset(node_to_idx[n] for n in H.edges[e]) for e in H.edges}
        assert {frozenset(back.edges[e]) for e in back.edges} == expected


@needs_hgx
class TestHypernetxToHypergraphx:
    """hypernetx → HyperGraphX。"""

    def test_edges_and_weights(self, H: hnx.Hypergraph) -> None:
        """成员数 ≥ 2 的超边全部保留，权重与关联矩阵列和一致。"""
        hgx = pytest.importorskip("hypergraphx")
        G = convert_hypernetx_to_hypergraphx(H)

        assert isinstance(G, hgx.Hypergraph)
        edges = [tuple(edge) for edge in G.edges]
        assert edges == [("a", "b", "c"), ("c", "d")]
        assert [G.get_weight(edge) for edge in edges] == [6.0, 1.0]

    def test_singleton_edge_dropped(self, H: hnx.Hypergraph) -> None:
        """成员数 < 2 的退化超边被跳过（HyperGraphX 只表示成对以上的交互）。"""
        G = convert_hypernetx_to_hypergraphx(H)
        assert len(list(G.edges)) == 2


@needs_hgx
class TestHypergraphxToHypernetx:
    """HyperGraphX → hypernetx。"""

    def test_edges_renamed(self) -> None:
        """HyperGraphX 没有边名，转换后按 e1、e2… 重新编号。"""
        hgx = pytest.importorskip("hypergraphx")
        G = hgx.Hypergraph([("a", "b", "c"), ("c", "d")], weighted=False)

        H = convert_hypergraphx_to_hypernetx(G)

        assert list(H.edges) == ["e1", "e2"]
        assert H.edges.incidence_dict == {"e1": ["a", "b", "c"], "e2": ["c", "d"]}

    def test_roundtrip_structure(self, H: hnx.Hypergraph) -> None:
        """hypernetx → HyperGraphX → hypernetx 的成员集合不变。"""
        back = convert_hypergraphx_to_hypernetx(convert_hypernetx_to_hypergraphx(H))
        original = {frozenset(H.edges[edge]) for edge in H.edges if len(H.edges[edge]) >= 2}
        assert {frozenset(back.edges[edge]) for edge in back.edges} == original

    def test_type_error(self) -> None:
        """非 HyperGraphX 对象抛 TypeError（与旧实现一致）。"""
        with pytest.raises(TypeError):
            convert_hypergraphx_to_hypernetx({"e1": ["a", "b"]})


class TestHypernetxToNxgraph:
    """hypernetx → NetworkX 团投影。"""

    def test_clique_projection(self, H: hnx.Hypergraph) -> None:
        """每条超边展开成完全图：e1 三条边 + e2 一条边。"""
        G = convert_hypernetx_to_nxgraph(H)

        assert isinstance(G, nx.Graph)
        assert G.number_of_edges() == 4
        assert set(map(frozenset, G.edges)) == {
            frozenset({"a", "b"}),
            frozenset({"a", "c"}),
            frozenset({"b", "c"}),
            frozenset({"c", "d"}),
        }

    def test_singleton_edge_adds_no_edge(self, H: hnx.Hypergraph) -> None:
        """单成员超边 e3 不产生自环，结点 e 因此不出现在图中。"""
        G = convert_hypernetx_to_nxgraph(H)
        assert "e" not in G.nodes
        assert not list(nx.selfloop_edges(G))

    def test_type_error(self) -> None:
        """非 hypernetx 对象抛 TypeError（与旧实现一致）。"""
        with pytest.raises(TypeError):
            convert_hypernetx_to_nxgraph({"e1": ["a", "b"]})


@needs_hat
class TestHypernetxToHatDirected:
    """hypernetx → 有向 HAT 超图。"""

    def test_nodes_and_edges(self, H: hnx.Hypergraph) -> None:
        """结点按名字排序编号，超边数与原图一致。"""
        HAT = pytest.importorskip("HAT", reason="需要 HypergraphAnalysisToolbox")
        HG = hypernetx_to_hat_directed(H)

        assert isinstance(HG, HAT.Hypergraph)
        assert HG.directed is True
        assert HG.nnodes == 5
        assert HG.nedges == 3

    def test_node_names_kept(self, H: hnx.Hypergraph) -> None:
        """结点表的 Names 列保留原名字，且按名字排序。"""
        HG = hypernetx_to_hat_directed(H)
        assert HG.nodes["Names"].tolist() == ["a", "b", "c", "d", "e"]
        assert HG.nodes["Nodes"].tolist() == [0, 1, 2, 3, 4]

    def test_head_equals_tail(self, H: hnx.Hypergraph) -> None:
        """无向超边转有向时，head 与 tail 都取全部成员（边内双向影响）。"""
        HG = hypernetx_to_hat_directed(H)
        for _, row in HG.edges.iterrows():
            assert list(row["head"]) == list(row["Nodes"])
            assert list(row["tail"]) == list(row["Nodes"])

    def test_compress_false_keeps_all_edges(self) -> None:
        """compress=False：成员集合相同的重复超边不会被合并掉。"""
        H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "a"]})
        HG = hypernetx_to_hat_directed(H)
        assert HG.nedges == 2


class TestHypernetxToHonet:
    """hypernetx.Hypergraph -> honet.Hypergraph 的结构、标签、权重与去重。"""

    def test_structure_preserved(self, H: hnx.Hypergraph) -> None:
        """结点数、边数、高阶度与源超图一致。"""
        from honet import Hypergraph

        g = convert_hypernetx_to_honet(H)
        assert isinstance(g, Hypergraph)
        assert g.num_nodes == len(list(H.nodes))
        assert g.num_edges == len(list(H.edges))
        # 高阶度逐个核对：结点的度 = 包含它的超边条数
        incidence = H.edges.incidence_dict
        for node in H.nodes:
            expected = sum(1 for members in incidence.values() if node in members)
            assert g.degree(node) == expected

    def test_labels_preserved(self, H: hnx.Hypergraph) -> None:
        """边名迁移为 label。"""
        g = convert_hypernetx_to_honet(H)
        assert sorted(e.label for e in g.edges) == sorted(str(e) for e in H.edges)

    def test_labels_dropped_when_disabled(self, H: hnx.Hypergraph) -> None:
        """keep_labels=False 时 label 全为 None。"""
        g = convert_hypernetx_to_honet(H, keep_labels=False)
        assert all(e.label is None for e in g.edges)

    def test_weights_preserved(self) -> None:
        """边权重迁移；hypernetx 2.4.3 的权重存在 edges.properties 里。"""
        H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        H.edges["e1"].weight = 5.0
        H.edges["e2"].weight = 2.5

        g = convert_hypernetx_to_honet(H)
        weights = {e.label: e.weight for e in g.edges}
        assert weights == {"e1": 5.0, "e2": 2.5}

    def test_weights_dropped_when_disabled(self) -> None:
        """keep_weights=False 时全部回落到默认 1.0。"""
        H = hnx.Hypergraph({"e1": ["a", "b"]})
        H.edges["e1"].weight = 9.0

        g = convert_hypernetx_to_honet(H, keep_weights=False)
        assert g.edges[0].weight == 1.0

    def test_duplicate_member_sets_merged_by_default(self) -> None:
        """成员集相同的重复边默认合并，权重累加。"""
        H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "a"]})
        H.edges["e1"].weight = 2.0
        H.edges["e2"].weight = 3.0

        g = convert_hypernetx_to_honet(H)
        assert g.num_edges == 1
        assert abs(g.edges[0].weight - 5.0) < 1e-12

    def test_duplicate_member_sets_kept_when_disabled(self) -> None:
        """merge_duplicates=False 时保留为两条边（靠不同 label 区分）。"""
        H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "a"]})

        g = convert_hypernetx_to_honet(H, merge_duplicates=False)
        assert g.num_edges == 2
        assert sorted(e.label for e in g.edges) == ["e1", "e2"]

    def test_duplicate_without_labels_raises(self) -> None:
        """不合并且不给 label 时，数据无法无损表达，应抛 ValueError。"""
        H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "a"]})

        with pytest.raises(ValueError, match="重复的超边"):
            convert_hypernetx_to_honet(H, keep_labels=False, merge_duplicates=False)

    def test_rejects_honet_input(self) -> None:
        """传入 honet.Hypergraph 应给出明确提示，而不是后续属性错误。"""
        from honet import Hypergraph

        with pytest.raises(TypeError, match="无需转换"):
            convert_hypernetx_to_honet(Hypergraph())

    def test_rejects_non_hypergraph(self) -> None:
        """传入无关对象应抛 TypeError。"""
        with pytest.raises(TypeError, match="需要 hypernetx 超图"):
            convert_hypernetx_to_honet(object())

    def test_result_usable_by_topology_api(self, H: hnx.Hypergraph) -> None:
        """转换结果能直接喂给 incidence_matrix / degree_sequence。"""
        from honet import degree_sequence, incidence_matrix

        g = convert_hypernetx_to_honet(H)
        matrix = incidence_matrix(g)
        assert len(matrix) == g.num_nodes
        assert len(matrix[0]) == g.num_edges

        degrees = degree_sequence(g)
        assert len(degrees) == g.num_nodes
        assert sum(degrees) == sum(e.size for e in g.edges)

    def test_empty_hypergraph(self) -> None:
        """空超图应返回空容器，而不是报错。"""
        g = convert_hypernetx_to_honet(hnx.Hypergraph())
        assert g.num_nodes == 0
        assert g.num_edges == 0
