"""``Hypergraph.from_dataframe`` 的单元测试。

覆盖：默认/自定义列名、顺序保持、权重与标签列、缺列、空值与脏数据拒绝。
运行方式：``uv run pytest``。
"""

from __future__ import annotations

import pandas as pd
import pytest

from honet import (
    Hypergraph,
    HypergraphNetwork,
    HypergraphNetworkD,
    HypergraphNetworkW,
    MultilayerNetwork,
    SimplicialComplexNetwork,
    TemporalNetwork,
)


class TestFromDataFrame:
    """把长表（edge, node[, weight][, label]）转换为 Hypergraph。"""

    def test_default_columns(self) -> None:
        """默认列名为 ``edge`` / ``node``。"""
        frame = pd.DataFrame({"edge": ["e1", "e1", "e2"], "node": ["a", "b", "a"]})
        graph = Hypergraph.from_dataframe(frame)

        assert graph.node_set == {"a", "b"}
        assert graph.num_edges == 2
        assert graph.degree("a") == 2
        assert graph.edges[0].members == frozenset({"a", "b"})

    def test_custom_columns_preserve_order(self) -> None:
        """自定义列名；超边与结点均按首次出现顺序排列。"""
        frame = pd.DataFrame({"paper": ["p2", "p1", "p2"], "author": ["c", "a", "b"]})
        graph = Hypergraph.from_dataframe(frame, edge_col="paper", node_col="author")

        assert [edge.members for edge in graph.edges] == [
            frozenset({"c", "b"}),
            frozenset({"a"}),
        ]
        assert graph.nodes == ("c", "b", "a")

    def test_weight_and_label_columns(self) -> None:
        """权重列转成 float，数值标签转成 str。"""
        frame = pd.DataFrame(
            {
                "e": ["e1", "e1", "e2"],
                "n": ["a", "b", "a"],
                "w": [2, 2.0, 0.5],
                "lab": [1, 1, 2],
            }
        )
        graph = Hypergraph.from_dataframe(
            frame,
            edge_col="e",
            node_col="n",
            weight_col="w",
            label_col="lab",
        )

        assert [edge.weight for edge in graph.edges] == [2.0, 0.5]
        assert [edge.label for edge in graph.edges] == ["1", "2"]

    def test_weight_defaults_to_one(self) -> None:
        """不给权重列时权重为 1.0，与手写 add_edge 一致。"""
        graph = Hypergraph.from_dataframe(pd.DataFrame({"edge": ["e1"], "node": ["a"]}))
        assert graph.edges[0].weight == 1.0

    def test_duplicate_rows_are_deduplicated(self) -> None:
        """同一超边内重复出现的结点只计一次。"""
        frame = pd.DataFrame({"edge": ["e1", "e1", "e1"], "node": ["a", "a", "b"]})
        graph = Hypergraph.from_dataframe(frame)
        assert graph.edges[0].members == frozenset({"a", "b"})

    def test_empty_frame(self) -> None:
        """空表得到空超图。"""
        graph = Hypergraph.from_dataframe(pd.DataFrame({"edge": [], "node": []}))
        assert graph.num_nodes == 0
        assert graph.num_edges == 0

    def test_missing_column(self) -> None:
        """缺少指定列时抛 KeyError 并列出实际列。"""
        frame = pd.DataFrame({"edge": ["e1"], "node": ["a"]})
        with pytest.raises(KeyError, match="缺少列"):
            Hypergraph.from_dataframe(frame, weight_col="w")

    def test_missing_node_rejected(self) -> None:
        """结点为空属于脏数据。"""
        frame = pd.DataFrame({"edge": ["e1", "e1"], "node": ["a", None]})
        with pytest.raises(ValueError, match="node.*为空"):
            Hypergraph.from_dataframe(frame)

    def test_pandas_na_rejected(self) -> None:
        """``pandas.NA`` 与 ``None`` 一样视为缺失。"""
        frame = pd.DataFrame(
            {
                "edge": pd.Series(["e1"], dtype="string"),
                "node": pd.Series([pd.NA], dtype="string"),
            }
        )
        with pytest.raises(ValueError, match="为空"):
            Hypergraph.from_dataframe(frame)

    def test_non_real_weight_rejected(self) -> None:
        """权重列出现非数值。"""
        frame = pd.DataFrame({"edge": ["e1"], "node": ["a"], "w": ["heavy"]})
        with pytest.raises(ValueError, match="不是实数"):
            Hypergraph.from_dataframe(frame, weight_col="w")

    def test_inconsistent_weight_rejected(self) -> None:
        """同一超边的不同行给出不同权重。"""
        frame = pd.DataFrame({"edge": ["e1", "e1"], "node": ["a", "b"], "w": [1.0, 2.0]})
        with pytest.raises(ValueError, match="权重不一致"):
            Hypergraph.from_dataframe(frame, weight_col="w")

    def test_inconsistent_label_rejected(self) -> None:
        """同一超边的不同行给出不同标签。"""
        frame = pd.DataFrame({"edge": ["e1", "e1"], "node": ["a", "b"], "lab": ["x", "y"]})
        with pytest.raises(ValueError, match="标签不一致"):
            Hypergraph.from_dataframe(frame, label_col="lab")

    def test_duplicate_member_sets_rejected(self) -> None:
        """不同标识但成员与标签完全相同，按重复超边拒绝。"""
        frame = pd.DataFrame({"edge": ["e1", "e2"], "node": ["a", "a"]})
        with pytest.raises(ValueError, match="重复"):
            Hypergraph.from_dataframe(frame)


class TestNetworkFromDataFrame:
    """四种形态的高阶网络都能直接由 DataFrame 构造。"""

    def test_hypergraph_network(self) -> None:
        """超图网络：按超边分组，不向下封闭。"""
        frame = pd.DataFrame({"edge": ["e1", "e1", "e2"], "node": ["a", "b", "a"]})
        graph = HypergraphNetwork.from_dataframe(frame)

        assert graph.num_edges == 2
        assert graph.degree("a") == 2
        assert graph.edges[0] == frozenset({"a", "b"})

    def test_weighted_variant_reads_weight_column(self) -> None:
        """加权变体：权重列必须同组一致。"""
        frame = pd.DataFrame({"edge": ["e1", "e1"], "node": ["a", "b"], "w": [2.0, 2.0]})
        graph = HypergraphNetworkW.from_dataframe(frame, weight_col="w")
        assert graph.weights == (2.0,)

    def test_unweighted_variant_rejects_non_unit_weight(self) -> None:
        """无权变体收到非 1.0 权重时提示改用 W / WD 变体。"""
        frame = pd.DataFrame({"edge": ["e1"], "node": ["a"], "w": [2.0]})
        with pytest.raises(TypeError, match="W"):
            HypergraphNetwork.from_dataframe(frame, weight_col="w")

    def test_directed_variant_keeps_row_order(self) -> None:
        """有向变体按行序保存成员元组。"""
        frame = pd.DataFrame({"edge": ["e1", "e1"], "node": ["b", "a"]})
        graph = HypergraphNetworkD.from_dataframe(frame)
        assert graph.edges[0] == ("b", "a")

    def test_simplicial_facets_and_faces(self) -> None:
        """单纯复形：一行组即一个极大面，子面自动向下封闭。"""
        frame = pd.DataFrame({"edge": ["s1", "s1", "s1"], "node": ["a", "b", "c"]})
        complex_ = SimplicialComplexNetwork.from_dataframe(frame)

        assert complex_.num_facets == 1
        assert complex_.dimension == 2
        assert complex_.num_faces == 7
        assert frozenset({"a", "b"}) in complex_.faces()

    def test_multilayer_groups_by_layer_and_edge(self) -> None:
        """同一个超边标识出现在不同层时互不合并。"""
        frame = pd.DataFrame(
            {
                "edge": ["e1", "e1", "e1"],
                "node": ["a", "b", "a"],
                "layer": ["work", "work", "family"],
            }
        )
        graph = MultilayerNetwork.from_dataframe(frame)

        assert graph.layers == ("work", "family")
        assert graph.layer_degree("work", "a") == 1
        assert graph.degree("a") == 2

    def test_temporal_splits_events_by_time(self) -> None:
        """同一成员集合在不同时刻算不同事件，并按时间排序。"""
        frame = pd.DataFrame(
            {
                "edge": ["e1", "e1", "e2", "e2"],
                "node": ["a", "b", "a", "b"],
                "time": [2, 2, 1, 1],
            }
        )
        graph = TemporalNetwork.from_dataframe(frame)

        assert graph.times == (1.0, 2.0)
        assert graph.num_edges == 2

    def test_temporal_rejects_non_real_time(self) -> None:
        """时间列无法转成实数时报错。"""
        frame = pd.DataFrame({"edge": ["e1"], "node": ["a"], "time": ["2024-01"]})
        with pytest.raises(ValueError, match="不是实数"):
            TemporalNetwork.from_dataframe(frame)
