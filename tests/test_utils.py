"""honet.utils 度量函数的单元测试。

覆盖点：关联矩阵的形状与内容、度序列的降序与长度、团投影对孤立结点的处理。
运行方式：``uv run pytest``。
"""

from __future__ import annotations

import pytest

from honet import Hypergraph, clique_projection, degree_sequence, incidence_matrix


@pytest.fixture()
def hg() -> Hypergraph:
    """固定测试拓扑：结点 a,b,c,d；超边 {a,b,c} 与 {c,d}；d 只有 1 条边。"""
    return Hypergraph(nodes=["a", "b", "c", "d"], edges=[["a", "b", "c"], ["c", "d"]])


class TestIncidenceMatrix:
    """关联矩阵 B 的正确性。"""

    def test_shape_and_values(self, hg: Hypergraph) -> None:
        """矩阵为 4×2，行列顺序与 nodes/edges 一致。"""
        matrix = incidence_matrix(hg)
        assert len(matrix) == 4
        assert all(len(row) == 2 for row in matrix)
        # 行顺序：a, b, c, d
        assert matrix == [
            [1, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ]

    def test_empty_hypergraph(self) -> None:
        """空超图返回空矩阵。"""
        assert incidence_matrix(Hypergraph()) == []

    def test_each_column_sums_to_edge_size(self, hg: Hypergraph) -> None:
        """矩阵列和等于对应超边大小。"""
        matrix = incidence_matrix(hg)
        column_sums = [sum(row[col] for row in matrix) for col in range(hg.num_edges)]
        assert column_sums == [3, 2]


class TestDegreeSequence:
    """高阶度序列。"""

    def test_descending(self, hg: Hypergraph) -> None:
        """度序列降序，且有孤立结点时包含 0。"""
        assert degree_sequence(hg) == [2, 1, 1, 1]

    def test_with_isolated_node(self) -> None:
        """孤立结点的度记 0 并保留在序列中。"""
        hg = Hypergraph(nodes=["a", "iso"], edges=[["a", "b"]])
        assert degree_sequence(hg) == [1, 1, 0]

    def test_empty(self) -> None:
        """空图返回空列表。"""
        assert degree_sequence(Hypergraph()) == []


class TestCliqueProjection:
    """团投影到成对网络。"""

    def test_neighbors_match_hypergraph(self, hg: Hypergraph) -> None:
        """投影结果与 Hypergraph.neighbors 完全一致。"""
        projection = clique_projection(hg)
        for node in hg.nodes:
            assert projection[node] == hg.neighbors(node)

    def test_isolated_node_present(self) -> None:
        """孤立结点也出现在投影字典中，值为空集合。"""
        hg = Hypergraph(nodes=["a", "iso"], edges=[["a", "b"]])
        projection = clique_projection(hg)
        assert projection["iso"] == frozenset()
        assert projection["a"] == frozenset({"b"})

    def test_symmetry(self, hg: Hypergraph) -> None:
        """投影得到的邻接关系是对称的。"""
        projection = clique_projection(hg)
        for node, neighbors in projection.items():
            for other in neighbors:
                assert node in projection[other]

    def test_values_are_frozenset(self, hg: Hypergraph) -> None:
        """返回值为不可变集合，调用方无法篡改。"""
        assert all(isinstance(v, frozenset) for v in clique_projection(hg).values())
