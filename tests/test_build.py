"""honet.build 子包的单元测试。

覆盖点：
* 六个公开函数的真实调用（小规模真实数据，3~6 个结点，不使用 mock）；
* ``build_hypernet`` 的无权重 / 有权重 / 按年份三种分支，以及深度属性语义；
* 四个随机生成函数在固定 ``seed`` 下两次调用结果完全一致；
* 子包的惰性导出契约：``import honet`` 不拉 hypernetx，且导出符号齐全。

运行方式：``uv run pytest tests/test_build.py -q``。
"""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from honet.build import (
    build_hypernet,
    build_hypernets,
    clustered_hypergraph,
    generate_random_hypernet,
    scale_free_hypergraph,
    small_world_hypergraph,
)

#: 边表夹具的列：协议（超边）/ 国家（成员）/ 深度 / 年份。
E_COLUMNS = ("agreement", "country", "depth", "year")


@pytest.fixture()
def edges() -> pd.DataFrame:
    """3 条协议、4 个国家、跨 2 个年份的长表。"""
    return pd.DataFrame(
        {
            "agreement": ["A1", "A1", "A1", "A2", "A2", "A3"],
            "country": ["CN", "JP", "US", "CN", "DE", "JP"],
            "depth": [2.0, 2.0, 2.0, 1.0, 1.0, 3.0],
            "year": [2000, 2000, 2000, 2000, 2000, 2001],
        }
    )


@pytest.fixture()
def weights() -> pd.DataFrame:
    """各国出口额（按年份给 period 列）。"""
    return pd.DataFrame(
        {
            "reporterISO": ["CN", "JP", "US", "DE", "CN", "JP"],
            "primaryValue": [6.0, 3.0, 1.0, 2.0, 4.0, 4.0],
            "period": [2000, 2000, 2000, 2000, 2001, 2001],
        }
    )


def _members(H) -> dict:
    """把超图取成 ``{边: 成员列表}``，便于断言拓扑。"""
    return {edge: list(H.edges[edge]) for edge in H.edges}


class TestLazyExports:
    """子包惰性导出契约。"""

    def test_exports_complete(self) -> None:
        """六个公开函数都可通过子包访问且可调用。"""
        from honet import build

        assert build.__all__ == [
            "build_hypernet",
            "build_hypernets",
            "clustered_hypergraph",
            "generate_random_hypernet",
            "scale_free_hypergraph",
            "small_world_hypergraph",
        ]
        for name in build.__all__:
            assert callable(getattr(build, name))

    def test_import_honet_does_not_pull_hypernetx(self) -> None:
        """``import honet`` 不得触发 hypernetx 导入（零第三方依赖契约）。"""
        code = "import sys, honet; print('hypernetx' in sys.modules or 'numpy' in sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "False"


class TestBuildHypernet:
    """由长表构造单张超图。"""

    def test_unweighted(self, edges: pd.DataFrame) -> None:
        """无权重：超边 = agreement，成员 = country，边权重取默认 1。"""
        H = build_hypernet(edges)
        assert _members(H) == {"A1": ["CN", "JP", "US"], "A2": ["CN", "DE"], "A3": ["JP"]}
        assert {edge: H.edges[edge].weight for edge in H.edges} == {
            "A1": 1,
            "A2": 1,
            "A3": 1,
        }

    def test_weighted_cell_weights(self, edges: pd.DataFrame, weights: pd.DataFrame) -> None:
        """有 ``w``：关联矩阵单元格权重 = 归一化出口额 × 深度。

        注意 ``merge_ew`` 按 ``reporterISO`` 对**整张权重表**汇总（不按年份
        过滤），因此这里的期望值用全表出口额：CN=10、JP=7、US=1、DE=2。
        """
        H = build_hypernet(edges, weights)
        matrix, row_index, col_index = H.incidence_matrix(index=True, weights=True)
        table = pd.DataFrame(matrix.toarray(), index=row_index, columns=col_index)
        # A1 的参与国 CN=10 JP=7 US=1，合计 18，深度 2
        assert table.loc["CN", "A1"] == pytest.approx(10 / 18 * 2)
        assert table.loc["JP", "A1"] == pytest.approx(7 / 18 * 2)
        assert table.loc["US", "A1"] == pytest.approx(1 / 18 * 2)
        # A2 的参与国 CN=10 DE=2，合计 12，深度 1
        assert table.loc["CN", "A2"] == pytest.approx(10 / 12 * 1)
        assert table.loc["DE", "A2"] == pytest.approx(2 / 12 * 1)

    def test_by_year_filters_and_sets_depth(self, edges: pd.DataFrame) -> None:
        """给定年份：只保留该年数据，且边权重与 depth 属性都等于深度。"""
        H = build_hypernet(edges, i=2000)
        assert _members(H) == {"A1": ["CN", "JP", "US"], "A2": ["CN", "DE"]}
        for edge, depth in (("A1", 2.0), ("A2", 1.0)):
            assert H.edges[edge].weight == depth
            assert H.edges[edge].properties["depth"] == depth

    def test_by_year_with_weights(self, edges: pd.DataFrame, weights: pd.DataFrame) -> None:
        """按年份 + 权重：年份同时过滤 e 与 w，单元格权重按该年出口额归一化。"""
        H = build_hypernet(edges, weights, 2001)
        matrix, row_index, col_index = H.incidence_matrix(index=True, weights=True)
        table = pd.DataFrame(matrix.toarray(), index=row_index, columns=col_index)
        assert list(table.columns) == ["A3"]
        # 2001 年 JP=4，A3 只有 JP，深度 3
        assert table.loc["JP", "A3"] == pytest.approx(4 / 4 * 3)
        assert H.edges["A3"].weight == 3.0

    def test_missing_column_raises(self) -> None:
        """缺 ``country`` 列时由 merge_ew 的列校验抛 ValueError。"""
        broken = pd.DataFrame({"agreement": ["A1"], "depth": [1.0]})
        with pytest.raises(ValueError, match="country"):
            build_hypernet(broken, pd.DataFrame({"reporterISO": ["CN"], "primaryValue": [1.0]}))


class TestBuildHypernets:
    """按年份批量构造。"""

    def test_returns_one_hypergraph_per_year(self, edges: pd.DataFrame) -> None:
        """键与年份一一对应，且各年份内容正确。"""
        nets = build_hypernets(edges, [2000, 2001])
        assert sorted(nets) == [2000, 2001]
        assert sorted(nets[2000].nodes) == ["CN", "DE", "JP", "US"]
        assert sorted(nets[2001].nodes) == ["JP"]
        assert _members(nets[2001]) == {"A3": ["JP"]}

    def test_year_without_data_gives_empty_hypergraph(self, edges: pd.DataFrame) -> None:
        """表中没有的年份仍返回空超图，不丢键。"""
        nets = build_hypernets(edges, [1999])
        assert list(nets) == [1999]
        assert nets[1999].shape == (0, 0)

    def test_with_weights(self, edges: pd.DataFrame, weights: pd.DataFrame) -> None:
        """带权重时各年份超图的边权重等于该年深度。"""
        nets = build_hypernets(edges, [2000, 2001], weights)
        assert nets[2000].edges["A1"].weight == 2.0
        assert nets[2001].edges["A3"].weight == 3.0


class TestGenerateRandomHypernet:
    """均匀随机超图。"""

    def test_shape_and_edge_sizes(self) -> None:
        """边数精确、每条边大小落在 [2, max_edge_size]、成员来自结点集。"""
        H = generate_random_hypernet(n_nodes=5, n_edges=6, max_edge_size=3, seed=1)
        assert H.shape[1] == 6
        for edge in H.edges:
            members = list(H.edges[edge])
            assert 2 <= len(members) <= 3
            assert len(set(members)) == len(members)
            assert set(members) <= set(range(5))

    def test_same_seed_reproducible(self) -> None:
        """固定 seed 下两次调用得到完全相同的拓扑。"""
        first = generate_random_hypernet(n_nodes=6, n_edges=8, max_edge_size=4, seed=42)
        second = generate_random_hypernet(n_nodes=6, n_edges=8, max_edge_size=4, seed=42)
        assert _members(first) == _members(second)

    def test_different_seed_differs(self) -> None:
        """不同 seed 一般得到不同拓扑（用较大规模降低偶然相同概率）。"""
        first = generate_random_hypernet(n_nodes=6, n_edges=8, max_edge_size=4, seed=1)
        second = generate_random_hypernet(n_nodes=6, n_edges=8, max_edge_size=4, seed=2)
        assert _members(first) != _members(second)

    def test_invalid_arguments(self) -> None:
        """越界参数提前抛 ValueError 并指明参数名。"""
        with pytest.raises(ValueError, match="max_edge_size"):
            generate_random_hypernet(n_nodes=5, n_edges=1, max_edge_size=1)
        with pytest.raises(ValueError, match="n_nodes"):
            generate_random_hypernet(n_nodes=3, n_edges=1, max_edge_size=4)


class TestScaleFreeHypergraph:
    """无标度超图（优先连接）。"""

    def test_edge_count_and_head_tail(self) -> None:
        """边数等于 m，每条边都有非空 head 与 k 个 tail，且 head/tail 不相交。"""
        H = scale_free_hypergraph(n=6, k=2, m=5, seed=3)
        assert H.shape[1] == 5
        for edge in H.edges:
            head = H.edges[edge].head
            tail = H.edges[edge].tail
            assert len(tail) == 2
            assert 1 <= len(head) <= 1  # hsize ∈ [1, min(k-1, ...)] = 1
            assert set(head).isdisjoint(tail)
            assert set(list(H.edges[edge])) == set(tail) | set(head)

    def test_same_seed_reproducible(self) -> None:
        """固定 seed 下两次调用得到完全相同的拓扑。"""
        first = scale_free_hypergraph(n=6, k=2, m=5, seed=11)
        second = scale_free_hypergraph(n=6, k=2, m=5, seed=11)
        assert _members(first) == _members(second)

    def test_prefers_high_degree_nodes(self) -> None:
        """幂指数越大越集中：gamma 大时度分布的极差应不小于 gamma 小时。"""
        low = scale_free_hypergraph(n=6, k=2, m=30, gamma=1.0, seed=5)
        high = scale_free_hypergraph(n=6, k=2, m=30, gamma=4.0, seed=5)

        def spread(H) -> int:
            degrees = [H.degree(node) for node in H.nodes]
            return max(degrees) - min(degrees)

        assert spread(high) >= spread(low)

    def test_invalid_arguments(self) -> None:
        """k > n 等非法组合提前抛 ValueError。"""
        with pytest.raises(ValueError, match="k"):
            scale_free_hypergraph(n=4, k=5, m=3)
        with pytest.raises(ValueError, match="gamma"):
            scale_free_hypergraph(n=4, k=2, m=3, gamma=0)


class TestClusteredHypergraph:
    """聚簇超图（社区结构）。"""

    def test_edge_count_and_structure(self) -> None:
        """边数等于 m，成员不重复，head/tail 不相交且都属于结点集。"""
        H = clustered_hypergraph(n=6, k=2, m=6, num_clusters=2, seed=5)
        assert H.shape[1] == 6
        for edge in H.edges:
            head = H.edges[edge].head
            tail = H.edges[edge].tail
            assert len(tail) == 2
            assert set(head).isdisjoint(tail)
            assert set(list(H.edges[edge])) <= set(range(6))

    def test_same_seed_reproducible(self) -> None:
        """固定 seed 下两次调用得到完全相同的拓扑。"""
        first = clustered_hypergraph(n=6, k=2, m=6, num_clusters=2, seed=9)
        second = clustered_hypergraph(n=6, k=2, m=6, num_clusters=2, seed=9)
        assert _members(first) == _members(second)

    def test_intra_cluster_dominates(self) -> None:
        """簇内边占多数：结点对同簇的共现次数应多于跨簇。"""
        H = clustered_hypergraph(n=6, k=2, m=60, num_clusters=2, seed=4)
        clusters = {0: {0, 1, 2}, 1: {3, 4, 5}}
        same_cluster = cross_cluster = 0
        for edge in H.edges:
            members = set(H.edges[edge])
            for node in members:
                partner = clusters[1] if node in clusters[0] else clusters[0]
                hits = len(members & partner)
                if node in clusters[0]:
                    same_cluster += len(members & clusters[0]) - 1
                    cross_cluster += hits
                else:
                    same_cluster += len(members & clusters[1]) - 1
                    cross_cluster += hits
        assert same_cluster > cross_cluster

    def test_invalid_arguments(self) -> None:
        """num_clusters < 1 提前抛 ValueError。"""
        with pytest.raises(ValueError, match="num_clusters"):
            clustered_hypergraph(n=6, k=2, m=3, num_clusters=0)


class TestSmallWorldHypergraph:
    """小世界超图（环点阵 + 重连）。"""

    def test_edge_count_and_tail_is_ring_neighbourhood(self) -> None:
        """边数等于 m；tail 恰为中心点之后的 k 个环上邻居，head 不在 tail 内。"""
        H = small_world_hypergraph(n=6, k=2, m=5, rewire_prob=0.2, seed=7)
        assert H.shape[1] == 5
        for edge in H.edges:
            tail = H.edges[edge].tail
            head = H.edges[edge].head
            assert len(tail) == 2
            center = (tail[0] - 1) % 6
            assert tail == [(center + 1) % 6, (center + 2) % 6]
            assert len(head) == 1
            assert set(head).isdisjoint(tail)

    def test_same_seed_reproducible(self) -> None:
        """固定 seed 下两次调用得到完全相同的拓扑。"""
        first = small_world_hypergraph(n=6, k=2, m=5, seed=13)
        second = small_world_hypergraph(n=6, k=2, m=5, seed=13)
        assert _members(first) == _members(second)

    def test_no_rewire_keeps_head_nearby(self) -> None:
        """rewire_prob=0 时 head 始终落在中心点的 2k 邻域内（无长程捷径）。"""
        H = small_world_hypergraph(n=8, k=2, m=16, rewire_prob=0.0, seed=2)
        for edge in H.edges:
            tail = H.edges[edge].tail
            center = (tail[0] - 1) % 8
            nearby = {(center + offset) % 8 for offset in range(-4, 5)}
            assert set(H.edges[edge].head) <= nearby

    def test_full_rewire_can_shortcut(self) -> None:
        """rewire_prob=1 时 head 取剩余结点中的随机点，可能出现远端点。

        取 ``n=12, k=2``：``nearby`` 窗口为 ``-4..4`` 共 9 个位置，小于 12，
        因此存在真正的“远端点”，重连才可能跳出窗口。
        """
        H = small_world_hypergraph(n=12, k=2, m=24, rewire_prob=1.0, seed=2)
        far = 0
        for edge in H.edges:
            tail = H.edges[edge].tail
            center = (tail[0] - 1) % 12
            nearby = {(center + offset) % 12 for offset in range(-4, 5)}
            if not set(H.edges[edge].head) <= nearby:
                far += 1
        assert far > 0

    def test_invalid_arguments(self) -> None:
        """k >= n 会死循环，改为提前抛 ValueError。"""
        with pytest.raises(ValueError, match="k"):
            small_world_hypergraph(n=4, k=4, m=2)
        with pytest.raises(ValueError, match="rewire_prob"):
            small_world_hypergraph(n=4, k=2, m=2, rewire_prob=1.5)
