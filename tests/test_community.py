"""honet.community 社区划分函数的单元测试。

覆盖点：``Method`` 枚举、五种划分方法（random / basic / nmf / recursive /
hierarchy）、分区检查 ``check_partition``、统一入口
``extract_community_partition`` 以及两个比较函数 ``compare_methods`` /
``compare_graphs``，并验证旧驼峰名与 snake_case 名指向同一个对象。

测试数据是真实的 hypernetx 超图（3~7 个结点、2~6 条超边），不使用任何 mock：

* ``H_two_groups`` —— 6 条超边、7 个结点，明显分成 ``{a,b,c,d}`` 与
  ``{e,f,g}`` 两块（d 是桥结点）；
* ``H_chained`` —— 3 条超边、6 个结点，超边两两共享结点呈链状，用于验证
  划分覆盖性与层次聚类的切树行为；
* ``H_small`` —— 2 条超边的极小图，用于验证退化输入。

注意：hypernetx 的模块度算法内部用集合迭代，部分方法的**具体成员归属**在
不同进程间可能不同（模块度与社区数稳定）。因此断言针对不变量（覆盖性、
社区数、模块度数值、可复现性），不断言具体分到哪个社区。

运行方式：``uv run pytest tests/test_community.py -q``。
"""

from __future__ import annotations

from importlib.util import find_spec

import pytest

from honet.community import (
    Basic_Method,
    Hierarchy_Method,
    Method,
    NMF_Method,
    Random_Method,
    Recursive_Method,
    basic_method,
    check_partition,
    compare_graphs,
    compare_methods,
    extract_community_partition,
    hierarchy_method,
    nmf_method,
    random_method,
    recursive_method,
)

# analysis 组的依赖缺失时整个模块无意义，直接跳过。
hnx = pytest.importorskip("hypernetx", reason="需要 hypernetx（honet[analysis]）")
pytest.importorskip("networkx", reason="需要 networkx（honet[analysis]）")
pytest.importorskip("numpy", reason="需要 numpy（honet[analysis]）")
pytest.importorskip("scipy", reason="需要 scipy（honet[analysis]）")
pytest.importorskip("sklearn", reason="需要 scikit-learn（honet[analysis]）")


def _available(module: str) -> bool:
    """判断可选依赖是否可导入；find_spec 自身抛错时也视为不可用。"""
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


HAS_HGX = _available("hypergraphx")

needs_hgx = pytest.mark.skipif(not HAS_HGX, reason="需要 hypergraphx")


@pytest.fixture()
def H_two_groups() -> hnx.Hypergraph:
    """7 结点 6 超边：{a,b,c,d} 与 {e,f,g} 两块，d 为桥结点。"""
    return hnx.Hypergraph(
        {
            "e1": ["a", "b", "c"],
            "e2": ["a", "b", "d"],
            "e3": ["a", "c", "d"],
            "e4": ["d", "e", "f"],
            "e5": ["d", "e", "g"],
            "e6": ["e", "f", "g"],
        }
    )


@pytest.fixture()
def H_chained() -> hnx.Hypergraph:
    """6 结点 3 超边，超边两两共享结点（e1-e2 共享 c，e2-e3 共享 d）。"""
    return hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["c", "d"], "e3": ["d", "e", "f"]})


@pytest.fixture()
def H_small() -> hnx.Hypergraph:
    """4 结点 2 超边的极小图。"""
    return hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["c", "d"]})


def _covered(partition) -> set:
    """把分区摊平成结点集合，用于断言覆盖性。"""
    return {node for group in partition for node in group}


class TestMethod:
    """Method 枚举的成员与取值。"""

    def test_members(self) -> None:
        """成员名与旧库完全一致（常量标识符保持原样）。"""
        assert [member.name for member in Method] == [
            "KUMAR",
            "NMF",
            "Random",
            "LAST_STEP",
            "RECURSIVE",
            "Basic",
            "Hierarchy",
        ]

    def test_unique_values(self) -> None:
        """@unique 保证各成员取值互不相同。"""
        assert len({member.value for member in Method}) == len(list(Method))

    def test_lookup_by_name(self) -> None:
        """可以按名字反查成员。"""
        assert Method["KUMAR"] is Method.KUMAR
        with pytest.raises(KeyError):
            Method["kumar"]


class TestLegacyAliases:
    """旧驼峰名保留为别名，指向同一个函数对象。"""

    @pytest.mark.parametrize(
        ("legacy", "modern"),
        [
            (Random_Method, random_method),
            (Basic_Method, basic_method),
            (NMF_Method, nmf_method),
            (Recursive_Method, recursive_method),
            (Hierarchy_Method, hierarchy_method),
        ],
    )
    def test_alias_is_same_object(self, legacy, modern) -> None:
        """别名与新名是同一个对象，而不是包装了一层。"""
        assert legacy is modern

    def test_alias_is_callable(self) -> None:
        """包属性不会被同名子模块遮蔽成模块对象。"""
        import honet.community as community

        assert callable(community.random_method)
        assert callable(community.extract_community_partition)


class TestRandomMethod:
    """随机均分基线。"""

    def test_covers_all_nodes(self, H_two_groups) -> None:
        """分区必须覆盖全部结点且不重复。"""
        partition = random_method(H_two_groups, n_components=3, seed=0)
        assert len(partition) == 3
        assert _covered(partition) == set(H_two_groups.nodes)
        assert sum(len(group) for group in partition) == len(list(H_two_groups.nodes))

    def test_sizes_differ_by_at_most_one(self, H_two_groups) -> None:
        """轮转分配使各社区大小最多相差 1。"""
        partition = random_method(H_two_groups, n_components=3, seed=0)
        sizes = [len(group) for group in partition]
        assert max(sizes) - min(sizes) <= 1

    def test_seed_is_reproducible(self, H_two_groups) -> None:
        """相同 seed 得到相同分区。"""
        assert random_method(H_two_groups, 3, seed=7) == random_method(H_two_groups, 3, seed=7)

    def test_different_seeds_differ(self, H_two_groups) -> None:
        """不同 seed 至少在一个结点上分法不同（7 个结点、3 组，冲突概率极低）。"""
        assert random_method(H_two_groups, 3, seed=1) != random_method(H_two_groups, 3, seed=2)

    def test_more_components_than_nodes(self, H_small) -> None:
        """社区数多于结点数时，多出来的社区是空列表。"""
        partition = random_method(H_small, n_components=6, seed=0)
        assert len(partition) == 6
        assert sum(1 for group in partition if not group) >= 2

    def test_rejects_non_positive_components(self, H_small) -> None:
        """n_components < 1 抛 ValueError。"""
        with pytest.raises(ValueError, match="n_components"):
            random_method(H_small, n_components=0)


class TestBasicMethod:
    """团投影 + Louvain。"""

    def test_partition_is_valid(self, H_two_groups) -> None:
        """分区覆盖全部结点，且每块非空。"""
        partition = basic_method(H_two_groups)
        assert _covered(partition) == set(H_two_groups.nodes)
        assert all(group for group in partition)

    def test_returns_node_labels_not_indexes(self, H_two_groups) -> None:
        """返回的是原始结点标签（字符串），不是 Louvain 的整数下标。"""
        partition = basic_method(H_two_groups)
        assert all(isinstance(node, str) for node in _covered(partition))

    def test_recovers_two_blocks(self, H_two_groups) -> None:
        """结构明显的两块应被识别出来。"""
        partition = basic_method(H_two_groups)
        assert len(partition) == 2
        assert {frozenset(group) for group in partition} == {
            frozenset({"a", "b", "c", "d"}),
            frozenset({"e", "f", "g"}),
        }

    def test_is_deterministic(self, H_two_groups) -> None:
        """固定 seed 时结果可复现。"""
        assert basic_method(H_two_groups) == basic_method(H_two_groups)

    def test_works_on_minimal_graph(self, H_small) -> None:
        """极小图也能给出覆盖全部结点的分区。"""
        partition = basic_method(H_small)
        assert _covered(partition) == set(H_small.nodes)


class TestNmfMethod:
    """非负矩阵分解。"""

    def test_partition_is_valid(self, H_two_groups) -> None:
        """分区覆盖全部结点，且每块非空。"""
        partition = nmf_method(H_two_groups)
        assert _covered(partition) == set(H_two_groups.nodes)
        assert all(group for group in partition)

    def test_picks_best_modularity_among_candidates(self, H_two_groups) -> None:
        """返回的划分模块度不低于同一范围内其它候选社区数的最优值。"""
        from honet.community._hmod import modularity

        partition = nmf_method(H_two_groups)
        best_returned = modularity(H_two_groups, partition)
        # 只试一个候选数时，结果应与整体最优一致
        for n_components in (2, 3, 4, 5):
            single = nmf_method(
                H_two_groups, min_components=n_components, max_components=n_components + 1
            )
            assert modularity(H_two_groups, single) <= best_returned + 1e-12

    def test_is_reproducible(self, H_two_groups) -> None:
        """固定 random_state 后结果可复现。"""
        assert nmf_method(H_two_groups) == nmf_method(H_two_groups)

    def test_handles_small_graph(self, H_small) -> None:
        """4 结点图能跑出结果（旧实现从 3 起循环，此处从 2 起）。"""
        partition = nmf_method(H_small)
        assert partition is not None
        assert _covered(partition) == set(H_small.nodes)

    def test_rejects_too_few_nodes(self) -> None:
        """结点数不足以分成两块时返回 None。"""
        tiny = hnx.Hypergraph({"e1": ["a"]})
        assert nmf_method(tiny) is None


class TestRecursiveMethod:
    """递归二分。"""

    def test_partition_is_valid(self, H_two_groups) -> None:
        """分区覆盖全部结点，且每块非空。"""
        partition = recursive_method(H_two_groups)
        assert _covered(partition) == set(H_two_groups.nodes)
        assert all(group for group in partition)

    def test_never_worse_than_kumar_start(self, H_two_groups) -> None:
        """递归只在模块度严格提升时接受拆分，故模块度不低于 Kumar 初值。

        ``kumar`` 内部用 igraph 的 ``community_multilevel`` 且未固定随机种子，
        成员归属会在两次调用间抖动；这里只比较模块度——在该图上稳定为
        0.1265432，无论哪种归属都成立。
        """
        from honet.community._hmod import modularity

        assert modularity(H_two_groups, recursive_method(H_two_groups)) >= (
            0.1265432098765434 - 1e-12
        )

    def test_returns_sets(self, H_two_groups) -> None:
        """返回 list[set]，与 kumar 的输出类型一致。"""
        partition = recursive_method(H_two_groups)
        assert all(isinstance(group, set) for group in partition)

    def test_respects_max_depth(self, H_two_groups) -> None:
        """max_depth=0 时不再拆分，结果与 Kumar 初值的模块度相同。"""
        from honet.community._hmod import modularity

        # 两次调用都从各自的 Kumar 结果出发，只比对模块度而非成员
        assert modularity(H_two_groups, recursive_method(H_two_groups, max_depth=0)) == (
            pytest.approx(0.1265432098765434)
        )

    def test_works_on_minimal_graph(self, H_small) -> None:
        """极小图也能给出覆盖全部结点的分区。"""
        partition = recursive_method(H_small)
        assert _covered(partition) == set(H_small.nodes)


@needs_hgx
class TestHierarchyMethod:
    """超边层次聚类 + 多数票投影。"""

    def test_partition_is_valid(self, H_two_groups) -> None:
        """分区覆盖全部结点，且每块非空。"""
        partition = hierarchy_method(H_two_groups)
        assert _covered(partition) == set(H_two_groups.nodes)
        assert all(group for group in partition)

    def test_recovers_two_blocks(self, H_two_groups) -> None:
        """结构明显的两块应被识别出来。"""
        partition = hierarchy_method(H_two_groups)
        assert {frozenset(group) for group in partition} == {
            frozenset({"a", "b", "c", "d"}),
            frozenset({"e", "f", "g"}),
        }

    def test_larger_cut_height_merges_communities(self, H_chained) -> None:
        """切割距离阈值越大，社区数越少（单调性）。"""
        counts = [
            len(hierarchy_method(H_chained, cut_height=height)) for height in (0.5, 0.8, 0.95)
        ]
        assert counts == sorted(counts, reverse=True)
        assert counts[0] > counts[-1]

    def test_chained_graph_partition(self, H_chained) -> None:
        """链状图在默认阈值下切成三块。"""
        partition = hierarchy_method(H_chained)
        assert {frozenset(group) for group in partition} == {
            frozenset({"a", "b", "c"}),
            frozenset({"d"}),
            frozenset({"e", "f"}),
        }

    def test_is_deterministic(self, H_two_groups) -> None:
        """纯确定性流程，重复调用结果相同。"""
        assert hierarchy_method(H_two_groups) == hierarchy_method(H_two_groups)

    def test_single_edge_graph(self) -> None:
        """只有一条超边时整张图视为一个社区。"""
        single = hnx.Hypergraph({"e1": ["a", "b", "c"]})
        assert hierarchy_method(single) == [["a", "b", "c"]]

    def test_covers_isolated_node(self) -> None:
        """未被保留超边覆盖的结点单独成簇，不会丢失。"""
        graph = hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["a", "b", "d"]})
        partition = hierarchy_method(graph)
        assert _covered(partition) == set(graph.nodes)


class TestCheckPartition:
    """历史划分的覆盖性检查。"""

    def test_reports_missing_nodes(self, H_two_groups) -> None:
        """返回未被历史划分覆盖的结点，顺序与 nodes 一致。"""
        missing = check_partition(H_two_groups, {"a": 0, "b": 0})
        assert missing == ["c", "d", "e", "f", "g"]

    def test_complete_partition_reports_nothing(self, H_two_groups) -> None:
        """覆盖全部结点时返回空列表。"""
        prev = {node: 0 for node in H_two_groups.nodes}
        assert check_partition(H_two_groups, prev) == []

    def test_tolerates_stale_nodes(self, H_two_groups) -> None:
        """历史划分里多出来的旧结点不影响结果。"""
        prev = {node: 0 for node in H_two_groups.nodes}
        prev["已删除的旧结点"] = 1
        assert check_partition(H_two_groups, prev) == []

    def test_non_dict_returns_empty(self, H_two_groups) -> None:
        """非字典输入返回空列表（旧实现会打印警告，这里不产生输出）。"""
        assert check_partition(H_two_groups, [0, 1, 2]) == []


class TestExtractCommunityPartition:
    """统一入口：按方法分发并计算模块度。"""

    @pytest.mark.parametrize("method", list(Method))
    def test_every_method_returns_valid_result(self, H_two_groups, method) -> None:
        """每个方法都能返回 qh / communities / nodemap 三项。"""
        result = extract_community_partition(H_two_groups, method=method)
        assert set(result) == {"qh", "communities", "nodemap"}
        assert isinstance(result["qh"], float)
        assert _covered(result["communities"]) == set(H_two_groups.nodes)
        # nodemap 是 {结点: 社区下标}，覆盖全部结点
        assert set(result["nodemap"]) == set(H_two_groups.nodes)

    def test_only_qh_returns_float(self, H_two_groups) -> None:
        """only_qh=True 时只返回模块度。"""
        qh = extract_community_partition(H_two_groups, only_qh=True)
        assert isinstance(qh, float)
        # Kumar 的成员归属会因 igraph 未固定种子而抖动，但模块度稳定在该值
        assert qh == pytest.approx(0.1265432098765434)

    def test_default_method_is_kumar(self, H_two_groups) -> None:
        """默认方法是 Kumar，结果与显式指定一致。"""
        assert extract_community_partition(
            H_two_groups, only_qh=True
        ) == extract_community_partition(H_two_groups, method=Method.KUMAR, only_qh=True)

    def test_nodemap_matches_communities(self, H_two_groups) -> None:
        """nodemap 的社区下标与 communities 的顺序对应。"""
        result = extract_community_partition(H_two_groups)
        for index, group in enumerate(result["communities"]):
            for node in group:
                assert result["nodemap"][node] == index

    def test_basic_matches_direct_call(self, H_two_groups) -> None:
        """通过入口调用与直接调用得到相同模块度。"""
        from honet.community._hmod import modularity

        partition = basic_method(H_two_groups)
        assert extract_community_partition(
            H_two_groups, method=Method.Basic, only_qh=True
        ) == pytest.approx(modularity(H_two_groups, partition))

    def test_rejects_unknown_method(self, H_two_groups) -> None:
        """传入非 Method 值抛 ValueError。"""
        with pytest.raises(ValueError, match="未实现"):
            extract_community_partition(H_two_groups, method="kumar")


class TestCompareMethods:
    """同一张超图上的多方法比较。"""

    def test_returns_score_per_method(self, H_two_groups) -> None:
        """显式传入方法列表时，每个方法一项得分。"""
        scores = compare_methods(H_two_groups, [Method.KUMAR, Method.LAST_STEP])
        assert list(scores) == [Method.KUMAR, Method.LAST_STEP]
        assert all(isinstance(score, float) for score in scores.values())

    def test_default_compares_all_methods(self, H_two_groups) -> None:
        """不传 methods 时比较全部成员（旧实现的默认值是 GenericAlias，会抛错）。"""
        scores = compare_methods(H_two_groups)
        assert list(scores) == list(Method)

    def test_default_metric_is_modularity(self, H_two_groups) -> None:
        """默认得分等于统一入口算出的模块度。"""
        scores = compare_methods(H_two_groups, [Method.KUMAR])
        assert scores[Method.KUMAR] == pytest.approx(
            extract_community_partition(H_two_groups, method=Method.KUMAR, only_qh=True)
        )

    def test_custom_metric(self, H_two_groups) -> None:
        """自定义评价函数生效。"""
        scores = compare_methods(H_two_groups, [Method.KUMAR], compare_fun=lambda graph, method: 7)
        assert scores == {Method.KUMAR: 7}

    def test_failure_is_isolated(self, H_two_groups) -> None:
        """某个方法抛异常时记为 None，不影响其他方法。"""

        def boom(graph, method):
            if method is Method.Basic:
                msg = "故意失败"
                raise RuntimeError(msg)
            return 1.0

        scores = compare_methods(H_two_groups, [Method.KUMAR, Method.Basic], compare_fun=boom)
        assert scores[Method.KUMAR] == 1.0
        assert scores[Method.Basic] is None


class TestCompareGraphs:
    """多张超图的同一方法比较。"""

    def test_accepts_sequence(self, H_two_groups, H_chained) -> None:
        """传序列时以下标为标签。"""
        result = compare_graphs([H_two_groups, H_chained])
        assert list(result) == [0, 1]
        assert all(set(value) == {"qh", "communities", "nodemap"} for value in result.values())

    def test_accepts_dict(self, H_two_groups, H_chained) -> None:
        """传字典时用字典的键作标签（旧实现用不可哈希的超图对象作键会抛错）。"""
        result = compare_graphs({"两组": H_two_groups, "链状": H_chained})
        assert list(result) == ["两组", "链状"]

    def test_scores_match_direct_call(self, H_two_groups, H_chained) -> None:
        """每张图的模块度与单独调用一致。

        这里用 ``Method.Basic``：它固定了 Louvain 的随机种子，结果完全确定。
        ``Method.KUMAR`` 内部依赖 igraph 的 ``community_multilevel`` 且未固定
        随机种子，同一张图两次调用的成员归属可能不同（模块度相同），不适合做
        逐值比对。
        """
        result = compare_graphs({"两组": H_two_groups, "链状": H_chained}, method=Method.Basic)
        for label, graph in (("两组", H_two_groups), ("链状", H_chained)):
            assert result[label]["qh"] == pytest.approx(
                extract_community_partition(graph, method=Method.Basic, only_qh=True)
            )

    def test_respects_method_argument(self, H_two_groups, H_chained) -> None:
        """method 参数生效，可换用其他方法。"""
        result = compare_graphs({"A": H_two_groups}, method=Method.Basic)
        assert result["A"]["qh"] == pytest.approx(
            extract_community_partition(H_two_groups, method=Method.Basic, only_qh=True)
        )


class TestPackageLaziness:
    """子包的惰性导出约束。"""

    def test_all_exports_are_importable(self) -> None:
        """__all__ 里的每个名字都能真正解析出来。"""
        import honet.community as community

        for name in community.__all__:
            assert getattr(community, name) is not None

    def test_import_honet_loads_no_third_party(self) -> None:
        """import honet 不拉入任何第三方依赖。"""
        import subprocess
        import sys

        code = (
            "import sys, honet; "
            "bad=[m for m in ('numpy','networkx','hypernetx','sklearn','hypergraphx') "
            "if m in sys.modules]; "
            "assert not bad, bad"
        )
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr

    def test_submodule_shadowing_is_handled(self) -> None:
        """直接 import 实现子模块后，包属性仍是可调用对象。"""
        import importlib

        import honet.community as community

        importlib.import_module("honet.community.random_method")
        from honet.community import random_method as resolved

        assert callable(resolved)
        assert callable(community.random_method)
