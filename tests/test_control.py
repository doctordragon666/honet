"""honet.control 子包的单元测试。

覆盖点：

* 星形图 / 关联矩阵 / 随机超图等结构构造；
* dilation 检测（匹配口径）与可达性传播；
* 暴力最小驱动集、贪心补点、按规模自动选算法的驱动集合计算；
* hypernetx → HAT 的驱动计算与批量 ``calc_drivers``；
* 惰性导出：``import honet.control`` 不会拉入 HAT / networkx / pandas / numpy，
  以及公开符号在导入同名实现子模块后仍是函数（不被模块遮蔽）。

全部用例使用 4~6 个结点的小规模真实超图，实际调用每个公开函数，不做 mock。
运行方式：``uv run pytest tests/test_control.py -q``。
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import pytest

from honet.control import (
    add_node_fields,
    brute_force_min_driver_sets,
    calc_drivers,
    compute_and_export_drivers,
    compute_drivers_from_hypernetx,
    detect_dilation_and_drivers,
    detect_dilation_with_given_drivers,
    edge_list_to_incidence,
    greedy_accessibility,
    hypergraph_reachable,
    random_uniform_hypergraph,
    star_graph,
)

# 整个子包依赖 HAT（honet[hat]），缺依赖时整文件跳过。
HAT = pytest.importorskip("HAT", reason="需要 HypergraphAnalysisToolbox（honet[hat]）")
hnx = pytest.importorskip("hypernetx", reason="需要 hypernetx（honet[analysis]）")
np = pytest.importorskip("numpy", reason="需要 numpy（honet[analysis]）")
pd = pytest.importorskip("pandas", reason="需要 pandas（honet[analysis]）")


def make_directed_hypergraph() -> HAT.Hypergraph:
    """4 结点 2 超边的有向超图：e0: {1} → {0}，e1: {2} → {1, 3}。

    关联矩阵用 ±1 表达方向（head 为 +1、tail 为 -1），HAT 据此判定为有向超图。
    传播链是 2 → 1 → 0，因此结点 2 是唯一的「根」：从它出发可到达全部结点，
    从 0 或 3 出发则只能到达自己。
    """
    incidence = np.zeros((4, 2), dtype=int)
    incidence[0, 0] = 1  # 0 是 e0 的头结点
    incidence[1, 0] = -1  # 1 是 e0 的尾结点
    incidence[1, 1] = 1  # 1 是 e1 的头结点
    incidence[2, 1] = -1  # 2 是 e1 的尾结点
    incidence[3, 1] = 1  # 3 是 e1 的头结点
    nodes = pd.DataFrame({"Nodes": np.arange(4)})
    return HAT.Hypergraph(incidence_matrix=incidence, directed=True, nodes=nodes)


def make_undirected_hypergraph() -> HAT.Hypergraph:
    """3 结点 2 超边的无向超图：e0={0,1}、e1={1,2}。

    必须显式给出结点表：HAT 只用 ``edges=`` 构造时会因 ``num_nodes`` 未定义而抛
    ``UnboundLocalError``（库自身缺陷，不是本子包的行为）。
    """
    edges = pd.DataFrame({"Nodes": [[0, 1], [1, 2]]})
    nodes = pd.DataFrame({"Nodes": np.arange(3)})
    return HAT.Hypergraph(edges=edges, nodes=nodes)


def make_edge_list() -> list[dict[str, list[int]]]:
    """5 结点 4 超边的 head/tail 边列表，覆盖全部结点 0..4。

    结构：{0,1}←{2,3}、{1}←{2,4}、{0,2}←{1}、{3,4}←{0}，
    相互耦合，没有任何结点能单独触发全部超边。
    """
    return [
        {"head": [0, 1], "tail": [2, 3]},
        {"head": [1], "tail": [2, 4]},
        {"head": [0, 2], "tail": [1]},
        {"head": [3, 4], "tail": [0]},
    ]


@pytest.fixture()
def HG() -> HAT.Hypergraph:
    """4 结点链式有向超图（见 :func:`make_directed_hypergraph`）。"""
    return make_directed_hypergraph()


class TestStarGraph:
    """星形图（二分图）展开。"""

    def test_bipartite_nodes(self, HG: HAT.Hypergraph) -> None:
        """系统结点与超边结点各占一侧，名字分别是 n<i> 与 e<j>。"""
        G = star_graph(HG)
        assert sorted(G.nodes) == ["e0", "e1", "n0", "n1", "n2", "n3"]
        assert G.nodes["n0"]["bipartite"] == 0
        assert G.nodes["e0"]["bipartite"] == 1

    def test_directed_edges_orientation(self, HG: HAT.Hypergraph) -> None:
        """有向超边按「尾结点 → 边结点 → 头结点」连边。"""
        G = star_graph(HG)
        assert G.has_edge("n1", "e0")  # 1 是 e0 的尾结点
        assert G.has_edge("e0", "n0")  # 0 是 e0 的头结点
        assert not G.has_edge("e0", "n1")  # 尾结点不会被超边指向
        assert not G.has_edge("n0", "e0")

    def test_undirected_hypergraph(self) -> None:
        """无向超图返回无向图，边结点与全部成员相连。"""
        G = star_graph(make_undirected_hypergraph())
        assert not G.is_directed()
        assert G.has_edge("e0", "n0")
        assert G.has_edge("e0", "n1")
        assert not G.has_edge("e1", "n0")  # 结点 0 不属于 e1


class TestAddNodeFields:
    """结点表的 edges / head / tail 三列。"""

    def test_edge_lists(self, HG: HAT.Hypergraph) -> None:
        """edges 列记录所属超边行号，head/tail 按角色分开。"""
        result = add_node_fields(HG)
        assert result is HG  # 原地修改
        assert HG.nodes["edges"].tolist() == [[0], [0, 1], [1], [1]]
        assert HG.nodes["head"].tolist() == [[0], [1], [], [1]]
        assert HG.nodes["tail"].tolist() == [[], [0], [1], []]

    def test_undirected_members_in_both(self) -> None:
        """无向超图没有方向，成员同时计入 head 与 tail。"""
        HG = make_undirected_hypergraph()
        add_node_fields(HG)
        assert HG.nodes["head"].tolist() == [[0], [0, 1], [1]]
        assert HG.nodes["tail"].tolist() == [[0], [0, 1], [1]]


class TestEdgeListToIncidence:
    """head/tail 边列表 → 关联矩阵。"""

    def test_shape_and_signs(self) -> None:
        """形状为 (结点数, 超边数)，head 为 +1、tail 为 -1。"""
        incidence = edge_list_to_incidence(make_edge_list())
        assert incidence.shape == (5, 4)
        assert incidence.dtype == np.dtype(int)
        # 第一条边：head=[0,1]、tail=[2,3]，结点 4 不参与
        assert incidence[:, 0].tolist() == [1, 1, -1, -1, 0]

    def test_compressed_node_index(self) -> None:
        """行号按「出现过的结点排序」重新编号，而非直接用结点编号。"""
        incidence = edge_list_to_incidence([{"head": [3], "tail": [2]}])
        assert incidence.tolist() == [[-1], [1]]


class TestRandomUniformHypergraph:
    """随机 k-均匀有向超图。"""

    def test_structure(self) -> None:
        """每条边 head 非空、head 与 tail 不相交、tail 大小为 k。"""
        random.seed(0)
        edges = random_uniform_hypergraph(6, 3, 5)
        assert len(edges) == 5
        for edge in edges:
            assert len(edge["tail"]) == 3
            assert 1 <= len(edge["head"]) <= 2
            assert set(edge["head"]).isdisjoint(edge["tail"])
            assert all(0 <= node < 6 for node in edge["head"] + edge["tail"])

    def test_no_duplicate_edges(self) -> None:
        """同一条 (head, tail) 不会重复出现。"""
        random.seed(1)
        edges = random_uniform_hypergraph(6, 3, 6)
        keys = {(tuple(e["head"]), tuple(e["tail"])) for e in edges}
        assert len(keys) == len(edges)

    def test_seed_reproducible(self) -> None:
        """同一种子给出同一组超边。"""
        random.seed(42)
        first = random_uniform_hypergraph(6, 3, 4)
        random.seed(42)
        assert random_uniform_hypergraph(6, 3, 4) == first

    def test_invalid_k(self) -> None:
        """k 不满足 1 <= k < n 时抛 ValueError（旧实现会死循环）。"""
        with pytest.raises(ValueError):
            random_uniform_hypergraph(4, 4, 2)


class TestHypergraphReachable:
    """可达性传播。"""

    def test_chain(self, HG: HAT.Hypergraph) -> None:
        """从根结点 {2} 出发可以到达全部结点（2 → 1 → 0，且 1 触发 e1 带出 3）。"""
        assert hypergraph_reachable(HG, {2}) == {0, 1, 2, 3}

    def test_isolated_start(self, HG: HAT.Hypergraph) -> None:
        """从 0 出发时没有任何超边的尾结点集合是 {0} 的子集，传播不出去。"""
        assert hypergraph_reachable(HG, {0}) == {0}

    def test_partial_start(self, HG: HAT.Hypergraph) -> None:
        """从 3 出发只能到达自己（3 只是 e1 的头结点，不触发任何边）。"""
        assert hypergraph_reachable(HG, {3}) == {3}

    def test_start_included(self, HG: HAT.Hypergraph) -> None:
        """返回集合包含起始结点本身。"""
        assert 2 in hypergraph_reachable(HG, {2})

    def test_empty_start(self, HG: HAT.Hypergraph) -> None:
        """空起点可达集合为空（没有任何超边的尾结点集合是空集的子集之外的误判）。"""
        assert hypergraph_reachable(HG, set()) == set()


class TestDetectDilation:
    """dilation 检测。"""

    def test_matching_drivers(self, HG: HAT.Hypergraph) -> None:
        """匹配只覆盖头结点 {0, 1}（3 与 0 争夺 e1/e0 的匹配位），驱动结点为剩余结点。"""
        dilation_exists, matched_heads, driver_nodes, matching = detect_dilation_and_drivers(
            HG, HG.nnodes
        )
        assert matched_heads == {0, 1}
        assert driver_nodes == {2, 3}
        assert dilation_exists is True
        assert isinstance(matching, dict)

    def test_no_dilation_after_drivers(self, HG: HAT.Hypergraph) -> None:
        """补上 {2, 3} 作为驱动结点后 dilation 消失。"""
        assert detect_dilation_with_given_drivers(HG, {2, 3}) is False

    def test_dilation_remains(self, HG: HAT.Hypergraph) -> None:
        """只给 {0} 时结点 2、3 仍无匹配对象，dilation 依然存在。"""
        assert detect_dilation_with_given_drivers(HG, {0}) is True

    def test_full_driver_set_always_ok(self, HG: HAT.Hypergraph) -> None:
        """全部结点都作为驱动结点时必然无 dilation。"""
        assert detect_dilation_with_given_drivers(HG, set(range(HG.nnodes))) is False


class TestBruteForceMinDriverSets:
    """暴力最小驱动集合。"""

    def test_minimal_sets(self, HG: HAT.Hypergraph) -> None:
        """最小规模为 2，且 {1, 2} 与 {2, 3} 都可行。"""
        size, driver_sets = brute_force_min_driver_sets(HG)
        assert size == 2
        assert driver_sets == [{1, 2}, {2, 3}]

    def test_verbose_runs(self, HG: HAT.Hypergraph, capsys: pytest.CaptureFixture[str]) -> None:
        """verbose=True 会打印搜索过程。"""
        brute_force_min_driver_sets(HG, verbose=True)
        assert "最小驱动规模" in capsys.readouterr().out

    def test_no_feasible_set(self) -> None:
        """一条超边都没有时可达性无法满足，返回 (None, [])。"""
        HG = HAT.Hypergraph(
            nodes=pd.DataFrame({"Nodes": np.arange(3)}),
            edges=pd.DataFrame({"Nodes": [], "Edges": []}),
        )
        size, driver_sets = brute_force_min_driver_sets(HG)
        assert (size, driver_sets) == (None, [])


class TestGreedyAccessibility:
    """贪心补齐驱动结点。"""

    def test_from_empty(self, HG: HAT.Hypergraph) -> None:
        """从空集开始，贪心只需加入 2 即可覆盖全系统。"""
        assert greedy_accessibility(HG) == {2}

    def test_with_initial_drivers(self, HG: HAT.Hypergraph) -> None:
        """初始驱动结点被保留在结果中。"""
        drivers = greedy_accessibility(HG, {0})
        assert 0 in drivers
        assert hypergraph_reachable(HG, drivers) == {0, 1, 2, 3}

    def test_without_node_fields(self, HG: HAT.Hypergraph) -> None:
        """未调用 add_node_fields 也能工作（旧实现会抛 KeyError）。"""
        assert "edges" not in HG.nodes.columns
        assert greedy_accessibility(HG) == {2}

    def test_already_covered(self, HG: HAT.Hypergraph) -> None:
        """初始驱动结点已覆盖全系统时不再新增结点。"""
        assert greedy_accessibility(HG, {0, 2}) == {0, 2}

    def test_seeding_is_one_hop(self, HG: HAT.Hypergraph) -> None:
        """贪心的初始播种只展开一跳，比 ``hypergraph_reachable`` 保守。

        从 {2} 出发：e1 被点亮并带出 1、3，但播种循环只遍历初始驱动结点，不会
        再用 1 去点亮 e0，因此 0 仍不可达，贪心会补上 0。旧实现即此语义
        （``for node in list(accessible)`` 遍历的是快照），这里保持不变。
        """
        assert hypergraph_reachable(HG, {2}) == {0, 1, 2, 3}
        assert greedy_accessibility(HG, {2}) == {0, 2}

    def test_gap_filled_by_greedy(self, HG: HAT.Hypergraph) -> None:
        """初始驱动结点覆盖不全时，贪心会补上能让传播继续的结点。"""
        drivers = greedy_accessibility(HG, {3})
        assert 3 in drivers
        assert len(drivers) > 1
        assert hypergraph_reachable(HG, drivers) == {0, 1, 2, 3}


class TestComputeAndExportDrivers:
    """按规模自动选算法的驱动集合计算。"""

    def test_small_graph_uses_brute_force(self, tmp_path: Path) -> None:
        """5 结点（<= 8）走暴力分支，结果与暴力最小集合一致。"""
        out = tmp_path / "drivers.csv"
        res = compute_and_export_drivers(make_edge_list(), out_csv=str(out))
        assert res["algorithm"] == "brute_force"
        assert res["dilation_exists"] is False
        # 5 结点里能覆盖全系统的最小驱动集规模为 2
        assert len(res["drivers"]) == 2
        assert out.exists()
        saved = pd.read_csv(out)
        assert saved["driver_node"].tolist() == res["drivers"]
        assert set(saved["algorithm"]) == {"brute_force"}

    def test_random_generation(self, tmp_path: Path) -> None:
        """edge_list=None 时按 n/k/m 随机生成超图，仍能算出驱动集合。"""
        random.seed(3)
        res = compute_and_export_drivers(n=6, k=3, m=5, out_csv=None)
        assert res["algorithm"] == "brute_force"
        assert 0 < len(res["drivers"]) <= 6
        assert res["drivers"] == sorted(res["drivers"])

    def test_no_csv(self) -> None:
        """out_csv=None 时不写文件，只返回结果字典。"""
        res = compute_and_export_drivers(make_edge_list(), out_csv=None)
        assert set(res) == {"algorithm", "dilation_exists", "drivers"}


class TestComputeDriversFromHypernetx:
    """hypernetx → HAT 的驱动计算。"""

    def test_small_hypergraph(self, tmp_path: Path) -> None:
        """4 结点 2 超边的小超图，转换后走暴力分支并导出 CSV。"""
        H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        out = tmp_path / "hnx.csv"
        res = compute_drivers_from_hypernetx(H, out_csv=str(out))
        assert res["algorithm"] == "brute_force"
        assert res["dilation_exists"] is False
        # 每条超边的 head == tail，驱动集合必须覆盖全部结点才能自触发
        assert res["drivers"] == [0, 1, 2]
        assert out.exists()

    def test_hat_type_passthrough(self) -> None:
        """HAT_type=True 时直接使用传入的 HAT 超图（旧实现会 UnboundLocalError）。"""
        HG = add_node_fields(make_directed_hypergraph())
        res = compute_drivers_from_hypernetx(HG, out_csv=None, HAT_type=True)
        assert res["algorithm"] == "brute_force"
        # 与同一超图上的暴力最小集合一致：{1, 2} 或 {2, 3}，算法取首个
        assert res["drivers"] == [1, 2]

    def test_node_names_sorted(self) -> None:
        """结点按字典序重新编号，drivers 下标对应 Names 列的位置。"""
        H = hnx.Hypergraph({"e1": ["d", "a"], "e2": ["a", "c"]})
        res = compute_drivers_from_hypernetx(H, out_csv=None)
        # 转换时 Names 排序为 [a, c, d]，驱动集合下标需落在 0..2
        assert all(0 <= driver < 3 for driver in res["drivers"])


class TestCalcDrivers:
    """批量驱动结点计算。"""

    def test_dataframe_output(self) -> None:
        """返回三列 DataFrame，drivers 是原始结点名，partition 为比例。"""
        hgs = {
            "2001": hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]}),
            "2002": hnx.Hypergraph({"e1": ["x", "y"]}),
        }
        df = calc_drivers(hgs)
        assert df.columns.tolist() == ["year", "drivers", "partition"]
        assert df["year"].tolist() == ["2001", "2002"]
        # 结点名必须来自原超图，不能是下标
        assert df.loc[0, "drivers"] == "a,b,c"
        assert 0 < df.loc[0, "partition"] <= 1.0

    def test_dict_output(self) -> None:
        """to_df=False 时返回 {键: [原始结点名]}。"""
        hgs = {"2001": hnx.Hypergraph({"e1": ["a", "b"]})}
        result = calc_drivers(hgs, to_df=False)
        assert set(result) == {"2001"}
        assert set(result["2001"]) <= {"a", "b"}

    def test_partition_uses_total_nodes(self) -> None:
        """partition 的分母是超图结点总数。"""
        hgs = {"k": hnx.Hypergraph({"e1": ["a", "b"], "e2": ["c", "d"]})}
        df = calc_drivers(hgs)
        drivers = df.loc[0, "drivers"].split(",")
        assert df.loc[0, "partition"] == pytest.approx(len(drivers) / 4)


class TestLazyExports:
    """惰性导出与导入顺序的稳定性。"""

    def test_import_does_not_pull_hat(self) -> None:
        """``import honet.control`` 不得拉入 HAT / networkx / pandas / numpy。"""
        code = (
            "import sys, honet.control;"
            "print(sorted(m for m in ('HAT', 'networkx', 'pandas', 'numpy') if m in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "[]"

    def test_public_names_are_functions(self) -> None:
        """``__all__`` 里每个名字都是可调用函数，而不是被同名模块遮蔽。"""
        import honet.control as control

        for name in control.__all__:
            assert callable(getattr(control, name)), name

    def test_survives_submodule_import(self) -> None:
        """显式导入同名实现子模块后，包属性仍是函数（惰性导出不被遮蔽）。"""
        import importlib

        import honet.control as control

        importlib.import_module("honet.control.star_graph")
        importlib.import_module("honet.control.calc_drivers")
        assert callable(control.star_graph)
        assert callable(control.calc_drivers)
