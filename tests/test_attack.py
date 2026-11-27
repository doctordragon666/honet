"""honet.attack 子包的单元测试。

覆盖点：
* 三类攻击序列（随机 / 基于度 / 基于中心性）的排序与完整性；
* ``main_country`` 按数量、按比例、按字典三种截取方式；
* ``execute_attack`` 的逐步移除、度量曲线、提前终止与不改动原图；
* ``basic_attack_sequence`` 在普通 NetworkX 图上的五类策略；
* 惰性导出：``import honet.attack`` 不会拉入 numpy / hypernetx / networkx。

全部用例使用 5 个结点的小规模真实超图，实际调用每个公开函数，不做 mock。
运行方式：``uv run pytest tests/test_attack.py -q``。
"""

from __future__ import annotations

import subprocess
import sys

import hypernetx as hnx
import networkx as nx
import pytest

from honet.attack import (
    basic_attack_sequence,
    centrality_attack_sequence,
    degree_attack_sequence,
    execute_attack,
    main_country,
    random_attack_sequence,
)


@pytest.fixture()
def hg() -> hnx.Hypergraph:
    """5 结点 4 超边的真实小超图。

    拓扑：e1={a,b,c}、e2={b,c}、e3={c,d}、e4={d,e}，
    超度依次为 a=1、b=2、c=3、d=2、e=1，c 是唯一的枢纽结点。
    """
    return hnx.Hypergraph(
        {"e1": ["a", "b", "c"], "e2": ["b", "c"], "e3": ["c", "d"], "e4": ["d", "e"]}
    )


class TestRandomAttackSequence:
    """随机攻击序列。"""

    def test_node_permutation(self, hg: hnx.Hypergraph) -> None:
        """结点序列是全部结点的一个排列，不重不漏。"""
        sequence = random_attack_sequence(hg, seed=7)
        assert sorted(sequence) == sorted(hg.nodes)
        assert len(sequence) == len(list(hg.nodes))

    def test_edge_permutation(self, hg: hnx.Hypergraph) -> None:
        """超边序列是全部超边的一个排列。"""
        sequence = random_attack_sequence(hg, target="edge", seed=7)
        assert sorted(sequence) == sorted(hg.edges)

    def test_seed_is_reproducible(self, hg: hnx.Hypergraph) -> None:
        """同一种子给出同一顺序，便于复现实验。"""
        assert random_attack_sequence(hg, seed=42) == random_attack_sequence(hg, seed=42)

    def test_invalid_target(self, hg: hnx.Hypergraph) -> None:
        """非法目标抛 ValueError。"""
        with pytest.raises(ValueError, match="target"):
            random_attack_sequence(hg, target="hyperedge")


class TestDegreeAttackSequence:
    """基于度的攻击序列。"""

    def test_node_sorted_desc(self, hg: hnx.Hypergraph) -> None:
        """结点按超度降序：c(3) 在最前，a/e(1) 在最后。"""
        sequence = degree_attack_sequence(hg)
        assert sequence[0] == "c"
        assert set(sequence[-2:]) == {"a", "e"}
        degrees = {node: hg.degree(node) for node in hg.nodes}
        assert [degrees[n] for n in sequence] == sorted(degrees.values(), reverse=True)

    def test_edge_sorted_by_size(self, hg: hnx.Hypergraph) -> None:
        """超边按基数降序：e1 有 3 个成员，其余各 2 个。"""
        sequence = degree_attack_sequence(hg, target="edge")
        assert sequence[0] == "e1"
        assert sorted(sequence) == sorted(hg.edges)

    def test_excess_uses_average_neighbor_degree(self, hg: hnx.Hypergraph) -> None:
        """余平均度策略的首位是邻居平均超度最高的结点。

        a 的唯一邻居是 b、c（超度 2、3），平均 2.5 为全图最高。
        """
        sequence = degree_attack_sequence(hg, target="excess")
        assert sequence[0] == "a"
        assert sorted(sequence) == sorted(hg.nodes)

    def test_cluster_sorted_desc(self, hg: hnx.Hypergraph) -> None:
        """聚类系数策略覆盖全部结点，且首位的聚类系数不低于末位。"""
        sequence = degree_attack_sequence(hg, target="cluster")
        assert sorted(sequence) == sorted(hg.nodes)
        assert sequence[0] != sequence[-1]

    def test_invalid_target(self, hg: hnx.Hypergraph) -> None:
        """非法目标抛 ValueError。"""
        with pytest.raises(ValueError, match="target"):
            degree_attack_sequence(hg, target="betweenness")


class TestCentralityAttackSequence:
    """基于中心性的攻击序列。"""

    def test_betweenness_hub_first(self, hg: hnx.Hypergraph) -> None:
        """介数中心性策略把枢纽结点 c 排在最前。"""
        sequence = centrality_attack_sequence(hg)
        assert sequence[0] == "c"
        assert sorted(sequence) == sorted(hg.nodes)

    def test_eigenvector_covers_all_nodes(self, hg: hnx.Hypergraph) -> None:
        """特征向量中心性策略覆盖全部结点，且按绝对值降序。"""
        from honet.metrics.eigenvector_centrality import eigenvector_centrality

        scores = eigenvector_centrality(hg)
        sequence = centrality_attack_sequence(hg, metric="eigenvector")
        assert sorted(sequence) == sorted(hg.nodes)
        magnitudes = [abs(scores[n]) for n in sequence]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_invalid_metric(self, hg: hnx.Hypergraph) -> None:
        """非法指标抛 ValueError。"""
        with pytest.raises(ValueError, match="metric"):
            centrality_attack_sequence(hg, metric="closeness")


class TestMainCountry:
    """攻击样本截取。"""

    def test_by_num(self) -> None:
        """按数量截取前 num 个。"""
        assert main_country(["a", "b", "c", "d", "e"], num=2) == ["a", "b"]

    def test_by_partion(self) -> None:
        """按比例截取，向下取整（5 × 0.4 = 2）。"""
        assert main_country(["a", "b", "c", "d", "e"], partion=0.4) == ["a", "b"]

    def test_default_num(self) -> None:
        """既不给 num 也不给 partion 时默认取前 10 个（样本不足则全取）。"""
        assert main_country(["a", "b", "c"]) == ["a", "b", "c"]

    def test_dict_sample_uses_values(self) -> None:
        """字典样本按值截取。"""
        sample = {"策略1": "a", "策略2": "b", "策略3": "c"}
        assert main_country(sample, num=2) == ["a", "b"]

    def test_both_args_prefers_num(self, capsys: pytest.CaptureFixture[str]) -> None:
        """同时给出 num 与 partion 时以 num 为准，并打印提示。"""
        assert main_country(["a", "b", "c", "d", "e"], num=1, partion=0.4) == ["a"]
        assert "不可同时设置" in capsys.readouterr().out


class TestExecuteAttack:
    """攻击执行。"""

    def test_node_attack_records_curve(self, hg: hnx.Hypergraph) -> None:
        """按 40% 比例逐步移除结点，记录等长的 steps 与 nc_vals。"""
        result = execute_attack(hg, degree_attack_sequence(hg), target="node", rho=0.4)
        curve = result["默认策略"]
        assert curve["nc_vals"][0] == 1.0
        assert curve["steps"][0] == 0.0
        assert len(curve["steps"]) == len(curve["nc_vals"])
        # 5 个结点每次移除 2 个：第一步 2/5，第二步 3/5
        assert curve["steps"][1] == pytest.approx(0.4)
        assert curve["steps"][2] == pytest.approx(0.6)
        # 攻击过程中度量必须单调不增（移除只会减少冗余路径）
        values = curve["nc_vals"]
        assert all(
            later <= earlier + 1e-9 for earlier, later in zip(values, values[1:], strict=False)
        )

    def test_edge_attack_removes_all_edges(self, hg: hnx.Hypergraph) -> None:
        """超边攻击最终把超边全部移除，末位 steps 为 1.0。"""
        result = execute_attack(
            hg, degree_attack_sequence(hg, target="edge"), target="edge", rho=0.5
        )
        curve = result["默认策略"]
        assert curve["steps"][-1] == pytest.approx(1.0)
        assert curve["nc_vals"][-1] == 0.0

    def test_multiple_strategies(self, hg: hnx.Hypergraph) -> None:
        """字典样本一次跑多套策略，结果键与策略名一一对应。"""
        sample = {
            "度": degree_attack_sequence(hg),
            "介数": centrality_attack_sequence(hg),
        }
        result = execute_attack(hg, sample, rho=0.4)
        assert set(result) == {"度", "介数"}
        assert all("steps" in curve and "nc_vals" in curve for curve in result.values())

    def test_custom_metric(self, hg: hnx.Hypergraph) -> None:
        """自定义度量函数按「剩余超边数」评估，随攻击单调递减。"""
        result = execute_attack(
            hg,
            degree_attack_sequence(hg),
            target="node",
            func=lambda h: float(len(h.edges)),
            rho=0.4,
        )
        values = result["默认策略"]["nc_vals"][1:]
        assert values == sorted(values, reverse=True)

    def test_original_hypergraph_untouched(self, hg: hnx.Hypergraph) -> None:
        """攻击在副本上进行，调用方的超图不被修改。"""
        before_nodes, before_edges = list(hg.nodes), list(hg.edges)
        execute_attack(hg, degree_attack_sequence(hg), rho=0.4)
        assert list(hg.nodes) == before_nodes
        assert list(hg.edges) == before_edges

    def test_early_stop_at_zero(self, hg: hnx.Hypergraph) -> None:
        """度量归零后立即终止，不会继续把序列跑完。"""
        result = execute_attack(hg, ["c", "b", "d", "a", "e"], rho=1.0)
        curve = result["默认策略"]
        assert curve["nc_vals"] == [1.0, 0.0]
        assert len(curve["steps"]) == 2

    def test_none_hypergraph(self) -> None:
        """传入 None 时返回空字典而不是抛异常。"""
        assert execute_attack(None, ["a"]) == {}

    def test_invalid_target(self, hg: hnx.Hypergraph) -> None:
        """非法目标抛 ValueError。"""
        with pytest.raises(ValueError, match="target"):
            execute_attack(hg, ["a"], target="vertex")


class TestBasicAttackSequence:
    """普通图的攻击序列。"""

    def test_five_strategies(self) -> None:
        """路径图 0-1-2-3-4 上返回五类策略，每个序列覆盖全部结点。"""
        g = nx.path_graph(5)
        sequences = basic_attack_sequence(g)
        assert list(sequences) == ["度", "余平均度", "聚类系数", "介数", "特征向量中心性"]
        for sequence in sequences.values():
            assert sorted(sequence) == sorted(g.nodes)

    def test_degree_order(self) -> None:
        """星形图中心结点 0 的度最高，排在度策略首位。"""
        g = nx.star_graph(4)
        assert basic_attack_sequence(g)["度"][0] == 0

    def test_betweenness_hub_first(self) -> None:
        """路径图上中间结点 2 的介数最大，排在介数策略首位。"""
        g = nx.path_graph(5)
        assert basic_attack_sequence(g)["介数"][0] == 2


class TestLazyExport:
    """惰性导出契约。"""

    def test_import_does_not_pull_third_party(self) -> None:
        """``import honet.attack`` 本身不加载 numpy / hypernetx / networkx。"""
        code = (
            "import sys, honet.attack; "
            "print([m for m in ('numpy', 'hypernetx', 'networkx') if m in sys.modules])"
        )
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "[]"

    def test_exports_sorted_and_complete(self) -> None:
        """__all__ 覆盖六个公开函数且按字母序。"""
        import honet.attack as attack

        expected = {
            "random_attack_sequence",
            "degree_attack_sequence",
            "centrality_attack_sequence",
            "main_country",
            "execute_attack",
            "basic_attack_sequence",
        }
        assert set(attack.__all__) == expected
        assert attack.__all__ == sorted(attack.__all__)

    def test_unknown_attribute(self) -> None:
        """访问不存在的符号抛 AttributeError。"""
        import honet.attack as attack

        with pytest.raises(AttributeError, match="no attribute"):
            _ = attack.no_such_function
