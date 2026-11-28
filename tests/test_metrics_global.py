"""``honet.metrics`` 全局指标函数的单元测试。

覆盖点：
* ``calc_entropy``：度分布香农熵，与手工算出的熵一致；退化输入取 0；
* ``calc_information_gain``：投影熵与高阶熵之差，两侧熵可手工复算；2 元边网络
  与星形超图的增益为 0、投影更集中时为负；``detail=True`` 返回两侧熵；
* ``calc_clustering_coefficient_g``：等于局部聚类系数的算术平均；
* ``calc_natural_connectivity``：等于团展开邻接矩阵特征值的指数平均对数；
* ``assortativity_test``：按超度分组的平均余平均度，数值手工可验算；
* ``calc_assortativity``：超度最大/第二大结点的秩相关 + 配置模型双侧 p 值；
* ``sub_reducibility`` / ``calc_reducibility``：加权修正版与官方无权版的关系、
  参数校验、输入类型校验；
* ``hnxs_global``：多超图汇总成 DataFrame 的形状与取值；
* 惰性导出契约：``import honet.metrics`` 不拉第三方依赖，且访问后包属性是
  函数而不是被 import 机制写回的模块对象。

数据规模：5 个结点 / 4 条超边的真实小超图（``a,b,c,d,e``，超度 1,2,3,2,1），
全部用例实际调用公开函数，不做 mock。运行方式：``uv run pytest
tests/test_metrics_global.py -q``。
"""

from __future__ import annotations

import math
import subprocess
import sys

import hypergraphx as hgx
import hypernetx as hnx
import numpy as np
import pytest

from honet.convert import convert_hypernetx_to_hypergraphx
from honet.metrics import (
    assortativity_test,
    calc_assortativity,
    calc_clustering_coefficient_g,
    calc_entropy,
    calc_information_gain,
    calc_natural_connectivity,
    calc_reducibility,
    hnxs_global,
    sub_reducibility,
    temporal_closeness_centrality,
)
from honet.temporal import TemporalNetwork, TemporalNetworkW

#: 全局指标子包新增的公开函数（不含已有六个文件里的函数）。
_NEW_EXPORTS = {
    "assortativity_test",
    "calc_assortativity",
    "calc_clustering_coefficient_g",
    "calc_entropy",
    "calc_information_gain",
    "calc_natural_connectivity",
    "calc_reducibility",
    "hnxs_global",
    "sub_reducibility",
    "temporal_closeness_centrality",
}


@pytest.fixture()
def hg() -> hnx.Hypergraph:
    """5 结点 4 超边的真实小超图。

    拓扑：``e1={a,b,c}``、``e2={b,c}``、``e3={c,d}``、``e4={d,e}``；
    超度为 a=1、b=2、c=3、d=2、e=1，c 是唯一枢纽。
    """
    return hnx.Hypergraph(
        {
            "e1": ["a", "b", "c"],
            "e2": ["b", "c"],
            "e3": ["c", "d"],
            "e4": ["d", "e"],
        }
    )


@pytest.fixture()
def hgx_uniform(hg: hnx.Hypergraph) -> hgx.Hypergraph:
    """同一超图转成 HyperGraphX 对象，权重全相等（层内无权重异质性）。"""
    graph = convert_hypernetx_to_hypergraphx(hg)
    return graph


@pytest.fixture()
def hgx_skewed() -> hgx.Hypergraph:
    """层内权重严重不均的 HyperGraphX 超图（用于检验加权修正生效）。

    两条 2 元边权重 1.0 / 9.0，两条 3 元边权重 1.0 / 2.0；
    与 ``hgx_uniform`` 结构相同、仅权重不同。
    """
    return hgx.Hypergraph(
        [("a", "b"), ("c", "d"), ("a", "b", "c"), ("b", "c", "d")],
        weighted=True,
        weights=[1.0, 9.0, 1.0, 2.0],
    )


class TestCalcEntropy:
    """度分布熵。"""

    def test_value_matches_manual_entropy(self, hg: hnx.Hypergraph) -> None:
        """超度 [1,2,3,2,1] 的香农熵为 1.5229550675…（手工按公式算得）。"""
        expected = -sum(p * math.log(p) for p in (1 / 9, 2 / 9, 3 / 9, 2 / 9, 1 / 9))
        assert calc_entropy(hg) == pytest.approx(expected)

    def test_regular_hypergraph_has_max_entropy(self) -> None:
        """每个结点超度都是 2 时分布均匀，熵等于 ln(结点数)。"""
        H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["c", "d"]})
        assert calc_entropy(H) == pytest.approx(math.log(4))

    def test_degenerate_inputs_return_zero(self) -> None:
        """空超图与单结点超图的分布退化，熵为 0。"""
        assert calc_entropy(hnx.Hypergraph()) == 0.0
        assert calc_entropy(hnx.Hypergraph({"e1": ["a"]})) == 0.0


class TestCalcInformationGain:
    """高阶网络相对成对（投影）网络的信息增益。"""

    def test_matches_manual_entropy_difference(self, hg: hnx.Hypergraph) -> None:
        """信息增益 = 投影度分布熵 − 超度分布熵，两者均可手工复算。

        ``hg`` 的超度为 a=1、b=2、c=3、d=2、e=1；团投影里成对度为
        a=2、b=2、c=3、d=2、e=1（c 是唯一枢纽，e 只连 d）。
        """
        expected_pairwise = -sum(p * math.log(p) for p in (2 / 10, 2 / 10, 3 / 10, 2 / 10, 1 / 10))
        expected_high = -sum(p * math.log(p) for p in (1 / 9, 2 / 9, 3 / 9, 2 / 9, 1 / 9))
        assert calc_information_gain(hg) == pytest.approx(expected_pairwise - expected_high)

    def test_detail_returns_both_entropies(self, hg: hnx.Hypergraph) -> None:
        """``detail=True`` 返回两侧熵与差值，且差值等于标量口径。"""
        detail = calc_information_gain(hg, detail=True)
        assert set(detail) == {"pairwise_entropy", "hypergraph_entropy", "information_gain"}
        assert detail["information_gain"] == pytest.approx(
            detail["pairwise_entropy"] - detail["hypergraph_entropy"]
        )
        assert detail["information_gain"] == pytest.approx(calc_information_gain(hg))
        # 高阶侧熵就是 calc_entropy 的口径
        assert detail["hypergraph_entropy"] == pytest.approx(calc_entropy(hg))

    def test_pairwise_only_hypergraph_has_zero_gain(self) -> None:
        """全部超边都是 2 元边时，团投影就是原网络本身，信息增益为 0。"""
        H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"], "e3": ["c", "d"]})
        assert calc_information_gain(H) == pytest.approx(0.0)

    def test_star_hypergraph_gain_is_zero(self) -> None:
        """星形超图的投影度是超度的均匀放大，两个熵相等，增益为 0。

        中心结点 ``h`` 超度 3、其余各 1；投影里 ``h`` 的成对度为 6、其余各 2，
        即恰好是超度向量的 2 倍。熵对分布的等比缩放不变，因此增益为 0——
        这个例子说明「有高阶结构」并不自动等于「有信息增益」。
        """
        star = hnx.Hypergraph({"s1": ["h", "a", "b"], "s2": ["h", "c", "d"], "s3": ["h", "e", "f"]})
        assert calc_information_gain(star) == pytest.approx(0.0)

    def test_projection_more_skewed_gives_negative_gain(self) -> None:
        """投影把分布压得比高阶表示更不均匀时，增益为负。

        ``e1={a,b,c}``、``e2={c,d}``：c 的超度为 2，但投影里 c 的成对度为 3，
        比 a/b/d 的 2 更突出；投影熵因此略低于超度熵。
        """
        H = hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["c", "d"]})
        assert calc_information_gain(H) < 0.0

    def test_degenerate_inputs_return_zero(self) -> None:
        """空超图与单结点超图的分布退化，两侧熵均为 0，增益为 0。"""
        assert calc_information_gain(hnx.Hypergraph()) == 0.0
        assert calc_information_gain(hnx.Hypergraph({"e1": ["a"]})) == 0.0

    def test_is_cached_and_reflects_structure_change(self) -> None:
        """重复调用结果一致（走缓存）；结构变化后结果随之更新。"""
        H = hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["c", "d"]})
        first = calc_information_gain(H)
        assert calc_information_gain(H) == pytest.approx(first)
        H.remove_edges("e1")
        assert calc_information_gain(H) != pytest.approx(first)


class TestCalcClusteringCoefficientG:
    """全局聚类系数。"""

    def test_equals_mean_of_local_values(self, hg: hnx.Hypergraph) -> None:
        """局部系数 a=1、b=1、c=1/3、d=0、e=0，平均为 0.4666…。"""
        expected = (1.0 + 1.0 + 1.0 / 3.0 + 0.0 + 0.0) / 5
        assert calc_clustering_coefficient_g(hg) == pytest.approx(expected)

    def test_single_clique_is_one(self) -> None:
        """单条超边内部任意两结点都共享超边，全局系数为 1。"""
        H = hnx.Hypergraph({"e1": ["a", "b", "c"]})
        assert calc_clustering_coefficient_g(H) == pytest.approx(1.0)

    def test_empty_hypergraph_is_zero(self) -> None:
        """没有结点时返回 0.0（旧实现同样返回 0.0）。"""
        assert calc_clustering_coefficient_g(hnx.Hypergraph()) == 0.0


class TestCalcNaturalConnectivity:
    """自然连通度。"""

    def test_matches_eigenvalue_formula(self, hg: hnx.Hypergraph) -> None:
        """与「团展开邻接矩阵特征值的指数平均取对数」独立算得的值一致。"""
        adjacency = np.zeros((5, 5))
        nodes = ["a", "b", "c", "d", "e"]
        index = {node: i for i, node in enumerate(nodes)}
        for members in (["a", "b", "c"], ["b", "c"], ["c", "d"], ["d", "e"]):
            for i, u in enumerate(members):
                for v in members[i + 1 :]:
                    adjacency[index[u], index[v]] = 1.0
                    adjacency[index[v], index[u]] = 1.0
        eigenvalues = np.linalg.eigvalsh(adjacency)
        expected = float(np.log(np.mean(np.exp(eigenvalues))))
        assert calc_natural_connectivity(hg) == pytest.approx(expected)

    def test_empty_hypergraph_is_zero(self) -> None:
        """空超图直接返回 0.0（避免对空矩阵求特征值）。"""
        assert calc_natural_connectivity(hnx.Hypergraph()) == 0.0

    def test_single_edge_is_positive(self) -> None:
        """单条 2 元边的邻接矩阵特征值为 ±1，自然连通度为 ln(cosh 1)。"""
        H = hnx.Hypergraph({"e1": ["a", "b"]})
        assert calc_natural_connectivity(H) == pytest.approx(math.log(math.cosh(1.0)))


class TestAssortativityTest:
    """按超度分组的余平均度。"""

    def test_values_match_manual_excess(self, hg: hnx.Hypergraph) -> None:
        """超度 1 的结点（a、e）余平均度均为 2.5；超度 2 的为 2.0；超度 3 的为 5/3。"""
        result = assortativity_test(hg)
        assert result == pytest.approx({1: 2.25, 2: 2.0, 3: 5.0 / 3.0})
        assert list(result) == [1, 2, 3]

    def test_empty_hypergraph_returns_empty_dict(self) -> None:
        """没有结点时返回空字典。"""
        assert assortativity_test(hnx.Hypergraph()) == {}


class TestCalcAssortativity:
    """高阶同配系数与零模型显著性。"""

    def test_rank_correlation_of_hub_edges(self, hg: hnx.Hypergraph) -> None:
        """四条超边的 (最大度结点秩, 第二大度结点秩) 完全同向，相关系数为 1。"""
        r, p = calc_assortativity(hg, n_rand=20, n_steps=20)
        assert r == pytest.approx(1.0)
        assert 0.0 <= p <= 1.0

    def test_reproducible_with_same_seed(self, hg: hnx.Hypergraph) -> None:
        """同一 seed 下两次调用的 p 值完全相同（零模型抽样可复现）。"""
        first = calc_assortativity(hg, n_rand=10, n_steps=10, seed=7)
        second = calc_assortativity(hg, n_rand=10, n_steps=10, seed=7)
        assert first == second

    def test_accepts_hypergraphx_input(
        self, hg: hnx.Hypergraph, hgx_uniform: hgx.Hypergraph
    ) -> None:
        """同一超图的 hypergraphx 与 hypernetx 表示给出相同观测同配系数。"""
        r_hgx, _ = calc_assortativity(hgx_uniform, n_rand=5, n_steps=5)
        r_hnx, _ = calc_assortativity(hg, n_rand=5, n_steps=5)
        assert r_hgx == pytest.approx(r_hnx)

    def test_degenerate_hypergraph_returns_zero(self) -> None:
        """只有一条 2 元边时秩序列方差为 0，按约定返回 0。"""
        H = hnx.Hypergraph({"e1": ["a", "b"]})
        r, _ = calc_assortativity(H, n_rand=5, n_steps=5)
        assert r == 0.0


class TestSubReducibility:
    """加权修正版可约性。"""

    def test_returns_value_and_representative_layers(self, hgx_uniform: hgx.Hypergraph) -> None:
        """返回 (可约性, 代表层元组)，可约性在 [0, 1] 内。"""
        value, reps = sub_reducibility(hgx_uniform, optimization="greedy")
        assert 0.0 <= value <= 1.0
        assert isinstance(reps, tuple)
        assert all(isinstance(layer, int) for layer in reps)

    def test_uniform_weights_match_official_reducibility(self, hgx_uniform: hgx.Hypergraph) -> None:
        """层内权重全相等时，权重成本为 0，结果退化为官方无权实现。"""
        official = hgx.measures.reducibility.reducibility(hgx_uniform, optimization="greedy")
        value, _ = sub_reducibility(hgx_uniform, optimization="greedy")
        assert value == pytest.approx(float(official[0]))

    def test_skewed_weights_lower_reducibility(self, hgx_skewed: hgx.Hypergraph) -> None:
        """层内权重越不均，层越难被代表，可约性应低于均匀权重版本。"""
        skewed, _ = sub_reducibility(hgx_skewed, optimization="greedy")
        uniform_graph = hgx.Hypergraph(
            [("a", "b"), ("c", "d"), ("a", "b", "c"), ("b", "c", "d")],
            weighted=True,
            weights=[1.0, 1.0, 1.0, 1.0],
        )
        uniform, _ = sub_reducibility(uniform_graph, optimization="greedy")
        assert skewed < uniform

    def test_exact_and_greedy_both_valid(self, hgx_skewed: hgx.Hypergraph) -> None:
        """exact 与 greedy 两种优化方式都返回合法结果（本图上取值相同）。"""
        exact, reps_exact = sub_reducibility(hgx_skewed, optimization="exact")
        greedy, reps_greedy = sub_reducibility(hgx_skewed, optimization="greedy")
        assert 0.0 <= exact <= 1.0
        assert 0.0 <= greedy <= 1.0
        assert exact == pytest.approx(greedy)
        assert reps_exact == reps_greedy

    def test_invalid_parameters_raise(self, hgx_uniform: hgx.Hypergraph) -> None:
        """非法的 optimization / entropy_method 抛 HyperGraphX 的参数异常。"""
        from hypergraphx.exceptions import InvalidParameterError

        with pytest.raises(InvalidParameterError):
            sub_reducibility(hgx_uniform, optimization="bogus")
        with pytest.raises(InvalidParameterError):
            sub_reducibility(hgx_uniform, entropy_method="bogus")


class TestCalcReducibility:
    """可约性入口（无权重走官方实现，加权走修正版）。"""

    def test_hypernetx_input_returns_float(self, hg: hnx.Hypergraph) -> None:
        """默认 no_layer=True，返回标量可约性。"""
        value = calc_reducibility(hg)
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    def test_no_layer_false_returns_pair(self, hg: hnx.Hypergraph) -> None:
        """no_layer=False 时返回 (可约性, 代表层元组)。"""
        value, reps = calc_reducibility(hg, no_layer=False)
        assert 0.0 <= value <= 1.0
        assert isinstance(reps, tuple)

    def test_weighted_differs_on_skewed_weights(self, hgx_skewed: hgx.Hypergraph) -> None:
        """权重严重不均时，加权版可约性严格低于无权版。"""
        weighted = calc_reducibility(hgx_skewed, weighted=True)
        plain = calc_reducibility(hgx_skewed, weighted=False)
        assert weighted < plain

    def test_weighted_matches_sub_reducibility(self, hgx_skewed: hgx.Hypergraph) -> None:
        """weighted=True 分支就是 sub_reducibility 的标量形式。"""
        expected, _ = sub_reducibility(hgx_skewed, optimization="greedy")
        assert calc_reducibility(hgx_skewed, weighted=True) == pytest.approx(expected)

    def test_rejects_non_hypergraph(self) -> None:
        """非超图对象在入口处抛 TypeError（旧实现会传到下游报难懂的错）。"""
        with pytest.raises(TypeError, match="hypernetx 或 hypergraphx"):
            calc_reducibility("not a hypergraph")


class TestHnxsGlobal:
    """多超图全局指标汇总表。"""

    def test_dataframe_shape_and_values(self, hg: hnx.Hypergraph) -> None:
        """行是时间键、列是「高阶网络可约性」，取值等于逐图算出的可约性。"""
        other = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        table = hnxs_global({2020: hg, 2021: other})

        assert list(table.columns) == ["高阶网络可约性"]
        assert list(table.index) == [2020, 2021]
        assert table.loc[2020, "高阶网络可约性"] == pytest.approx(
            calc_reducibility(hg, weighted=True)
        )

    def test_accepts_hypergraphx_input(self, hgx_uniform: hgx.Hypergraph) -> None:
        """hypergraphx 超图同样可以直接放进字典。"""
        table = hnxs_global({2019: hgx_uniform})
        assert list(table.index) == [2019]
        assert 0.0 <= table.loc[2019, "高阶网络可约性"] <= 1.0


class TestLazyExport:
    """惰性导出契约。"""

    def test_import_does_not_pull_third_party(self) -> None:
        """``import honet.metrics`` 本身不加载 numpy / hypernetx / hypergraphx。"""
        code = (
            "import sys, honet.metrics; "
            "print([m for m in ('numpy', 'hypernetx', 'hypergraphx', 'pandas') "
            "if m in sys.modules])"
        )
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "[]"

    def test_exports_sorted_and_complete(self) -> None:
        """``__all__`` 覆盖既有 14 个指标与新增的信息增益，且按字母序。"""
        import honet.metrics as metrics

        assert _NEW_EXPORTS <= set(metrics.__all__)
        assert metrics.__all__ == sorted(metrics.__all__)

    def test_unknown_attribute(self) -> None:
        """访问不存在的符号抛 AttributeError。"""
        import honet.metrics as metrics

        missing = "no_such_metric"
        with pytest.raises(AttributeError, match=missing):
            getattr(metrics, missing)

    def test_implementation_is_three_modules(self) -> None:
        """实现按粒度收敛到三个模块：结点 / 超边 / 全局。"""
        import honet.metrics as metrics

        modules = {
            target.partition(":")[0]
            for target in metrics._EXPORTS.values()  # noqa: SLF001 - 契约测试需要读导出表
        }
        assert modules == {
            "honet.metrics.nodemetrics",
            "honet.metrics.edgemetrics",
            "honet.metrics.globalmetrics",
        }

    def test_package_attributes_stay_callable(self) -> None:
        """访问后包属性是函数而非模块对象。

        实现模块名（``nodemetrics`` 等）与公开函数名（``clustering_coefficient``
        等）已不再相同，子模块 import 不会再遮蔽同名包属性；这里仍逐项校验，
        防止将来把某个导出指向与函数同名的模块而重新引入遮蔽。
        """
        import honet.metrics as metrics

        for name in ("clustering_coefficient", "excess_average_degree"):
            assert callable(getattr(metrics, name))
        # 内部复用后仍然可调用
        calc_clustering_coefficient_g(hnx.Hypergraph({"e1": ["a", "b", "c"]}))
        assert callable(metrics.clustering_coefficient)

    def test_survives_external_submodule_import(self, hg: hnx.Hypergraph) -> None:
        """其它子包在模块级 import 本子包的实现模块后，公开函数仍可调用。

        ``honet.attack`` 的实现模块在模块级 import 了
        ``honet.metrics.nodemetrics``。由于实现模块名与公开函数名不同，
        import 机制只会写回 ``honet.metrics.nodemetrics`` 这个属性，
        不会影响 ``honet.metrics.clustering_coefficient``。
        """
        import honet.attack  # noqa: F401 - 触发对 honet.metrics 实现模块的导入
        import honet.metrics as metrics

        # 真正加载 attack 的实现模块（其模块级 import 会加载 nodemetrics）
        assert callable(honet.attack.degree_attack_sequence)
        for name in ("clustering_coefficient", "excess_average_degree"):
            assert callable(getattr(metrics, name))

        # 函数仍能正常计算
        assert metrics.clustering_coefficient(hg)["a"] == pytest.approx(1.0)
        assert set(metrics.excess_average_degree(hg)) == set(hg.nodes)


class TestTemporalClosenessCentrality:
    """时序接近中心性：只承认时间尊重路径。"""

    def test_chain_matches_hand_computation(self) -> None:
        """``(1,a-b) (2,b-c)``：a→b 为 1 跳、a→c 为 2 跳，b 两向都是 1 跳。

        手算：C(a)=(1/1+1/2)/2=0.75，C(b)=(1/1+1/1)/2=1.0，C(c)=1/1/2=0.5。
        """
        tn = TemporalNetwork(events=[(1, ["a", "b"]), (2, ["b", "c"])])
        scores = temporal_closeness_centrality(tn)

        assert scores["a"] == pytest.approx(0.75)
        assert scores["b"] == pytest.approx(1.0)
        assert scores["c"] == pytest.approx(0.5)

    def test_time_order_is_respected(self) -> None:
        """同样的边、颠倒时间顺序，结果必须不同——这是时序指标的核心。

        事件 ``b-c`` 早于 ``a-b`` 时，a 无法沿时间到达 c。
        """
        forward = TemporalNetwork(events=[(1, ["a", "b"]), (2, ["b", "c"])])
        backward = TemporalNetwork(events=[(2, ["a", "b"]), (1, ["b", "c"])])

        assert temporal_closeness_centrality(forward)["a"] == pytest.approx(0.75)
        # a 只能到 b：1 跳，除以 n-1=2
        assert temporal_closeness_centrality(backward)["a"] == pytest.approx(0.5)

    def test_three_hop_chain(self) -> None:
        """``a-b, b-c, c-d`` 依次发生：a 到 d 需 3 跳。"""
        tn = TemporalNetwork(events=[(1, ["a", "b"]), (2, ["b", "c"]), (3, ["c", "d"])])
        scores = temporal_closeness_centrality(tn)

        assert scores["a"] == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)
        assert scores["d"] == pytest.approx(1 / 3)

    def test_scores_within_unit_interval(self) -> None:
        """调和形式保证取值落在 ``[0, 1]``。"""
        tn = TemporalNetwork(events=[(1, ["a", "b", "c"]), (2, ["c", "d"]), (3, ["d", "e", "a"])])
        scores = temporal_closeness_centrality(tn)

        assert all(0.0 <= value <= 1.0 for value in scores.values())

    def test_isolated_node_scores_zero(self) -> None:
        """没有任何时间路径的结点取 0.0，且不污染其他结点。"""
        tn = TemporalNetwork(nodes=["z"], events=[(1, ["a", "b"])])
        scores = temporal_closeness_centrality(tn)

        assert scores["z"] == pytest.approx(0.0)
        # a 到 b 是 1 跳，分母为 n-1=2（z 虽不可达仍占分母，与调和中心性一致）
        assert scores["a"] == pytest.approx(0.5)
        # 关键：z 的不可达只体现为它自己贡献 0，没有把 a 的分子也清零
        assert scores["a"] > 0.0

    def test_max_hops_limits_reach(self) -> None:
        """``max_hops`` 限制可见跳数，超出部分不计入。"""
        tn = TemporalNetwork(events=[(1, ["a", "b"]), (2, ["b", "c"]), (3, ["c", "d"])])

        one_hop = temporal_closeness_centrality(tn, max_hops=1)
        two_hops = temporal_closeness_centrality(tn, max_hops=2)

        assert one_hop["a"] == pytest.approx(1 / 3)
        assert two_hops["a"] == pytest.approx((1 + 1 / 2) / 3)

    def test_max_hops_must_be_positive_int(self) -> None:
        """非法 ``max_hops`` 抛 ValueError（含 bool）。"""
        tn = TemporalNetwork(events=[(1, ["a", "b"])])

        for bad in (0, -1, True, 1.5):
            with pytest.raises(ValueError, match="max_hops"):
                temporal_closeness_centrality(tn, max_hops=bad)  # type: ignore[arg-type]

    def test_rejects_non_temporal_network(self) -> None:
        """只接受 TemporalNetwork：一次性超图 / 普通对象一律拒绝。"""
        for bad in (hnx.Hypergraph({"e1": ["a", "b"]}), ["a", "b"], {"e1": {"a", "b"}}):
            with pytest.raises(TypeError, match="TemporalNetwork"):
                temporal_closeness_centrality(bad)  # type: ignore[arg-type]

    def test_accepts_weighted_variant(self) -> None:
        """加权 / 有向变体继承自 TemporalNetwork，同样可用。"""
        tn = TemporalNetworkW(events=[(1, ["a", "b"]), (2, ["b", "c"])])
        scores = temporal_closeness_centrality(tn)

        assert scores["a"] == pytest.approx(0.75)

    def test_empty_and_single_node(self) -> None:
        """空网络返回空字典；单结点网络该结点取 0.0。"""
        assert temporal_closeness_centrality(TemporalNetwork()) == {}
        assert temporal_closeness_centrality(TemporalNetwork(nodes=["x"])) == {"x": 0.0}

    def test_cached_result_invalidated_by_new_event(self) -> None:
        """结构变化后缓存必须失效，否则会返回旧网络的答案。"""
        tn = TemporalNetwork(events=[(1, ["a", "b"])])
        before = temporal_closeness_centrality(tn)
        assert set(before) == {"a", "b"}

        tn.add_event(["b", "c"], time=2)
        after = temporal_closeness_centrality(tn)

        assert set(after) == {"a", "b", "c"}
        assert after["a"] == pytest.approx(0.75)

    def test_cache_distinguishes_event_order(self) -> None:
        """边数/结点数/成员数完全相同、仅时间顺序不同的网络不能共用缓存。

        三个计数相同意味着旧的一级校验会误判命中；时序指标依赖时间顺序，
        因此结构特征必须把时间线纳入。
        """
        from honet import _cache

        first = TemporalNetwork(events=[(2, ["a", "b"]), (1, ["b", "c"])])
        second = TemporalNetwork(events=[(1, ["a", "b"]), (2, ["b", "c"])])

        # 计数三元组一致，但结构特征必须不同
        assert _cache.structure(first)[:3] == _cache.structure(second)[:3]
        assert _cache.structure(first) != _cache.structure(second)

        assert temporal_closeness_centrality(first)["a"] == pytest.approx(0.5)
        assert temporal_closeness_centrality(second)["a"] == pytest.approx(0.75)
