"""从旧库 hypertrade 的测试迁移而来。

来源：``D:\\1-Project\\Paper\\hypertrade\\tests``（``test_core.py``、``test_dump.py``、
``test_honmetric.py``、``test_honcommunity.py``、``test_honcontrol.py``）。

迁移原则：

1. **断言值不变**：原测试检查的数值、集合、类型全部保留，只把导入路径从
   ``hypertrade.*`` 换成 ``honet.*``（公共 API 一律从子包取）。
2. **保留原测试意图**，即使 honet 的接口形态已经不同（例如 ``build.*`` 现在返回
   超图对象而不是 ``{"head","tail"}`` 列表），也通过显式转换复现原测试场景。
3. **不迁入已知在原库就失败、或依赖过时语义的断言**，改为记录当前正确语义，
   并在测试内注明原因（见 :func:`test_calc_assortativity_returns_pair`）。

原库与 honet 的接口差异（迁移时逐条确认过）：

=============================== ==========================================
旧库 hypertrade                  honet
=============================== ==========================================
``hypertrade.core``              ``honet.convert``
``hypertrade.dump.to_latex``     ``honet.export.to_latex``
``hypertrade.honmetric``         ``honet.metrics``
``hypertrade.metric.*``          ``honet.metrics``
``hypertrade.honcommunity``      ``honet.community``
``hypertrade.honcontrol``        ``honet.control``
``hypertrade.honbuild``          ``honet.build``
``build.*`` 返回 dict 列表       ``build.*`` 返回 ``hypernetx.Hypergraph``
=============================== ==========================================
"""

from __future__ import annotations

import random
import time

import hypernetx as hnx
import networkx as nx
import numpy as np
import pandas as pd
import pytest

import honet
from honet.build import (
    clustered_hypergraph,
    scale_free_hypergraph,
    small_world_hypergraph,
)
from honet.control import (
    add_node_fields,
    brute_force_min_driver_sets,
    compute_and_export_drivers,
    compute_drivers_from_hypernetx,
    detect_dilation_and_drivers,
    detect_dilation_with_given_drivers,
    edge_list_to_incidence,
    greedy_accessibility,
    random_uniform_hypergraph,
)
from honet.convert import (
    convert_hypernetx_to_hypergraphx,
    convert_hypernetx_to_nxgraph,
    convert_hypernetx_to_xgi,
    convert_xgi_to_hypernetx,
    get_hyperedge_weights,
)
from honet.export import to_latex
from honet.metrics import (
    assortativity_test,
    calc_assortativity,
    calc_clustering_coefficient_g,
    calc_entropy,
    calc_reducibility,
    get_avg_degree_by_edge_size,
    weighted_degree,
)


def _weighted_country_frame() -> pd.DataFrame:
    """原测试反复使用的国家—协议加权边表。"""
    return pd.DataFrame(
        {
            "country": ["China", "USA", "China", "EU", "USA"],
            "agreement": ["Trade", "Trade", "Climate", "Climate", "Tech"],
            "weight": [2.0, 3.0, 1.5, 2.5, 4.0],
        }
    )


def _build_weighted_hypergraph() -> hnx.Hypergraph:
    """由上面的边表构建带权超图（原测试的构造方式）。"""
    return hnx.Hypergraph(
        _weighted_country_frame(),
        edge_col="agreement",
        node_col="country",
        cell_weight_col="weight",
    )


def _to_head_tail_dicts(H: hnx.Hypergraph) -> list[dict[str, list[int]]]:
    """把 ``build.*`` 返回的超图还原成旧库的 ``[{"head","tail"}, ...]`` 形式。

    honet 的 ``build.*`` 有意改为直接返回 ``hypernetx.Hypergraph``（见其
    docstring 的「与旧实现的差异」），head/tail 保留为超边属性，因此可以无损
    还原。原测试里的 ``edge_list_to_incidence`` 需要这个形式。
    """
    return [
        {"head": list(H.edges[edge].head), "tail": list(H.edges[edge].tail)} for edge in H.edges
    ]


class TestCore:
    """迁移自 ``test_core.py``。"""

    def test_load_map_data(self):
        """加载 shp 得到非空 GeoDataFrame。"""
        pytest.importorskip("geopandas")
        from honet.io import load_map_data

        path = r"D:\1-Project\Paper\hypertrade\tests\tests_data\test.shp"
        df = load_map_data(path)
        assert not df.empty, "The loaded GeoDataFrame should not be empty."

    def test_convert_hypernetx_to_hypergraphx(self):
        """转成 HyperGraphX 后结点数不变，退化超边被丢弃故边数变少。"""
        H_hnx = _build_weighted_hypergraph()

        H_hgx = convert_hypernetx_to_hypergraphx(H_hnx)
        assert H_hgx.num_nodes() == H_hnx.shape[0], "节点数不匹配"
        assert H_hgx.num_edges() != H_hnx.shape[1], "转换会删掉没用的超边，所以应该是不相等的"

    def test_convert_hypernetx_to_xgi(self):
        """hypernetx → XGI → hypernetx 往返保持结点数与边数。"""
        test_edges = {
            "e1": [1, 2, 3],
            "e2": [2, 3, 4],
            "e3": [4, 5],
            "e4": [1, 5],
        }
        H_hnx = hnx.Hypergraph(test_edges)

        H_xgi, _ = convert_hypernetx_to_xgi(H_hnx)
        assert H_xgi.num_nodes == H_hnx.shape[0], "节点数不匹配"
        assert H_xgi.num_edges == H_hnx.shape[1], "边数不匹配"

        H_hnx_converted = convert_xgi_to_hypernetx(H_xgi)
        assert H_hnx_converted.shape[0] == H_hnx.shape[0], "转换回的节点数不匹配"
        assert H_hnx_converted.shape[1] == H_hnx.shape[1], "转换回的边数不匹配"

    def test_convert_hypernetx_to_nxgraph(self):
        """团投影：每条超边展开成完全图，单结点超边被忽略。"""
        edge_dict = {
            "e1": {"a", "b", "c"},  # 3 个节点 → 3 条边
            "e2": {"c", "d"},  # 2 个节点 → 1 条边
            "e3": {"x"},  # 单节点超边 → 应被忽略
        }
        h = hnx.Hypergraph(edge_dict)
        G = convert_hypernetx_to_nxgraph(h)

        assert isinstance(G, nx.Graph), "返回值必须是 NetworkX 图对象"
        assert not G.is_directed(), "应返回无向图"

        assert G.number_of_nodes() == 4, f"节点数错误: {G.number_of_nodes()}"
        assert set(G.nodes) == {"a", "b", "c", "d"}, f"节点集合错误: {set(G.nodes)}"

        expected_edges = {
            frozenset(["a", "b"]),
            frozenset(["a", "c"]),
            frozenset(["b", "c"]),
            frozenset(["c", "d"]),
        }
        actual_edges = {frozenset(e) for e in G.edges}
        assert actual_edges == expected_edges, f"边集合错误: {actual_edges}"
        assert G.number_of_edges() == 4

        assert G.has_edge("a", "b") and G.has_edge("a", "c") and G.has_edge("b", "c")
        assert G.has_edge("c", "d")
        assert not G.has_edge("a", "d"), "a 和 d 不同时出现在任何超边中"
        assert not G.has_edge("a", "x"), "单节点超边应被忽略"

        # 单节点超边全部忽略
        h2 = hnx.Hypergraph({"e1": {"a"}, "e2": {"b"}, "e3": {"c"}})
        G2 = convert_hypernetx_to_nxgraph(h2)
        assert G2.number_of_nodes() == 0
        assert G2.number_of_edges() == 0

        # 两个节点互相重复出现：无向图应去重
        h3 = hnx.Hypergraph(
            {
                "e1": {"a", "b"},
                "e2": {"a", "b"},
                "e3": {"b", "a"},
            }
        )
        G3 = convert_hypernetx_to_nxgraph(h3)
        assert G3.number_of_edges() == 1, f"重复边应去重, 实际: {G3.number_of_edges()}"
        assert G3.has_edge("a", "b")

        # 大超边的组合数验证：C(4, 2) = 6
        h4 = hnx.Hypergraph({"e1": {"a", "b", "c", "d"}})
        G4 = convert_hypernetx_to_nxgraph(h4)
        assert G4.number_of_edges() == 6, f"边数错误: {G4.number_of_edges()}"
        assert G4.number_of_nodes() == 4

    def test_get_hyperedge_weights(self):
        """超边权重是成员权重之和。"""
        result = get_hyperedge_weights(_build_weighted_hypergraph())
        assert result["Trade"] == 5.0  # China + USA
        assert result["Climate"] == 4.0  # China + EU
        assert result["Tech"] == 4.0  # USA


class TestDump:
    """迁移自 ``test_dump.py``。"""

    def test_to_latex_writes_full_table(self):
        """LaTeX 输出逐字符匹配（含缩进、表头、标签位置）。"""
        df = pd.DataFrame(
            {
                "年份": [2017, 2018],
                "密度": [0.039, 0.038],
                "熵": [4.646, 4.702],
            }
        )

        text = to_latex(
            df,
            "全局超网络系数",
            "Global hypernetwork coefficients",
            "tab:global_coef",
        )

        assert text == (
            "\\begin{center}\n"
            "\t\\bicaption[table]{全局超网络系数}{Global hypernetwork coefficients}\n"
            "\t{\\small\n"
            "\t\t\\setlength{\\tabcolsep}{4pt}\n"
            "\t\t\\renewcommand{\\arraystretch}{1.2}\n"
            "\t\t\\begin{tabular}{l*{2}{c}}\n"
            "\t\t\t\\hline\n"
            "\t\t\t年份 & 密度 & 熵 \\\\\n"
            "\t\t\t\\hline\n"
            "\t\t\t2017 & 0.039 & 4.646 \\\\\n"
            "\t\t\t2018 & 0.038 & 4.702 \\\\\n"
            "\t\t\t\\hline\n"
            "\t\t\\end{tabular}\n"
            "\t}\n"
            "\t\\label{tab:global_coef}\n"
            "\\end{center}\n"
        )


class TestMetrics:
    """迁移自 ``test_honmetric.py``。"""

    def test_get_avg_degree_by_edge_size(self):
        """按超边规模分组求全部结点的平均度。"""
        edges = {
            "e1": {1, 2},  # 大小 2
            "e2": {2, 3},  # 大小 2
            "e3": {1, 2, 3},  # 大小 3
            "e4": {4, 5, 6, 7},  # 大小 4
            "e5": {1, 4},  # 大小 2
        }
        H = hnx.Hypergraph(edges)
        avg_degrees = get_avg_degree_by_edge_size(H)

        # 分母是全部 7 个结点：size=2 时度依次为 2,2,1,1,0,0,0，和为 6
        assert avg_degrees[2] == 6 / 7
        assert avg_degrees[3] == 3 / 7
        assert avg_degrees[4] == 4 / 7

    def test_calc_reducibility_returns_expected_value_for_simple_hypergraph(self):
        """两层超边都只含 size=2，可约性为 1，代表层为 (2,)。"""
        H = hnx.Hypergraph({1: {"a", "b"}, 2: {"b", "c"}})

        assert calc_reducibility(H) == pytest.approx(1.0)
        assert calc_reducibility(H, no_layer=False)[0] == pytest.approx(1.0)
        assert calc_reducibility(H, no_layer=False)[1] == (2,)

    def test_calc_entropy_returns_expected_value_for_simple_hypergraph(self):
        """单条二结点超边的熵为 ln2。"""
        H = hnx.Hypergraph({1: {"a", "b"}})

        assert calc_entropy(H) == pytest.approx(0.6931471805599453)

    def test_calc_assortativity_returns_pair(self):
        """``calc_assortativity`` 返回 ``(r, p)``，单超边时 r 为 0。

        原测试断言 ``calc_assortativity(H) == 0.0``，但该函数自旧库起就返回
        ``(r, p)`` 二元组（旧库 docstring 明确写着 ``(r, z)``），因此原断言在
        旧库上也是失败的（基线运行：``(0.0, 1.0) == 0.0``）。这里改为校验当前
        真实语义：相关系数为 0、p 值落在 [0, 1]。
        """
        H = hnx.Hypergraph({1: {"a", "b"}})

        r, p = calc_assortativity(H)
        assert r == pytest.approx(0.0)
        assert 0.0 <= p <= 1.0

    def test_calc_clustering_coefficient_g_returns_zero_for_simple_edge_case(self):
        """单条超边的全局聚类系数为 0。"""
        H = hnx.Hypergraph({1: {"a", "b"}})

        assert calc_clustering_coefficient_g(H) == pytest.approx(0.0)

    def test_test_assortativity(self):
        """同度结点的余平均度应相等（无度相关性）。"""
        # 所有边成员完全相同
        H = hnx.Hypergraph(
            {
                "e1": ["C", "A", "B"],
                "e2": ["C", "A", "B"],
                "e3": ["C", "A", "B"],
            }
        )
        assert len(set(assortativity_test(H).values())) == 1, "所有的度为k的节点的余平均度相等"

        # 所有边没有交点
        H = hnx.Hypergraph(
            {
                "e1": ["A", "B"],
                "e2": ["C", "D"],
                "e3": ["F", "E"],
            }
        )
        assert len(set(assortativity_test(H).values())) == 1, "所有的度为k的节点的余平均度相等"

    def test_node_degree(self):
        """加权度等于所在超边权重之和。"""
        result = weighted_degree(_build_weighted_hypergraph())
        assert result["China"] == 3.5, "China的加权度应为3.5"
        assert result["USA"] == 7.0, "USA的加权度应为7.0"
        assert result["EU"] == 2.5, "EU的加权度应为2.5"


class TestCommunity:
    """迁移自 ``test_honcommunity.py``。"""

    def test_hierarchy_method(self):
        """层次社区发现在简单超图上返回非空结果。"""
        edges = {0: [1, 2], 1: [2, 3], 2: [3, 4], 3: [4, 5]}
        hnx_g = hnx.Hypergraph(edges)

        result = honet.community.Hierarchy_Method(hnx_g)
        assert result is not None


class TestControl:
    """迁移自 ``test_honcontrol.py``。

    这些用例依赖 ``HypergraphAnalysisToolbox``（``honet[hat]``）。
    """

    def test_compute_and_export_drivers(self):
        """随机均匀有向超图（n=30, k=4, m=40）应检测到 dilation。"""
        pytest.importorskip("HAT")

        res1 = compute_and_export_drivers(edge_list=None, n=30, k=4, m=40, out_csv=None)
        assert res1 is not None
        assert res1["algorithm"] == "matching_plus_greedy"
        assert res1["dilation_exists"] is True

    def test_compute_drivers_from_hypernetx_small(self):
        """非均匀超图（边大小 2/3/4/5）→ 10 个驱动结点。"""
        pytest.importorskip("HAT")

        hnx_hg = hnx.Hypergraph(
            {
                "e1": ["A", "B", "C"],  # 3
                "e2": ["B", "D"],  # 2
                "e3": ["A", "D", "E", "F"],  # 4
                "e4": ["C", "E", "G"],  # 3
                "e5": ["F", "G", "H", "I", "J"],  # 5
                "e6": ["A", "H"],  # 2
                "e7": ["B", "I", "J"],  # 3
            }
        )
        assert hnx_hg.shape[0] == 10
        assert hnx_hg.shape[1] == 7
        assert not len(set(len(e) for e in hnx_hg.incidence_dict.values())) == 1

        res2 = compute_drivers_from_hypernetx(hnx_hg, out_csv=None)
        assert res2 is not None
        assert res2["dilation_exists"] is True
        assert len(res2["drivers"]) == 10
        assert res2["algorithm"] == "matching_plus_greedy"

    def test_compute_drivers_from_hypernetx_large(self):
        """更大的非均匀超图（n=20 随机边）→ 18 个驱动结点。"""
        pytest.importorskip("HAT")

        np.random.seed(42)
        random.seed(42)
        node_labels = [chr(ord("A") + i) for i in range(20)]
        hnx_big = {}
        for i in range(15):
            edge_size = random.randint(2, 6)
            edge_nodes = tuple(sorted(random.sample(node_labels, edge_size)))
            hnx_big[f"e{i}"] = list(edge_nodes)
        hnx_big_hg = hnx.Hypergraph(hnx_big)
        assert hnx_big_hg.shape[0] == 18
        assert hnx_big_hg.shape[1] == 15
        assert not len(set(len(e) for e in hnx_big_hg.incidence_dict.values())) == 1

        res3 = compute_drivers_from_hypernetx(hnx_big_hg, out_csv=None)
        assert res3 is not None
        assert res3["dilation_exists"] is True
        assert len(res3["drivers"]) == 18
        assert res3["algorithm"] == "matching_plus_greedy"

    def test_run_on_small(self):
        """小网络（n=10, k=3, m=15）上四种策略的规模与耗时关系。"""
        pytest.importorskip("HAT")
        HAT = pytest.importorskip("HAT")

        n, k, m = 10, 3, 15
        edge_list = random_uniform_hypergraph(n, k, m)
        incidence = edge_list_to_incidence(edge_list)
        nodes = pd.DataFrame({"Nodes": np.arange(n)})
        HG = HAT.Hypergraph(incidence_matrix=incidence, directed=True, nodes=nodes)
        HG = add_node_fields(HG)

        # 1. 匹配（用于 dilation 检测）
        t0 = time.perf_counter()
        dilation_exists, _matched_nodes, driver_nodes, _matching = detect_dilation_and_drivers(
            HG, HG.nnodes
        )
        t_matching = time.perf_counter() - t0

        if dilation_exists:
            unmatched_nodes = {int(str(u).replace("n", "")) for u in driver_nodes}
        else:
            unmatched_nodes = set()

        raw_matching_size = len(unmatched_nodes)
        matching_size = max(1, raw_matching_size)

        # 2. 匹配 + 贪心
        t0 = time.perf_counter()
        drivers_mg = greedy_accessibility(HG, unmatched_nodes)
        t_mg = time.perf_counter() - t0 + t_matching
        greedy_size = len(drivers_mg)

        # 3. 纯贪心（允许失败）
        t0 = time.perf_counter()
        drivers_pg = greedy_accessibility(HG, set())

        has_dil_pg = detect_dilation_with_given_drivers(HG, drivers_pg)
        if has_dil_pg:
            pure_greedy_size = np.nan
            t_pg = np.nan
        else:
            pure_greedy_size = len(drivers_pg)
            t_pg = time.perf_counter() - t0

        # 4. 暴力最优（可能很慢）
        t0 = time.perf_counter()
        optimal_size, _min_sets = brute_force_min_driver_sets(HG, verbose=False)
        t_optimal = time.perf_counter() - t0

        assert matching_size <= greedy_size
        assert n >= optimal_size
        assert pure_greedy_size >= greedy_size or np.isnan(pure_greedy_size)
        assert optimal_size <= greedy_size
        assert t_matching < t_mg
        assert t_pg < t_optimal or np.isnan(t_pg)
        assert optimal_size == 3  # 只有解析的稳定

    @pytest.mark.parametrize(
        "builder",
        [scale_free_hypergraph, clustered_hypergraph, small_world_hypergraph],
        ids=["scale_free", "clustered", "small_world"],
    )
    def test_run_on_different_toplogy(self, builder):
        """三种拓扑（无标度 / 聚类 / 小世界）都应检测到 dilation。

        原测试用 ``build.*`` 的返回值直接喂 ``edge_list_to_incidence``；honet 的
        ``build.*`` 改为返回超图对象，因此这里先还原成 head/tail 列表再走同一条
        链路，保持原测试的构造方式不变。
        """
        pytest.importorskip("HAT")

        edge_list = _to_head_tail_dicts(builder(n=30, k=3, m=40))
        incidence = edge_list_to_incidence(edge_list)
        nodes_df = pd.DataFrame({"Nodes": np.arange(30)})
        hg = hnx.Hypergraph(incidence_matrix=incidence, nodes=nodes_df, directed=True)

        res = compute_drivers_from_hypernetx(hg, out_csv=None)
        assert res is not None
        assert res["dilation_exists"] is True

    def test_positive_example(self):
        """结构可控的正面例子：无 dilation，5 个驱动结点。"""
        pytest.importorskip("HAT")

        edges = {
            "e1": ("u1", "v1"),  # 尾部 {u1} -> 头部 {v1}
            "e2": ("u2", "v2"),  # 尾部 {u2} -> 头部 {v2}
            "e3": ("v1", "v2", "v3"),  # 尾部 {v1, v2} -> 头部 {v3}
        }

        H = hnx.Hypergraph(edges)

        res_h = compute_drivers_from_hypernetx(H, out_csv=None)
        assert res_h["dilation_exists"] is False
        assert len(res_h["drivers"]) == 5

    def test_negative_example(self):
        """结构可控的反面例子：存在 dilation，12 个驱动结点。"""
        pytest.importorskip("HAT")

        ne_edges = {
            "e1": ["C1", "R"],
            "e2": ["C2", "O"],
            "e3": ["C3", "G"],
            "e4": ["R", "T"],
            "e5": ["O", "B"],
            "e6": ["T", "B", "L1", "L2"],
            "e7": ["G", "K", "DP"],
        }
        H = hnx.Hypergraph(ne_edges)
        res_h = compute_drivers_from_hypernetx(H, out_csv=None)
        assert res_h["dilation_exists"] is True
        assert len(res_h["drivers"]) == 12
        assert res_h.keys() == {"algorithm", "dilation_exists", "drivers"}
