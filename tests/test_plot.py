"""honet.plot 子包的单元测试。

覆盖点：

* 12 个公开绘图函数（含旧名别名 ``draw_sis_conmmunity``）与 2 个私有子绘图函数
  都能在真实数据上跑通，且返回正确的 ``Axes`` / ``(Figure, Axes)`` 对象；
* 传入 ``ax`` 时函数就在该坐标轴上绘制（不新建画布），不传时自行新建；
* 事件字典的状态回放（``'I'`` / ``'S'`` / ``'R'``）与按社团计数结果正确；
* 惰性导出：``import honet.plot`` 不会拉入 matplotlib / numpy。

全部用例使用 5 个结点的小规模真实数据，实际调用每个函数，不做 mock。
``matplotlib`` 强制使用 Agg 后端，任何用例都不会弹出窗口；每个用例结束用
``plt.close("all")`` 清理画布。

运行方式：``uv run pytest tests/test_plot.py -q``。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import hypernetx as hnx  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402 - 必须在 use("Agg") 之后导入
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from honet.attack import degree_attack_sequence, execute_attack  # noqa: E402
from honet.plot import (  # noqa: E402
    _sub_draw_sir_community,
    _sub_draw_sis_community,
    draw_attack_results,
    draw_communities_onmap,
    draw_community_partition,
    draw_control_partition,
    draw_hnxs_global,
    draw_sir,
    draw_sis,
    draw_sis_community,
    draw_sis_conmmunity,
    draw_sis_init,
    draw_sis_init_beta,
    draw_sis_phase,
    draw_test_assortativity,
)


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    """每个用例结束后关闭全部画布，避免用例之间互相影响。"""
    yield
    plt.close("all")


@pytest.fixture()
def hg() -> hnx.Hypergraph:
    """5 结点 4 超边的真实小超图。

    拓扑：e1={a,b,c}、e2={b,c}、e3={c,d}、e4={d,e}。
    """
    return hnx.Hypergraph(
        {"e1": ["a", "b", "c"], "e2": ["b", "c"], "e3": ["c", "d"], "e4": ["d", "e"]}
    )


@pytest.fixture()
def communities() -> list[list[str]]:
    """把上面 5 个结点分成两个社团：{a,b,c} 与 {d,e}。"""
    return [["a", "b", "c"], ["d", "e"]]


@pytest.fixture()
def sis_events(hg: hnx.Hypergraph) -> dict:
    """真实跑一遍 SIS 得到的事件字典（``{时间: [('I', 结点, 超边), ...]}``）。

    直接调用 hypernetx 的 ``discrete_SIS``：``honet.plot`` 消费的是这一层的
    事件格式，测试里不必再绕经 ``honet.contagion`` 的门面函数。
    """
    return hnx.discrete_SIS(
        hg,
        {2: 0.4, 3: 0.5},
        0.3,
        rho=0.2,
        tmin=0,
        tmax=5,
        dt=1,
        return_full_data=True,
    )


@pytest.fixture()
def sir_events(hg: hnx.Hypergraph) -> dict:
    """真实跑一遍 SIR 得到的事件字典（含 ``'R'`` 事件）。"""
    return hnx.discrete_SIR(
        hg,
        {2: 0.4, 3: 0.5},
        0.3,
        rho=0.2,
        tmin=0,
        tmax=5,
        dt=1,
        return_full_data=True,
    )


class TestDrawSis:
    """``draw_sis``。"""

    def test_plots_two_curves_and_mean_line(self) -> None:
        """画出 S、I 两条曲线与一条感染者均值参考线。"""
        t = np.arange(5)
        susceptible = np.array([10, 8, 6, 5, 4], dtype=float)
        infected = np.array([0, 2, 4, 5, 6], dtype=float)

        ax = draw_sis(susceptible, infected, t)

        assert isinstance(ax, Axes)
        # 两条曲线 + 一条 axhline
        assert len(ax.lines) == 3
        labels = [line.get_label() for line in ax.lines]
        assert r"$S$" in labels
        assert r"$I$" in labels
        # 均值参考线标签里带均值数值（I 的均值为 3.4）
        assert any("3.40" in label for label in labels)

    def test_uses_given_axes(self) -> None:
        """传入 ax 时在同一个坐标轴上绘制，不新建画布。"""
        _, ax = plt.subplots()
        t = np.arange(4)
        returned = draw_sis(t, t, t, ax=ax)
        assert returned is ax
        assert len(plt.get_fignums()) == 1

    def test_title_is_applied(self) -> None:
        """非空标题会写到坐标轴上。"""
        t = np.arange(3)
        ax = draw_sis(t, t, t, title="SIS 曲线")
        assert ax.get_title() == "SIS 曲线"


class TestDrawSir:
    """``draw_sir``。"""

    def test_plots_three_curves(self) -> None:
        """画出易感 / 感染 / 康复三条曲线加均值线。"""
        t = np.arange(5)
        susceptible = np.array([10, 7, 5, 4, 4], dtype=float)
        infected = np.array([0, 3, 4, 3, 2], dtype=float)
        recovered = np.array([0, 0, 1, 3, 4], dtype=float)

        ax = draw_sir(susceptible, infected, recovered, t)

        assert isinstance(ax, Axes)
        assert len(ax.lines) == 4
        labels = [line.get_label() for line in ax.lines]
        assert {"易感", "感染", "康复"} <= set(labels)

    def test_uses_given_axes(self) -> None:
        """传入 ax 时复用该坐标轴。"""
        _, ax = plt.subplots()
        t = np.arange(4)
        assert draw_sir(t, t, t, t, ax=ax) is ax


class TestDrawSisPhase:
    """``draw_sis_phase``。"""

    def test_returns_figure_and_axes(self) -> None:
        """返回 ``(Figure, Axes)``，并把两种相态的文字都标注出来。"""
        phase_map = np.array([[0, 0, 1], [0, 1, 1]])

        figure, ax = draw_sis_phase(phase_map, (0.0, 1.0), (0.0, 2.0))

        assert isinstance(figure, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.images) == 1
        texts = {text.get_text() for text in ax.texts}
        assert {"爆发感染", "无感染"} <= texts
        # extent 按传入边界设置
        assert ax.images[0].get_extent() == [0.0, 1.0, 0.0, 2.0]

    def test_single_phase_still_labelled(self) -> None:
        """全 0 矩阵只有一种相态，也要标出文字且不报错。"""
        phase_map = np.zeros((2, 3), dtype=int)
        _, ax = draw_sis_phase(phase_map, (0.0, 1.0), (0.0, 1.0))
        assert [text.get_text() for text in ax.texts] == ["爆发感染"]

    def test_uses_given_axes(self) -> None:
        """传入 ax 时复用该坐标轴。"""
        _, ax = plt.subplots()
        figure, returned = draw_sis_phase(np.ones((2, 2), dtype=int), (0, 1), (0, 1), ax=ax)
        assert returned is ax
        assert figure is ax.figure

    def test_rejects_one_dimensional_input(self) -> None:
        """一维相图数据抛 ValueError。"""
        with pytest.raises(ValueError, match="二维"):
            draw_sis_phase(np.zeros(3, dtype=int), (0, 1), (0, 1))


class TestDrawSisInit:
    """``draw_sis_init``。"""

    def test_one_line_per_initial_size(self) -> None:
        """每个初始感染规模一条曲线，图例标签即字典的键。"""
        t = np.arange(4)
        result = {
            1: (np.array([4, 3, 2, 2]), np.array([0, 1, 2, 2]), t),
            2: (np.array([3, 2, 1, 1]), np.array([1, 2, 3, 3]), t),
        }

        ax = draw_sis_init(result)

        assert isinstance(ax, Axes)
        assert len(ax.lines) == 2
        assert {text.get_text() for text in ax.get_legend().get_texts()} == {"1", "2"}

    def test_uses_given_axes(self) -> None:
        """传入 ax 时复用该坐标轴。"""
        _, ax = plt.subplots()
        t = np.arange(3)
        assert draw_sis_init({0: (t, t, t)}, ax=ax) is ax


class TestDrawSisInitBeta:
    """``draw_sis_init_beta``。"""

    def test_two_panels_from_long_form_table(self) -> None:
        """长表输入拆成「峰值」「时间」两个子图，每个指标各一条曲线。"""
        result = {
            "指标": ["峰值", "峰值", "时间", "时间"],
            "峰值": [10, 20, 0, 0],
            "时间": [0, 0, 3, 4],
            "P": [0.1, 0.2, 0.1, 0.2],
        }

        peak_ax, time_ax = draw_sis_init_beta(result)

        assert isinstance(peak_ax, Axes) and isinstance(time_ax, Axes)
        assert peak_ax is not time_ax
        # 「峰值」列画进第一个子图，「时间」列画进第二个子图
        assert any("峰值" in line.get_label() for line in peak_ax.lines)
        assert any("时间" in line.get_label() for line in time_ax.lines)

    def test_dataframe_input(self) -> None:
        """DataFrame 输入同样可用。"""
        frame = pd.DataFrame({"P": [0.1, 0.2, 0.3], "峰值": [5.0, 9.0, 7.0]})

        peak_ax, time_ax = draw_sis_init_beta(frame)

        assert any("峰值" in line.get_label() for line in peak_ax.lines)
        # 没有「时间」列时，第二个子图只给出提示文字
        assert "无" in time_ax.get_title()

    def test_uses_given_axes(self) -> None:
        """传入两个坐标轴时复用它们。"""
        figure, (first, second) = plt.subplots(1, 2)
        frame = pd.DataFrame({"P": [0.1, 0.2], "峰值": [1.0, 2.0], "时间": [3.0, 4.0]})

        peak_ax, time_ax = draw_sis_init_beta(frame, ax=[first, second])

        assert peak_ax is first
        assert time_ax is second
        assert peak_ax.figure is figure

    def test_missing_p_column(self) -> None:
        """缺少 P 列时抛 ValueError。"""
        with pytest.raises(ValueError, match="'P'"):
            draw_sis_init_beta({"峰值": [1.0, 2.0]})

    def test_missing_metric_columns(self) -> None:
        """既无「峰值」也无「时间」列时抛 ValueError。"""
        with pytest.raises(ValueError, match="峰值"):
            draw_sis_init_beta({"P": [0.1, 0.2], "其他": [1, 2]})


class TestSubDrawCommunity:
    """两个私有子绘图函数。"""

    def test_sis_counts_by_community(self, sis_events: dict, communities: list[list[str]]) -> None:
        """SIS 事件回放后，各社团的感染数曲线长度与时间步数一致。"""
        ax = _sub_draw_sis_community(sis_events, communities)

        assert isinstance(ax, Axes)
        assert len(ax.lines) == len(communities)
        # 每个时间步一个点
        assert len(ax.lines[0].get_xdata()) == len(sis_events)

    def test_sir_counts_by_community(self, sir_events: dict, communities: list[list[str]]) -> None:
        """SIR 事件里的 'R' 会让结点退出感染态，曲线非负且长度正确。"""
        ax = _sub_draw_sir_community(sir_events, communities)

        assert len(ax.lines) == len(communities)
        for line in ax.lines:
            counts = np.asarray(line.get_ydata(), dtype=float)
            assert len(counts) == len(sir_events)
            assert (counts >= 0).all()

    def test_sis_rejects_numeric_tuple(self, communities: list[list[str]]) -> None:
        """传数值统计元组而非事件字典时抛 RuntimeError。"""
        with pytest.raises(RuntimeError, match="return_all_data"):
            _sub_draw_sis_community(([0, 1], [5, 4], [1, 2]), communities)

    def test_sir_rejects_numeric_tuple(self, communities: list[list[str]]) -> None:
        """SIR 版本同样拒绝数值统计元组（旧实现在此会抛 ValueError）。"""
        with pytest.raises(RuntimeError, match="return_all_data"):
            _sub_draw_sir_community(([0, 1], [5, 4], [1, 2], [0, 0]), communities)

    def test_rejects_non_dict(self, communities: list[list[str]]) -> None:
        """既不是字典也不是序列时抛 TypeError。"""
        with pytest.raises(TypeError, match="事件字典"):
            _sub_draw_sis_community(42, communities)


class TestDrawSisCommunity:
    """``draw_sis_community`` 与旧名别名 ``draw_sis_conmmunity``。"""

    def test_alias_points_to_same_function(self) -> None:
        """旧名别名与规范名指向同一个函数对象。"""
        assert draw_sis_conmmunity is draw_sis_community

    def test_one_panel_per_partition(self, sis_events: dict) -> None:
        """两套社团划分各占一个面板，每个面板画出该套划分的全部社团曲线。"""
        partitions = [[["a", "b", "c"], ["d", "e"]], [["a", "e"], ["b", "c", "d"]]]
        result = {0: sis_events, 1: sis_events}

        axes = draw_sis_community(result, partitions, title="SIS")

        assert len(axes) == 2
        assert all(isinstance(ax, Axes) for ax in axes)
        # 每套划分各 2 个社团 → 每个面板 2 条曲线
        assert [len(ax.lines) for ax in axes] == [2, 2]
        # 面板标题带字母序号与社团编号
        assert axes[0].get_title() == "(a) SIS (社团0)"
        assert axes[1].get_title() == "(b) SIS (社团1)"
        # 曲线点数与时间步数一致
        assert all(len(line.get_xdata()) == len(sis_events) for line in axes[0].lines)

    def test_single_community_partition_accepted(self, sis_events: dict) -> None:
        """传单个社团的结点列表时按「单社团划分」处理，画 1 条曲线。"""
        axes = draw_sis_community({0: sis_events}, [["a", "b", "c", "d", "e"]])

        assert len(axes) == 1
        assert len(axes[0].lines) == 1

    def test_uses_given_axes(self, sis_events: dict) -> None:
        """传入坐标轴序列时复用它们。"""
        _, grid = plt.subplots(1, 2)
        partitions = [[["a", "b"], ["c", "d", "e"]]] * 2

        axes = draw_sis_community({0: sis_events, 1: sis_events}, partitions, ax=grid)

        assert axes == list(grid)

    def test_rejects_too_many_panels(self, sis_events: dict) -> None:
        """超过 4 个面板（2×2）时抛 ValueError。"""
        with pytest.raises(ValueError, match="4"):
            draw_sis_community({0: sis_events}, [["a"], ["b"], ["c"], ["d"], ["e"]])


class TestDrawPartitions:
    """``draw_community_partition`` 与 ``draw_control_partition``。"""

    def test_community_partition_wedges(self, communities: list[list[str]]) -> None:
        """社团规模饼图：扇区数等于社团数。"""
        ax = draw_community_partition(communities, title="社团规模")

        assert isinstance(ax, Axes)
        assert len(ax.patches) == len(communities)
        assert ax.get_title() == "社团规模"

    def test_community_partition_uses_given_axes(self, communities: list[list[str]]) -> None:
        """传入 ax 时复用该坐标轴。"""
        _, ax = plt.subplots()
        assert draw_community_partition(communities, ax=ax) is ax

    def test_community_partition_rejects_empty(self) -> None:
        """空划分抛 ValueError。"""
        with pytest.raises(ValueError, match="不能为空"):
            draw_community_partition([])

    def test_control_partition_counts_drivers(self, communities: list[list[str]]) -> None:
        """控制节点按社团计数：a 属于社团0、d 属于社团1。"""
        ax = draw_control_partition(["a", "d"], communities, title="控制节点")

        assert len(ax.patches) == 2
        assert ax.get_title() == "控制节点"
        # 两个社团各 1 个驱动节点 → 各占 50.0%
        percentages = [text.get_text() for text in ax.texts if "%" in text.get_text()]
        assert percentages == ["50.0%", "50.0%"]

    def test_control_partition_without_drivers(self, communities: list[list[str]]) -> None:
        """没有控制节点落入任何社团时给出提示文字而不是画空饼图。"""
        ax = draw_control_partition([], communities)

        assert len(ax.patches) == 0
        assert any("无控制节点" in text.get_text() for text in ax.texts)

    def test_control_partition_rejects_empty_nodemap(self) -> None:
        """空划分抛 ValueError。"""
        with pytest.raises(ValueError, match="不能为空"):
            draw_control_partition(["a"], [])


class TestDrawHnxsGlobal:
    """``draw_hnxs_global``。"""

    def test_normalises_each_column(self) -> None:
        """每列做 min-max 归一化，画出与列数相同的曲线。"""
        metrics = pd.DataFrame(
            {"平均度": [1.0, 2.0, 3.0], "聚类系数": [0.2, 0.5, 0.8]},
            index=[2019, 2020, 2021],
        )

        ax = draw_hnxs_global(metrics)

        assert len(ax.lines) == 2
        # 归一化后每列的最小值为 0、最大值为 1
        for line in ax.lines:
            values = np.asarray(line.get_ydata(), dtype=float)
            assert values.min() == pytest.approx(0.0)
            assert values.max() == pytest.approx(1.0)
        assert ax.get_title() == "超网络特征变化趋势"

    def test_constant_column_filled_with_one(self) -> None:
        """最大最小相等的列（除零风险）按旧实现口径填 1。"""
        metrics = pd.DataFrame({"常数": [5.0, 5.0, 5.0]}, index=[1, 2, 3])

        ax = draw_hnxs_global(metrics, title="趋势")

        values = np.asarray(ax.lines[0].get_ydata(), dtype=float)
        assert (values == 1.0).all()
        assert ax.get_title() == "趋势"

    def test_uses_given_axes(self) -> None:
        """传入 ax 时复用该坐标轴（旧实现会另建一张图）。"""
        _, ax = plt.subplots()
        metrics = pd.DataFrame({"平均度": [1.0, 2.0]}, index=[2019, 2020])
        assert draw_hnxs_global(metrics, ax=ax) is ax


class TestDrawTestAssortativity:
    """``draw_test_assortativity``。"""

    def test_scatter_points(self) -> None:
        """散点数量等于字典条目数。"""
        result = {1: 2.0, 2: 2.5, 3: 1.0}

        ax = draw_test_assortativity(result)

        assert len(ax.collections) == 1
        offsets = np.asarray(ax.collections[0].get_offsets())
        assert offsets.shape == (3, 2)
        assert sorted(offsets[:, 0]) == [1.0, 2.0, 3.0]

    def test_scatter_kwargs_override(self) -> None:
        """scatter_kwargs 覆盖默认样式。"""
        ax = draw_test_assortativity({1: 1.0, 2: 2.0}, scatter_kwargs={"s": 100})
        assert ax.collections[0].get_sizes()[0] == 100

    def test_uses_given_axes(self) -> None:
        """传入 ax 时复用该坐标轴。"""
        _, ax = plt.subplots()
        assert draw_test_assortativity({1: 1.0}, ax=ax) is ax


class TestDrawAttackResults:
    """``draw_attack_results``。"""

    def test_curves_from_real_attack(self, hg: hnx.Hypergraph) -> None:
        """用真实攻击结果绘图：每条策略一条曲线，点数为步骤数。"""
        results = execute_attack(hg, degree_attack_sequence(hg), rho=0.4)
        results["介数"] = execute_attack(hg, degree_attack_sequence(hg, target="excess"), rho=0.4)[
            "默认策略"
        ]

        ax = draw_attack_results(results, title="攻击")

        assert len(ax.lines) == 2
        assert ax.get_title() == "攻击"
        for line in ax.lines:
            assert len(line.get_xdata()) == len(line.get_ydata())
        assert {text.get_text() for text in ax.get_legend().get_texts()} == {"默认策略", "介数"}

    def test_uses_given_axes(self) -> None:
        """传入 ax 时复用该坐标轴。"""
        _, ax = plt.subplots()
        results = {"策略": {"steps": [0.0, 0.5], "nc_vals": [1.0, 0.4]}}
        assert draw_attack_results(results, ax=ax) is ax

    def test_missing_keys(self) -> None:
        """结果缺少 steps / nc_vals 时抛 ValueError。"""
        with pytest.raises(ValueError, match="steps"):
            draw_attack_results({"策略": {"nc_vals": [1.0]}})


@pytest.mark.skipif(
    importlib.util.find_spec("geopandas") is None,
    reason="未安装 geopandas（honet[viz] 之外的可选依赖），跳过地图绘制",
)
class TestDrawCommunitiesOnmapWithGeopandas:
    """``draw_communities_onmap``：geopandas 可用时的真实绘制。"""

    def test_merges_communities(self) -> None:
        """按 ISO_A3_EH 合并社团列，并剔除南极洲。"""
        gpd = pytest.importorskip("geopandas")
        shapely = pytest.importorskip("shapely.geometry")

        world = gpd.GeoDataFrame(
            {
                "ISO_A3_EH": ["CHN", "USA", "ATA"],
                "NAME": ["China", "United States", "Antarctica"],
                "geometry": [
                    shapely.box(0, 0, 1, 1),
                    shapely.box(2, 0, 3, 1),
                    shapely.box(0, -5, 1, -4),
                ],
            },
            crs="EPSG:4326",
        )

        merged = draw_communities_onmap({"CHN": 0, "USA": 1}, world, title="社团地图")

        assert "community" in merged.columns
        assert "Antarctica" not in set(merged["NAME"])
        assert merged.loc[merged["NAME"] == "China", "community"].iloc[0] == 0


@pytest.mark.skipif(
    importlib.util.find_spec("geopandas") is not None,
    reason="已安装 geopandas，改由 TestDrawCommunitiesOnmapWithGeopandas 覆盖",
)
class TestDrawCommunitiesOnmapWithoutGeopandas:
    """``draw_communities_onmap``：缺 geopandas 时给出可执行的安装提示。"""

    def test_import_error_mentions_viz_extra(self) -> None:
        """缺依赖时抛 ImportError，提示指向 honet[viz]。"""
        with pytest.raises(ImportError, match="viz"):
            draw_communities_onmap({"CHN": 0}, object())


class TestLazyExport:
    """惰性导出契约。"""

    def test_import_does_not_pull_third_party(self) -> None:
        """``import honet.plot`` 本身不加载 matplotlib / numpy / geopandas。"""
        code = (
            "import sys, honet.plot; "
            "print([m for m in ('matplotlib', 'numpy', 'geopandas') if m in sys.modules])"
        )
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "[]"

    def test_exports_sorted_and_complete(self) -> None:
        """``__all__`` 覆盖 13 个公开名字且按字母序（不含下划线开头的辅助函数）。"""
        import honet.plot as plot

        expected = {
            "draw_attack_results",
            "draw_communities_onmap",
            "draw_community_partition",
            "draw_control_partition",
            "draw_hnxs_global",
            "draw_sir",
            "draw_sis",
            "draw_sis_community",
            "draw_sis_conmmunity",
            "draw_sis_init",
            "draw_sis_init_beta",
            "draw_sis_phase",
            "draw_test_assortativity",
        }
        assert set(plot.__all__) == expected
        assert plot.__all__ == sorted(plot.__all__)

    def test_private_helpers_accessible(self) -> None:
        """两个私有子绘图函数可按名访问，但不在 ``__all__`` 里。"""
        import honet.plot as plot

        assert callable(plot._sub_draw_sis_community)
        assert callable(plot._sub_draw_sir_community)
        assert "_sub_draw_sis_community" not in plot.__all__

    def test_alias_survives_private_helper_import(self) -> None:
        """先访问私有辅助函数、再访问旧名别名，别名仍是可调用函数。

        这是一处真实的惰性遮蔽陷阱：``import honet.plot._sub_draw_sis_community``
        会把模块对象写回包的 ``_sub_draw_sis_community`` 属性；别名解析若不走
        规范名，就会被同名模块遮蔽。
        """
        import honet.plot as plot

        assert callable(plot._sub_draw_sis_community)
        assert callable(plot.draw_sis_conmmunity)
        assert plot.draw_sis_conmmunity is plot.draw_sis_community

    def test_unknown_attribute(self) -> None:
        """访问不存在的符号抛 AttributeError。"""
        import honet.plot as plot

        with pytest.raises(AttributeError, match="no attribute"):
            _ = plot.no_such_function
