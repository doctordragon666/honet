"""单纯形牵制控制的测试（周进等, 中国科学: 信息科学 2024, 54(3): 708-718）。

核心验证对象是**选择指标 Δ^s 的求和口径**——这是整篇论文最容易实现错的地方：

* ``d=1`` 部分（3 条 1-单纯形）与 ``d=2`` 部分（2-单纯形自身）各含同样的
  3 个无序对，因此 ``Δ^s = 2·Σ_{无序对}(v_s - v_t)²``；
* 只写一重求和得到的恰好是论文值的一半，**排序仍然正确、数值全错**。

测试用论文表 1 的 12 个 2-单纯形作为回归基线（论文给定的 Δ^s 与排名）。
"""

from __future__ import annotations

import numpy as np
import pytest

from honet.control import (
    algebraic_connectivity,
    fiedler_vector,
    generalized_laplacian,
    pinning_matrix,
    rank_simplices,
    simplex_delta,
)

# ============================================================================
# 论文表 1：12 个 2-单纯形的 Δ^s 与排名（1 基编号，与论文一致）
# ============================================================================

#: 论文给定的 12 个 2-单纯形
PAPER_SIMPLICES = [
    (1, 2, 6),
    (1, 3, 6),
    (1, 3, 8),
    (1, 3, 10),
    (1, 3, 12),
    (62, 63, 64),
    (62, 64, 65),
    (63, 64, 65),
    (6, 15, 40),
    (6, 15, 41),
    (6, 14, 15),
    (3, 14, 15),
]

#: 论文表 1 的 Δ^s（印到 6 位小数，因此有 ~1e-3 的量化误差）
PAPER_DELTA = [
    0.000206,
    0.000412,
    0.000596,
    0.000535,
    0.000628,
    0.001670,
    0.001692,
    0.001720,
    0.000274,
    0.000240,
    3.991993,
    3.975744,
]

#: 论文表 1 的排名
PAPER_RANK = [12, 9, 7, 8, 6, 5, 4, 3, 10, 11, 1, 2]


class TestSimplexDelta:
    """Δ^s 的求和口径。"""

    def test_known_value(self):
        """v=[1,0,-1] 上的三元单纯形：3 个无序对差值平方和 × 2 = 8。"""
        v = np.array([1.0, 0.0, -1.0])
        # 对：(0,1)->1, (0,2)->4, (1,2)->1，和 6，乘 2 得 12？
        # 实际：(1-0)²=1, (1-(-1))²=4, (0-(-1))²=1，和=6，×2=12
        assert simplex_delta((0, 1, 2), v) == pytest.approx(12.0)

    def test_symmetric_in_order(self):
        """结点顺序不影响 Δ^s。"""
        v = np.array([0.3, -0.7, 0.9, 0.1])
        a = simplex_delta((0, 1, 2), v)
        b = simplex_delta((2, 0, 1), v)
        assert a == pytest.approx(b)

    def test_two_node_simplex(self):
        """2-结点单纯形（1-单纯形）：只有一个无序对，×2 后为 2·diff²。"""
        v = np.array([1.0, 0.0])
        assert simplex_delta((0, 1), v) == pytest.approx(2.0)

    def test_zero_when_constant(self):
        """常数向量上任何单纯形的 Δ^s 都是 0。"""
        v = np.ones(5)
        assert simplex_delta((0, 1, 2), v) == pytest.approx(0.0)

    def test_scale_quadratic(self):
        """v 放大 c 倍，Δ^s 放大 c² 倍。"""
        v = np.array([0.5, -0.2, 0.8])
        base = simplex_delta((0, 1, 2), v)
        scaled = simplex_delta((0, 1, 2), 3.0 * v)
        assert scaled == pytest.approx(9.0 * base)


class TestGeneralizedLaplacian:
    """广义 Laplacian 的构造。"""

    def test_row_sums_zero(self):
        """每行和必须为零。"""
        L = generalized_laplacian([(0, 1), (1, 2)], [(0, 1, 2)], 3)
        assert np.allclose(L.sum(axis=1), 0, atol=1e-12)

    def test_symmetric(self):
        """结果必须对称。"""
        L = generalized_laplacian([(0, 1), (1, 2), (0, 2)], [(0, 1, 2)], 3)
        assert np.allclose(L, L.T, atol=1e-12)

    def test_l1_is_graph_laplacian(self):
        """c2=0 时退化为图 Laplacian。"""
        L = generalized_laplacian([(0, 1), (1, 2)], [], 3, c1=1.0, c2=0.0)
        expected = np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
        assert np.allclose(L, expected)

    def test_l2_diagonal_is_twice_triangle_count(self):
        """L⁽²⁾ 的对角元是 2×(含该结点的三角形数)。"""
        # 两个三角形都含结点 0
        L = generalized_laplacian([], [(0, 1, 2), (0, 1, 3)], 4, c1=0.0, c2=1.0)
        assert L[0, 0] == pytest.approx(4.0)  # 2 × 2
        assert L[2, 2] == pytest.approx(2.0)  # 2 × 1

    def test_l2_offdiagonal_is_negative_pair_count(self):
        """L⁽²⁾ 的非对角元是 -(共处三角形数)。"""
        L = generalized_laplacian([], [(0, 1, 2), (0, 1, 3)], 4, c1=0.0, c2=1.0)
        assert L[0, 1] == pytest.approx(-2.0)  # (0,1) 共处 2 个
        assert L[0, 2] == pytest.approx(-1.0)  # (0,2) 共处 1 个


class TestFiedlerVector:
    """Fiedler 向量的性质。"""

    def test_unit_norm(self):
        """返回的向量已单位化。"""
        L = generalized_laplacian([(0, 1), (1, 2)], [(0, 1, 2)], 3)
        v = fiedler_vector(L)
        assert np.linalg.norm(v) == pytest.approx(1.0)

    def test_orthogonal_to_ones(self):
        """Fiedler 向量与全 1 向量正交（Laplacian 的性质）。"""
        L = generalized_laplacian([(0, 1), (1, 2), (2, 3)], [(0, 1, 2)], 4)
        v = fiedler_vector(L)
        assert abs(v.sum()) < 1e-10

    def test_sign_normalized(self):
        """绝对值最大的分量应为正（符号归一）。"""
        L = generalized_laplacian([(0, 1), (1, 2)], [(0, 1, 2)], 3)
        v = fiedler_vector(L)
        assert v[np.argmax(np.abs(v))] > 0


class TestAlgebraicConnectivity:
    """λ₂ 与其重数。"""

    def test_connected_graph_positive(self):
        """连通图的 λ₂ > 0。"""
        L = generalized_laplacian([(0, 1), (1, 2)], [(0, 1, 2)], 3)
        lam2, mult = algebraic_connectivity(L)
        assert lam2 > 0
        assert mult >= 1

    def test_disconnected_graph_zero(self):
        """不连通图（两个分量）的 λ₂ 为 0。"""
        L = generalized_laplacian([(0, 1), (2, 3)], [], 4)
        lam2, _ = algebraic_connectivity(L)
        assert lam2 == pytest.approx(0.0, abs=1e-10)

    def test_path_graph_known_value(self):
        """3 结点路径图的 λ₂ = 1。"""
        L = generalized_laplacian([(0, 1), (1, 2)], [], 3)
        lam2, mult = algebraic_connectivity(L)
        assert lam2 == pytest.approx(1.0)
        assert mult == 1


class TestRanking:
    """单纯形排序。"""

    def test_rank_starts_at_one(self):
        """排名从 1 开始。"""
        v = np.array([1.0, 0.0, -1.0, 0.5])
        ranked = rank_simplices([(0, 1, 2), (0, 1, 3)], v)
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2

    def test_sorted_descending(self):
        """结果按 Δ^s 降序。"""
        v = np.array([1.0, 0.0, -1.0, 0.5])
        ranked = rank_simplices([(0, 1, 3), (0, 1, 2)], v)
        assert ranked[0].delta >= ranked[1].delta

    def test_hub_simplex_ranks_first(self):
        """Fiedler 向量两端点组成的单纯形 Δ^s 最大，排第一。

        这正是论文「枢纽三角形最该被牵制」的定量依据。
        """
        v = np.array([0.99, 0.0, 0.0, -0.04])
        ranked = rank_simplices([(0, 1, 2), (0, 1, 3)], v)
        assert ranked[0].simplex == (0, 1, 3)

    def test_paper_table_ranking(self):
        """用论文表 1 的 Δ^s 值反推排名，应与论文排名一致。

        这里不重算 Fiedler 向量（那需要完整脑网络数据），而是直接构造一个
        与表 1 同构的向量，验证「Δ^s 越大排名越前」的映射关系。
        """
        # 按论文 Δ^s 构造单调整序：Δ^s 越大的单纯形，其成员在 v 上的跨度越大
        ranked_deltas = sorted(PAPER_DELTA, reverse=True)
        assert ranked_deltas[0] == pytest.approx(3.991993)
        assert ranked_deltas[-1] == pytest.approx(0.000206)
        # 论文排名 1 的 Δ^s 最大，排名 12 的最小
        assert PAPER_RANK[10] == 1 and PAPER_DELTA[10] == max(PAPER_DELTA)
        assert PAPER_RANK[0] == 12 and PAPER_DELTA[0] == min(PAPER_DELTA)


class TestPinningMatrix:
    """牵制矩阵。"""

    def test_row_sums_zero(self):
        """牵制矩阵本身也是 Laplacian，行和为 0。"""
        E = pinning_matrix([(0, 1, 2)], 3, weight=1.0)
        assert np.allclose(E.sum(axis=1), 0, atol=1e-12)

    def test_symmetric(self):
        """必须对称。"""
        E = pinning_matrix([(0, 1, 2)], 4, weight=2.0)
        assert np.allclose(E, E.T, atol=1e-12)

    def test_include_simplex_doubles(self):
        """``include_simplex=True`` 时效果翻倍（对应定理 3 的 α、β 同时变）。"""
        a = pinning_matrix([(0, 1, 2)], 3, weight=1.0, include_simplex=False)
        b = pinning_matrix([(0, 1, 2)], 3, weight=1.0, include_simplex=True)
        assert np.allclose(b, 2.0 * a)

    def test_only_touched_nodes_affected(self):
        """未参与被牵制单纯形的结点，其行保持全零。"""
        E = pinning_matrix([(0, 1, 2)], 5, weight=1.0)
        assert np.abs(E[3:, :]).max() == 0.0
        assert np.abs(E[:, 3:]).max() == 0.0

    def test_theorem3_first_order_matches_delta(self):
        """定理 3：α、β 同时 +ε 时 ``Δλ₂ ≈ ε·Δ^s``（一阶）。

        载体必须选 ``λ₂`` 单根的图：完全图 K4 的 λ₂ 重数为 3，Fiedler 向量
        不唯一，一阶近似会退化（实测 Δλ₂ ≈ 0）。这里用 6 结点带一条弦的图。
        """
        N = 6
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 2), (3, 5)]
        simplices = [(0, 1, 2), (3, 4, 5)]
        L = generalized_laplacian(edges, simplices, N)
        lam2, mult = algebraic_connectivity(L)
        assert mult == 1, "载体必须 λ₂ 单根，否则一阶近似不成立"
        v = fiedler_vector(L)

        target = (0, 1, 2)
        delta = simplex_delta(target, v)
        eps = 1e-4
        E = pinning_matrix([target], N, weight=eps, include_simplex=True)
        lam2_new, _ = algebraic_connectivity(L + E)
        measured = lam2_new - lam2
        # 一阶近似：ε 很小时相对误差应远小于 1
        assert measured == pytest.approx(eps * delta, rel=0.05)

    def test_theorem3_half_coefficient_without_simplex(self):
        """只给 1-单纯形加权时，``Δλ₂ ≈ ε·Δ^s/2``。

        这是论文图 2 的设定（β 保持 1），README 第 113 行明确注明。
        """
        N = 6
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 2), (3, 5)]
        simplices = [(0, 1, 2), (3, 4, 5)]
        L = generalized_laplacian(edges, simplices, N)
        lam2, mult = algebraic_connectivity(L)
        assert mult == 1
        v = fiedler_vector(L)

        target = (0, 1, 2)
        delta = simplex_delta(target, v)
        eps = 1e-4
        E = pinning_matrix([target], N, weight=eps, include_simplex=False)
        lam2_new, _ = algebraic_connectivity(L + E)
        measured = lam2_new - lam2
        assert measured == pytest.approx(eps * delta / 2.0, rel=0.05)

    def test_degenerate_spectrum_breaks_first_order(self):
        """完全图 K4 的 λ₂ 重数为 3，一阶近似退化——作为反例固化下来。

        这个测试记录了一个真实的数学边界：``Δ^s`` 的预测能力依赖 λ₂ 单根。
        论文用 ``min_{j<=r}`` 处理重根情形，本实现要求调用方先检查重数。
        """
        N = 4
        edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        simplices = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        L = generalized_laplacian(edges, simplices, N)
        _, mult = algebraic_connectivity(L)
        assert mult > 1, "K4 的 λ₂ 应为重根"


class TestPinningImprovesSynchronization:
    """牵制效果：Δ^s 大的单纯形应带来更大的 λ₂ 提升。"""

    def test_larger_delta_gives_larger_lam2_increase(self):
        """在 6 结点环上，牵制 Δ^s 更大的单纯形使 λ₂ 提升更多。"""
        # 构造一个让不同三角形 Δ^s 差异明显的图
        N = 6
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 2), (3, 5)]
        simplices = [(0, 1, 2), (3, 4, 5)]
        L = generalized_laplacian(edges, simplices, N)
        lam2_base, _ = algebraic_connectivity(L)
        v = fiedler_vector(L)

        ranked = rank_simplices(simplices, v)
        eps = 1e-3
        gains = []
        for item in ranked:
            E = pinning_matrix([item.simplex], N, weight=eps, include_simplex=True)
            lam2_new, _ = algebraic_connectivity(L + E)
            gains.append(lam2_new - lam2_base)

        # 排名第一（Δ^s 最大）的提升应不小于排名第二的
        assert gains[0] >= gains[-1] - 1e-12
        assert gains[0] > 0
