"""稳态分析、可控性度量与带状态约束最优控制的测试。

本文件覆盖参考 nctpy 引入的三块能力：

1. **矩阵归一化与无限时域 Gramian**——Lyapunov 方程求解，含稳定性判定；
2. **可控性度量**——平均可控性与模态可控性（Gu et al. 2015）；
3. **带状态约束的最优控制**——Hamiltonian 状态-协态法。

关键验证策略：**能用解析解验证的绝不用「跑通即通过」**。例如对角系统
``A = diag(a_i)`` 的无限时域 Gramian 有闭式解 ``W_ii = -1/(2a_i)``，直接对照。
"""

from __future__ import annotations

import numpy as np
import pytest

from honet.control import (
    average_controllability,
    control_energy_ranking,
    gramian_auto,
    infinite_gramian,
    is_stable,
    matrix_normalization,
    modal_controllability,
    optimal_control,
)
from honet.control._control_law import minimum_energy_control
from honet.control._controllability import gramian


class TestMatrixNormalization:
    """矩阵归一化。"""

    def test_continuous_is_stable(self):
        """连续时间归一化后系统必须稳定（谱实部全负）。"""
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        A_norm = matrix_normalization(A, "continuous")
        assert is_stable(A_norm, "continuous")

    def test_discrete_spectral_radius_below_one(self):
        """离散时间归一化后谱半径必须 < 1。"""
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        A_norm = matrix_normalization(A, "discrete")
        assert np.abs(np.linalg.eigvals(A_norm)).max() < 1.0

    def test_normalization_formula(self):
        """归一化公式为 ``A / (c + λmax)``，连续时间再减单位阵。"""
        A = np.diag([2.0, 1.0])
        spectral = 2.0
        expected_disc = A / (1.0 + spectral)
        assert np.allclose(matrix_normalization(A, "discrete"), expected_disc)
        expected_cont = expected_disc - np.eye(2)
        assert np.allclose(matrix_normalization(A, "continuous"), expected_cont)

    def test_c_parameter(self):
        """``c`` 越大，归一化后谱半径越小。"""
        A = np.diag([2.0, 1.0])
        small = np.abs(np.linalg.eigvals(matrix_normalization(A, "discrete", c=1.0))).max()
        large = np.abs(np.linalg.eigvals(matrix_normalization(A, "discrete", c=10.0))).max()
        assert large < small

    def test_invalid_system_raises(self):
        with pytest.raises(ValueError, match="system"):
            matrix_normalization(np.eye(2), "nonsense")

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="方阵"):
            matrix_normalization(np.zeros((2, 3)), "discrete")


class TestStability:
    """稳定性判定。"""

    def test_continuous_stable(self):
        assert is_stable(np.diag([-1.0, -2.0]), "continuous") is True

    def test_continuous_unstable(self):
        assert is_stable(np.diag([1.0, -2.0]), "continuous") is False

    def test_discrete_stable(self):
        assert is_stable(np.diag([0.5, 0.9]), "discrete") is True

    def test_discrete_unstable(self):
        assert is_stable(np.diag([0.5, 1.2]), "discrete") is False

    def test_invalid_system_raises(self):
        with pytest.raises(ValueError, match="system"):
            is_stable(np.eye(2), "hybrid")


class TestInfiniteGramian:
    """无限时域 Gramian：用 Lyapunov 方程，可用解析解精确验证。"""

    def test_diagonal_analytic_solution(self):
        """对角系统 ``A=diag(a)`` 的 ``W_ii = -1/(2a_i)``（连续时间）。

        这是连续 Lyapunov 方程 ``AW + WAᵀ + BBᵀ = 0`` 在对角情形下的闭式解，
        可直接逐元素对照。
        """
        a = np.array([-1.0, -2.0, -4.0])
        A = np.diag(a)
        W = infinite_gramian(A, np.eye(3), "continuous")
        expected = np.diag(-1.0 / (2.0 * a))
        assert np.allclose(W, expected, atol=1e-14)

    def test_discrete_diagonal_analytic(self):
        """离散时间对角系统：``W_ii = 1/(1-a_i²)``。"""
        a = np.array([0.5, 0.2])
        A = np.diag(a)
        W = infinite_gramian(A, np.eye(2), "discrete")
        expected = np.diag(1.0 / (1.0 - a**2))
        assert np.allclose(W, expected, atol=1e-14)

    def test_gramian_is_symmetric_psd(self):
        """Gramian 必须对称半正定。"""
        rng = np.random.default_rng(3)
        raw = rng.normal(size=(5, 5))
        A = (raw + raw.T) / 2 - 3.0 * np.eye(5)
        W = infinite_gramian(A, np.eye(5), "continuous")
        assert np.allclose(W, W.T, atol=1e-12)
        assert np.linalg.eigvalsh(W).min() > -1e-12

    def test_unstable_system_raises(self):
        """不稳定系统的无限时域 Gramian 不存在，应报错而非返回 nan。"""
        with pytest.raises(ValueError, match="不稳定"):
            infinite_gramian(np.diag([1.0, -1.0]), np.eye(2), "continuous")

    def test_gramian_auto_selects_infinite(self):
        """``T=None`` 时走无限时域分支。"""
        A = np.diag([-1.0, -2.0])
        W_inf = gramian_auto(A, np.eye(2), None, "continuous")
        W_direct = infinite_gramian(A, np.eye(2), "continuous")
        assert np.allclose(W_inf, W_direct)

    def test_gramian_auto_selects_finite(self):
        """给定 ``T`` 时走有限时域分支，与 gramian 一致。"""
        A = np.diag([-1.0, -2.0])
        W = gramian_auto(A, np.eye(2), 1.0, "continuous")
        assert np.allclose(W, gramian(A, np.eye(2), 1.0))

    def test_finite_approaches_infinite(self):
        """``T`` 很大时，有限时域 Gramian 应趋近无限时域。"""
        A = np.diag([-1.0, -2.0])
        W_finite = gramian(A, np.eye(2), 30.0)
        W_infinite = infinite_gramian(A, np.eye(2), "continuous")
        assert np.allclose(W_finite, W_infinite, rtol=1e-10)

    def test_discrete_finite_gramian_auto(self):
        """离散时间有限时域的 ``gramian_auto`` 与逐项累加一致。"""
        A = np.diag([0.5, 0.2])
        B = np.eye(2)
        T = 5
        expected = np.zeros((2, 2))
        power = np.eye(2)
        for _ in range(T):
            expected += power @ B @ B.T @ power.T
            power = power @ A
        assert np.allclose(gramian_auto(A, B, T, "discrete"), expected)


class TestAverageControllability:
    """平均可控性（Gu et al. 2015）。"""

    def test_continuous_analytic(self):
        """对角系统：``ac_i = -1/(2a_i)``。"""
        A = np.diag([-1.0, -2.0, -4.0])
        ac = average_controllability(A, "continuous")
        assert np.allclose(ac, -1.0 / (2.0 * np.diag(A)))

    def test_continuous_equals_gramian_diagonal(self):
        """连续时间下平均可控性等于无限 Gramian 的对角元。"""
        rng = np.random.default_rng(5)
        raw = rng.normal(size=(4, 4))
        A = (raw + raw.T) / 2 - 3.0 * np.eye(4)
        ac = average_controllability(A, "continuous")
        W = infinite_gramian(A, np.eye(4), "continuous")
        assert np.allclose(ac, np.diag(W))

    def test_discrete_is_finite(self):
        """离散时间的结果必须全部有限（分母 1-λ² > 0）。"""
        rng = np.random.default_rng(9)
        raw = rng.normal(size=(4, 4))
        A = (raw + raw.T) / 2
        A = A / (np.abs(np.linalg.eigvals(A)).max() + 1.0)
        ac = average_controllability(A, "discrete")
        assert np.all(np.isfinite(ac))

    def test_unstable_raises(self):
        with pytest.raises(ValueError, match="不稳定"):
            average_controllability(np.diag([2.0, -1.0]), "continuous")

    def test_invalid_system_raises(self):
        with pytest.raises(ValueError, match="system"):
            average_controllability(np.eye(2), "continuous-discrete")


class TestModalControllability:
    """模态可控性。"""

    def test_shape_and_finite(self):
        rng = np.random.default_rng(13)
        raw = rng.normal(size=(5, 5))
        A = raw / (np.abs(np.linalg.eigvals(raw)).max() + 1.0)
        phi = modal_controllability(A)
        assert phi.shape == (5,)
        assert np.all(np.isfinite(phi))

    def test_non_negative(self):
        """所有项应非负（是 |U_ij|² 与 1-λ² 的非负加权和）。"""
        A = np.diag([0.5, 0.2, 0.1])
        phi = modal_controllability(A)
        assert np.all(phi >= 0)

    def test_diagonal_analytic(self):
        """对角系统：``φ_i = 1 - a_i²``。"""
        a = np.array([0.5, 0.2])
        phi = modal_controllability(np.diag(a))
        assert np.allclose(phi, 1.0 - a**2)

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="方阵"):
            modal_controllability(np.zeros((2, 3)))


class TestControlEnergyRanking:
    """结点排名。"""

    def test_sorted_descending(self):
        A = np.diag([-1.0, -3.0, -2.0])
        ranking = control_energy_ranking(A)
        values = [v for _, v in ranking]
        assert values == sorted(values, reverse=True)

    def test_most_controllable_first(self):
        """a 的绝对值越小，``-1/(2a)`` 越大，越该排前面。"""
        A = np.diag([-3.0, -1.0, -2.0])
        ranking = control_energy_ranking(A)
        assert ranking[0][0] == 1  # a=-1 对应下标 1

    def test_covers_all_nodes(self):
        A = np.diag([-1.0, -2.0, -3.0])
        ranking = control_energy_ranking(A)
        assert len(ranking) == 3
        assert {i for i, _ in ranking} == {0, 1, 2}


class TestOptimalControl:
    """带状态轨迹约束的最优控制。"""

    @staticmethod
    def _system() -> tuple:
        A = np.diag([-1.0, -2.0])
        B = np.eye(2)
        return A, B

    def test_reaches_target(self):
        """可控系统应能到达目标。"""
        A, B = self._system()
        result = optimal_control(A, B, [1.0, 1.0], [0.0, 0.0], 2.0, rho=1.0)
        assert result.final_error < 1e-2

    def test_initial_state_preserved(self):
        """轨迹首点必须等于给定初态。"""
        A, B = self._system()
        x0 = np.array([1.5, -0.5])
        result = optimal_control(A, B, x0, [0.0, 0.0], 2.0)
        assert np.allclose(result.states[0], x0)

    def test_boundary_error_small(self):
        """边值问题残差应接近 0（数值健康的标志）。"""
        A, B = self._system()
        result = optimal_control(A, B, [1.0, 1.0], [0.0, 0.0], 2.0)
        assert result.boundary_error < 1e-8

    def test_zero_constraint_matches_minimum_energy(self):
        """``S = 0`` 时与最小能量控制等价。

        两者都求解「给定端点、最小化 ∫‖u‖²dt」，因此能量应相同、轨迹应相同。
        注意必须对齐时间网格：``optimal_control`` 默认 ``dt=1e-3``（T=2 时
        2001 点），若拿 1001 点的 ``minimum_energy_control`` 直接按下标比较，
        会比到不同时刻的值而误判为不一致。
        """
        A, B = self._system()
        x0 = np.array([1.0, 2.0])
        target = np.zeros(2)
        T = 2.0

        constrained = optimal_control(A, B, x0, target, T, rho=1.0, S=np.zeros((2, 2)))
        minimal = minimum_energy_control(A, B, x0, target, T, num_points=len(constrained.times))

        assert len(constrained.times) == len(minimal.times)
        assert np.allclose(constrained.states, minimal.states, atol=1e-6)
        assert np.allclose(constrained.inputs, minimal.inputs, atol=1e-6)
        # 能量应等于解析值 reachᵀW⁻¹reach
        assert constrained.energy == pytest.approx(minimal.energy, rel=1e-6)

    def test_zero_constraint_energy_is_optimal(self):
        """``S=0`` 时的能量应等于解析最小能量 ``reachᵀW⁻¹reach``。"""
        from scipy.linalg import expm

        from honet.control._controllability import gramian

        A, B = self._system()
        x0 = np.array([1.0, 2.0])
        target = np.zeros(2)
        T = 2.0

        result = optimal_control(A, B, x0, target, T, rho=1.0, S=np.zeros((2, 2)))
        W = gramian(A, B, T)
        reach = target - expm(A * T) @ x0
        analytic = float(reach @ np.linalg.solve(W, reach))
        assert result.energy == pytest.approx(analytic, rel=1e-4)

    def test_rho_affects_energy(self):
        """``rho`` 越小，控制能量权重越低，可用更大输入换取更贴合的状态。"""
        A, B = self._system()
        low = optimal_control(A, B, [1.0, 1.0], [0.0, 0.0], 2.0, rho=0.1)
        high = optimal_control(A, B, [1.0, 1.0], [0.0, 0.0], 2.0, rho=10.0)
        # rho 小 -> 允许大输入 -> 能量通常更大
        assert low.energy >= high.energy

    def test_reference_state_changes_trajectory(self):
        """参考状态 ``xr`` 会影响轨迹（S 非零时）。"""
        A, B = self._system()
        zero_ref = optimal_control(A, B, [1.0, 1.0], [0.0, 0.0], 2.0, xr="zero")
        mid_ref = optimal_control(A, B, [1.0, 1.0], [0.0, 0.0], 2.0, xr="midpoint")
        assert not np.allclose(zero_ref.states, mid_ref.states)

    def test_invalid_rho_raises(self):
        A, B = self._system()
        with pytest.raises(ValueError, match="rho"):
            optimal_control(A, B, [1.0, 1.0], [0.0, 0.0], 2.0, rho=0.0)

    def test_invalid_system_raises(self):
        A, B = self._system()
        with pytest.raises(ValueError, match="system"):
            optimal_control(A, B, [1.0, 1.0], [0.0, 0.0], 2.0, system="semi-discrete")

    def test_dimension_mismatch_raises(self):
        A, _ = self._system()
        with pytest.raises(ValueError, match="x0"):
            optimal_control(A, np.eye(2), [1.0, 2.0, 3.0], [0.0, 0.0], 2.0)

    def test_invalid_xr_raises(self):
        A, B = self._system()
        with pytest.raises(ValueError, match="xr"):
            optimal_control(A, B, [1.0, 1.0], [0.0, 0.0], 2.0, xr="nowhere")

    def test_discrete_reaches_target(self):
        """离散时间系统也能到达目标。"""
        A = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
        B = np.eye(3)
        result = optimal_control(A, B, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 5, system="discrete")
        assert result.final_error < 1e-10
        assert result.states.shape == (5, 3)

    def test_discrete_too_short_raises(self):
        """离散时间 T < 2 时报错。"""
        A = np.diag([0.5, 0.3])
        with pytest.raises(ValueError, match="T 必须"):
            optimal_control(A, np.eye(2), [1.0, 1.0], [0.0, 0.0], 1, system="discrete")

    def test_converged_property(self):
        """``converged`` 反映终点误差是否达标。"""
        A, B = self._system()
        result = optimal_control(A, B, [1.0, 1.0], [0.0, 0.0], 2.0)
        assert result.converged is True
