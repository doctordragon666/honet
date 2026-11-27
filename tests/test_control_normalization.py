"""能量积分、归一化与统计检验的测试（参考 nctpy）。

这四块能力的共同特点是**容易被写成「看起来合理但偏离参考」的形式**，
因此每个测试都尽量锚定一个可独立验证的事实：

* 能量积分——常数信号的积分有闭式解 ``∫c²dt = c²T``；
* 快速最小能量——与精确 Gramian 解对照；
* 权重归一化——与 nctpy 的公式逐位一致（结果落在 ``[1, 2]``）；
* 统计检验——FDR 的 BH 步进法有确定的手算结果。
"""

from __future__ import annotations

import numpy as np
import pytest

from honet.control import (
    fdr_p_values,
    integrate_control_energy,
    label_to_state,
    minimum_energy_exact,
    minimum_energy_fast,
    normalize_state,
    normalize_weights,
    null_p_value,
)


class TestIntegrateControlEnergy:
    """控制能量积分。"""

    def test_constant_signal(self):
        """常数信号 ``u(t)=c`` 在 ``[0,T]`` 上的积分为 ``c²T``。"""
        steps, dt = 101, 0.01
        U = np.ones((steps, 2)) * 2.0
        energy = integrate_control_energy(U, dt=dt)
        # 实际时间跨度是 (steps-1)*dt = 1.0
        expected = 4.0 * (steps - 1) * dt
        assert np.allclose(energy, expected, rtol=1e-10)

    def test_zero_signal(self):
        """零输入的能量为 0。"""
        energy = integrate_control_energy(np.zeros((50, 3)), dt=0.1)
        assert np.allclose(energy, 0.0)

    def test_per_channel(self):
        """不同通道的能量独立计算。"""
        U = np.zeros((101, 2))
        U[:, 0] = 1.0
        U[:, 1] = 2.0
        energy = integrate_control_energy(U, dt=0.01)
        assert energy[1] == pytest.approx(4.0 * energy[0], rel=1e-10)

    def test_sine_wave(self):
        """正弦信号的积分：``∫₀^T sin²(t)dt ≈ T/2``。"""
        T = 2.0 * np.pi
        steps = 2001
        t = np.linspace(0, T, steps)
        U = np.sin(t).reshape(-1, 1)
        energy = integrate_control_energy(U, dt=T / (steps - 1))
        assert energy[0] == pytest.approx(T / 2.0, rel=1e-6)

    def test_short_signal_fallback(self):
        """点数少于 3 时退化为梯形法，仍能给出正确结果。"""
        energy = integrate_control_energy(np.ones((2, 1)), dt=1.0)
        assert energy[0] == pytest.approx(1.0)


class TestMinimumEnergyFast:
    """快速最小能量（Simpson 型 Gramian）。"""

    @staticmethod
    def _setup():
        A = np.diag([-1.0, -2.0, -3.0])
        B = np.eye(3)
        x0 = np.array([1.0, 2.0, 3.0])
        xf = np.zeros(3)
        return A, B, x0, xf, 2.0

    def test_close_to_exact(self):
        """与精确 Gramian 解应接近（相对误差 < 1e-2）。"""
        A, B, x0, xf, T = self._setup()
        fast = minimum_energy_fast(A, B, x0, xf, T, steps=1000)
        exact = minimum_energy_exact(A, B, x0, xf, T)
        assert fast == pytest.approx(exact, rel=1e-2)

    def test_converges_with_more_steps(self):
        """步数增加时快速解应更接近精确解。"""
        A, B, x0, xf, T = self._setup()
        exact = minimum_energy_exact(A, B, x0, xf, T)
        coarse = abs(minimum_energy_fast(A, B, x0, xf, T, steps=20) - exact)
        fine = abs(minimum_energy_fast(A, B, x0, xf, T, steps=1000) - exact)
        assert fine < coarse

    def test_positive(self):
        """能量必须为正。"""
        A, B, x0, xf, T = self._setup()
        assert minimum_energy_fast(A, B, x0, xf, T) > 0

    def test_odd_steps_handled(self):
        """奇数次步数会被上调为偶数，不应报错。"""
        A, B, x0, xf, T = self._setup()
        value = minimum_energy_fast(A, B, x0, xf, T, steps=101)
        assert np.isfinite(value)

    def test_exact_matches_analytic(self):
        """精确解与 ``reachᵀW⁻¹reach`` 一致。"""
        from scipy.linalg import expm

        from honet.control import gramian

        A, B, x0, xf, T = self._setup()
        W = gramian(A, B, T)
        reach = xf - expm(A * T) @ x0
        analytic = float(reach @ np.linalg.solve(W, reach))
        assert minimum_energy_exact(A, B, x0, xf, T) == pytest.approx(analytic, rel=1e-12)


class TestNormalizeState:
    """状态归一化。"""

    def test_unit_norm(self):
        """归一化后范数为 1。"""
        v = normalize_state(np.array([3.0, 4.0]))
        assert np.allclose(v, [0.6, 0.8])
        assert np.linalg.norm(v) == pytest.approx(1.0)

    def test_zero_vector_unchanged(self):
        """零向量原样返回（避免除零）。"""
        v = normalize_state(np.zeros(3))
        assert np.allclose(v, 0.0)

    def test_matrix_columns(self):
        """矩阵按列归一化。"""
        m = np.array([[3.0, 0.0], [4.0, 2.0]])
        result = normalize_state(m)
        assert np.linalg.norm(result[:, 0]) == pytest.approx(1.0)
        assert np.linalg.norm(result[:, 1]) == pytest.approx(1.0)


class TestNormalizeWeights:
    """控制权重归一化。"""

    def test_matches_nctpy_formula(self):
        """与 nctpy 公式一致：秩变换 → 缩放到 [0,1] → +1，结果落在 [1, 2]。

        nctpy 的 ``add_constant`` 是**整体加 1**，不是缩放到 (0,1]。
        实测 nctpy 对 ``[10,20,30,40]`` 给出 ``[1, 4/3, 5/3, 2]``。
        """
        w = normalize_weights(np.array([10.0, 20.0, 30.0, 40.0]))
        assert np.allclose(w, [1.0, 1.0 + 1 / 3, 1.0 + 2 / 3, 2.0])

    def test_range(self):
        """结果落在 [1, 2]。"""
        w = normalize_weights(np.array([5.0, 1.0, 9.0, 3.0]))
        assert w.min() >= 1.0 - 1e-12
        assert w.max() <= 2.0 + 1e-12

    def test_no_rank(self):
        """``rank=False`` 时直接用原值缩放。"""
        w = normalize_weights(np.array([0.0, 5.0, 10.0]), rank=False)
        assert np.allclose(w, [1.0, 1.5, 2.0])

    def test_no_constant(self):
        """``add_constant=False`` 时结果落在 [0, 1]。"""
        w = normalize_weights(np.array([0.0, 5.0, 10.0]), rank=False, add_constant=False)
        assert np.allclose(w, [0.0, 0.5, 1.0])

    def test_constant_input(self):
        """全相同的输入不应除零。"""
        w = normalize_weights(np.array([7.0, 7.0, 7.0]))
        assert np.all(np.isfinite(w))

    def test_empty(self):
        """空输入原样返回。"""
        assert normalize_weights(np.array([])).size == 0

    def test_rank_handles_ties(self):
        """并列值取平均秩（与 scipy.stats.rankdata 一致）。"""
        w = normalize_weights(np.array([1.0, 1.0, 3.0]), add_constant=False)
        # rankdata 给出 [1.5, 1.5, 3] -> 缩放到 [0, 0, 1]
        assert np.allclose(w, [0.0, 0.0, 1.0])


class TestNullPValue:
    """零分布 p 值。"""

    def test_all_below(self):
        """观测值大于所有零分布样本时，p 取加一平滑的下界。"""
        p = null_p_value(100.0, np.arange(10, dtype=float))
        assert p == pytest.approx(1.0 / 11.0)

    def test_all_above(self):
        """观测值小于所有样本时 p 接近 1。"""
        p = null_p_value(-100.0, np.arange(10, dtype=float))
        assert p == pytest.approx(1.0)

    def test_midpoint(self):
        """观测值在分布中点附近时 p 约为 0.5。"""
        samples = np.arange(99, dtype=float)
        p = null_p_value(49.0, samples)
        assert p == pytest.approx(0.5, abs=0.02)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="不能为空"):
            null_p_value(1.0, np.array([]))


class TestFdrPValues:
    """Benjamini-Hochberg FDR 校正。"""

    def test_basic_adjustment(self):
        """手算可验证的例子：p=[0.001,0.02,0.5]，m=3。

        升序后：0.001*3/1=0.003，0.02*3/2=0.03，0.5*3/3=0.5。
        累积最小值（从后往前）：[0.003, 0.03, 0.5]。
        """
        adjusted, rejected = fdr_p_values(np.array([0.001, 0.02, 0.5]))
        assert np.allclose(adjusted, [0.003, 0.03, 0.5])
        assert rejected.tolist() == [True, True, False]

    def test_monotone(self):
        """校正后的 p 值必须单调不减（与原始排序一致）。"""
        ps = np.array([0.5, 0.001, 0.2, 0.04])
        adjusted, _ = fdr_p_values(ps)
        order = np.argsort(ps)
        assert np.all(np.diff(adjusted[order]) >= -1e-12)

    def test_capped_at_one(self):
        """校正后不应超过 1。"""
        adjusted, _ = fdr_p_values(np.array([0.9, 0.95, 0.99]))
        assert np.all(adjusted <= 1.0)

    def test_alpha_threshold(self):
        """alpha 变大时被拒绝的假设不会减少。"""
        ps = np.array([0.01, 0.03, 0.06, 0.2])
        _, strict = fdr_p_values(ps, alpha=0.01)
        _, loose = fdr_p_values(ps, alpha=0.1)
        assert np.sum(loose) >= np.sum(strict)

    def test_empty(self):
        adjusted, rejected = fdr_p_values(np.array([]))
        assert adjusted.size == 0
        assert rejected.size == 0


class TestLabelToState:
    """标签转整数编码。"""

    def test_basic(self):
        codes, names = label_to_state(["B", "A", "B"])
        assert names == ["A", "B"]
        assert codes.tolist() == [1, 0, 1]

    def test_single_label(self):
        codes, names = label_to_state(["X", "X"])
        assert names == ["X"]
        assert codes.tolist() == [0, 0]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="不能为空"):
            label_to_state([])
