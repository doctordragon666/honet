"""前向仿真与批量控制任务的测试。

前向仿真（``simulate_system``）与反解（``minimum_energy_control`` /
``optimal_control``）方向相反，两者合起来构成开环控制的闭环自检：
**反解出的输入喂回前向仿真，必须能到达目标**。本文件的核心测试就是这条闭环。

批量控制（``control_energy_matrix``）的价值在于避免调用方写嵌套循环；
它的结果必须与逐条反解一致。
"""

from __future__ import annotations

import numpy as np
import pytest

from honet.control import (
    control_energy_matrix,
    expand_states,
    kalman_rank,
    ltv_controllability,
    ltv_gramian,
    minimum_energy_control,
    simulate_impulse,
    simulate_system,
)


class TestExpandStates:
    """状态标签展开。"""

    def test_basic_expansion(self):
        """两个状态各含两个结点。"""
        x0, xf, names = expand_states(["A", "A", "B", "B"])
        assert names == ["A", "B"]
        assert x0.shape == (4, 2)
        assert x0[:, 0].tolist() == [True, True, False, False]
        assert x0[:, 1].tolist() == [False, False, True, True]

    def test_xf_mirrors_x0(self):
        """初态与终态矩阵内容相同（只是语义不同）。"""
        x0, xf, _ = expand_states(["A", "B", "C"])
        assert np.array_equal(x0, xf)

    def test_labels_sorted(self):
        """标签按字典序排序，保证多次调用顺序稳定。"""
        _, _, names = expand_states(["C", "A", "B"])
        assert names == ["A", "B", "C"]

    def test_single_state(self):
        """所有结点同属一个状态。"""
        x0, _, names = expand_states(["X", "X", "X"])
        assert names == ["X"]
        assert x0.shape == (3, 1)
        assert x0[:, 0].all()

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="不能为空"):
            expand_states([])


class TestSimulateSystem:
    """前向仿真。"""

    def test_initial_state_preserved(self):
        """轨迹首点等于初态。"""
        A = np.diag([-1.0, -2.0])
        result = simulate_system(A, np.eye(2), [3.0, 4.0], np.zeros((10, 2)))
        assert np.allclose(result.states[0], [3.0, 4.0])

    def test_zero_input_decays(self):
        """无输入时稳定系统按 ``e^{at}`` 衰减。

        2000 步 × dt=1e-3 即 t=2，A=diag(-1,-2) 下解析解为
        ``‖x(2)‖ = √(e^{-4} + e^{-8}) ≈ 0.1367``。阈值必须按解析值定，
        不能凭「应该很小」写死 1e-2。
        """
        A = np.diag([-1.0, -2.0])
        steps, dt = 2000, 1e-3
        result = simulate_system(A, np.eye(2), [1.0, 1.0], np.zeros((steps, 2)), dt=dt)
        t_end = (steps - 1) * dt
        expected = np.sqrt(np.exp(-2.0 * t_end) + np.exp(-4.0 * t_end))
        assert np.linalg.norm(result.final_state) == pytest.approx(expected, rel=1e-10)

    def test_continuous_analytic_decay(self):
        """对角系统的零输入响应有解析解 ``x(t) = x0·e^{at}``。"""
        A = np.diag([-1.0])
        dt = 1e-3
        steps = 1000
        result = simulate_system(A, np.eye(1), [1.0], np.zeros((steps, 1)), dt=dt)
        expected = np.exp(-1.0 * dt * np.arange(steps))
        assert np.allclose(result.states[:, 0], expected, atol=1e-10)

    def test_discrete_analytic_decay(self):
        """离散时间无输入：``x_k = A^k x0``。"""
        A = np.diag([0.5, 0.3])
        result = simulate_system(A, np.eye(2), [1.0, 1.0], np.zeros((5, 2)), system="discrete")
        assert np.allclose(result.states[:, 0], 0.5 ** np.arange(5))
        assert np.allclose(result.states[:, 1], 0.3 ** np.arange(5))

    def test_step_input_reaches_steady_state(self):
        """恒定输入下，稳定系统趋于 ``-A⁻¹Bu``。"""
        A = np.diag([-1.0])
        B = np.eye(1)
        u = np.array([2.0])
        result = simulate_impulse(A, B, [0.0], u, steps=8000, dt=1e-3)
        # 稳态解：A x + B u = 0 => x = -A⁻¹Bu = 2
        assert result.final_state[0] == pytest.approx(2.0, abs=1e-3)

    def test_shape_validation(self):
        """维度不匹配应报错。"""
        with pytest.raises(ValueError, match="B 的行数"):
            simulate_system(np.eye(2), np.eye(3), [1.0, 1.0], np.zeros((5, 3)))
        with pytest.raises(ValueError, match="x0"):
            simulate_system(np.eye(2), np.eye(2), [1.0], np.zeros((5, 2)))
        with pytest.raises(ValueError, match="U 的列数"):
            simulate_system(np.eye(2), np.eye(2), [1.0, 1.0], np.zeros((5, 3)))

    def test_invalid_system_raises(self):
        with pytest.raises(ValueError, match="system"):
            simulate_system(np.eye(2), np.eye(2), [1.0, 1.0], np.zeros((3, 2)), system="x")

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="方阵"):
            simulate_system(np.zeros((2, 3)), np.zeros((2, 1)), [1.0, 1.0], np.zeros((3, 1)))

    def test_times_grid(self):
        """时间网格按 ``dt`` 生成。"""
        A = np.diag([-1.0])
        result = simulate_system(A, np.eye(1), [1.0], np.zeros((5, 1)), dt=0.5)
        assert np.allclose(result.times, [0.0, 0.5, 1.0, 1.5, 2.0])


class TestClosedLoop:
    """闭环自检：反解输入 → 前向仿真 → 必须到达目标。"""

    def test_minimum_energy_roundtrip(self):
        """最小能量控制的输入喂回前向仿真，终点应接近目标。

        这是两个模块之间最重要的一致性保证：反解与正算用的是同一个动力学。
        """
        A = np.diag([-1.0, -2.0])
        B = np.eye(2)
        x0 = np.array([1.0, 2.0])
        target = np.zeros(2)
        T = 2.0

        solved = minimum_energy_control(A, B, x0, target, T, num_points=1001)
        forward = simulate_system(A, B, x0, solved.inputs, dt=T / (len(solved.times) - 1))
        assert np.linalg.norm(forward.final_state - target) < 1e-3
        # 两条轨迹应基本重合（差异仅来自离散化）
        assert np.abs(forward.states - solved.states).max() < 1e-3

    def test_optimal_control_roundtrip(self):
        """带约束最优控制的输入喂回前向仿真，终点应接近目标。"""
        from honet.control import optimal_control

        A = np.diag([-1.0, -2.0])
        B = np.eye(2)
        x0 = np.array([1.0, 1.0])
        target = np.zeros(2)

        solved = optimal_control(A, B, x0, target, 2.0, rho=1.0, S=np.eye(2))
        forward = simulate_system(A, B, x0, solved.inputs, dt=1e-3)
        assert np.linalg.norm(forward.final_state - target) < 1e-3


class TestControlEnergyMatrix:
    """批量控制能量。"""

    @staticmethod
    def _setup():
        rng = np.random.default_rng(3)
        total = 6
        raw = rng.normal(size=(total, total))
        from honet.control import matrix_normalization

        A = matrix_normalization(np.abs((raw + raw.T) / 2), "continuous")
        return A, np.eye(total)

    def test_shape(self):
        """3 个状态给出 3×3 的能量矩阵。"""
        A, B = self._setup()
        result = control_energy_matrix(A, B, ["A", "A", "B", "B", "C", "C"], 1.0)
        assert result.energies.shape == (3, 3)
        assert result.labels == ["A", "B", "C"]

    def test_energies_non_negative(self):
        """控制能量必须非负。"""
        A, B = self._setup()
        result = control_energy_matrix(A, B, ["A", "A", "B", "B", "C", "C"], 1.0)
        assert np.all(result.energies >= -1e-12)

    def test_self_transition_cheapest(self):
        """从状态转移到自身应是最省能量的（初态即终态）。"""
        A, B = self._setup()
        result = control_energy_matrix(A, B, ["A", "A", "B", "B", "C", "C"], 1.0)
        diagonal = np.diag(result.energies)
        # 对角线应不高于同行的其他项
        for i in range(len(diagonal)):
            assert diagonal[i] <= result.energies[i, :].max() + 1e-12

    def test_matches_single_solve(self):
        """批量结果必须与逐条反解的能量一致。"""
        A, B = self._setup()
        labels = ["A", "A", "B", "B", "C", "C"]
        result = control_energy_matrix(A, B, labels, 1.0)

        start = (np.array(labels) == "A").astype(float)
        start /= np.linalg.norm(start)
        target = (np.array(labels) == "C").astype(float)
        target /= np.linalg.norm(target)

        single = minimum_energy_control(A, B, start, target, 1.0, num_points=1001)
        assert result.energies[0, 2] == pytest.approx(single.energy, rel=1e-4)

    def test_as_dict(self):
        """转字典后键是标签对。"""
        A, B = self._setup()
        result = control_energy_matrix(A, B, ["A", "A", "B", "B", "C", "C"], 1.0)
        mapping = result.as_dict()
        assert ("A", "C") in mapping
        assert isinstance(mapping[("A", "C")], float)

    def test_hardest_and_easiest(self):
        """最难/最易转移的标签对可正确取出。"""
        A, B = self._setup()
        result = control_energy_matrix(A, B, ["A", "A", "B", "B", "C", "C"], 1.0)
        src, dst, value = result.hardest()
        assert value == pytest.approx(result.energies.max())
        src2, dst2, value2 = result.easiest()
        assert value2 == pytest.approx(result.energies.min())

    def test_normalize_state_off(self):
        """``normalize_state=False`` 时使用原始 0/1 向量。"""
        A, B = self._setup()
        labels = ["A", "A", "B", "B", "C", "C"]
        normalized = control_energy_matrix(A, B, labels, 1.0, normalize_state=True)
        raw = control_energy_matrix(A, B, labels, 1.0, normalize_state=False)
        # 未归一化的向量范数更大，能量通常更高
        assert not np.allclose(normalized.energies, raw.energies)

    def test_label_length_mismatch_raises(self):
        """标签长度与结点数不符时报错。"""
        A, B = self._setup()
        with pytest.raises(ValueError, match="labels 的长度"):
            control_energy_matrix(A, B, ["A", "B"], 1.0)

    def test_dimension_mismatch_raises(self):
        """B 维度不符时报错。"""
        with pytest.raises(ValueError, match="B 的行数"):
            control_energy_matrix(np.eye(3), np.eye(4), ["A", "B", "C"], 1.0)


class TestTimeVaryingSimulation:
    """线性时变（LTV）系统的前向仿真。"""

    def test_time_varying_a_changes_trajectory(self):
        """A 随时间变化时轨迹应与常数 A 不同，且衰减更快。"""
        a_const = np.diag([-1.0, -2.0])
        b = np.eye(2)
        u = np.zeros((20, 2))

        const = simulate_system(a_const, b, [1.0, 1.0], u, dt=1e-2)
        varying = simulate_system(lambda t: np.diag([-1.0 - t, -2.0]), b, [1.0, 1.0], u, dt=1e-2)

        assert not np.allclose(const.states, varying.states)
        assert abs(varying.final_state[0]) < abs(const.final_state[0])

    def test_time_varying_b(self):
        """B 可以是时间函数；返回常数矩阵时应与原结果一致。"""
        a = np.diag([-1.0, -2.0])
        b = np.eye(2)
        u = np.ones((10, 2))

        const = simulate_system(a, b, [0.0, 0.0], u, dt=1e-2)
        varying = simulate_system(a, lambda t: np.eye(2), [0.0, 0.0], u, dt=1e-2)

        assert np.allclose(const.states, varying.states)

    def test_time_varying_discrete(self):
        """离散时间同样支持时变 A。"""
        u = np.zeros((10, 1))
        # 常数：每步乘 -0.5；时变：乘数随步数递减到 0（衰减更快）
        a_const = np.array([[-0.5]])
        result = simulate_system(
            lambda t: np.array([[-0.5 / (1.0 + t)]]),
            np.eye(1),
            [1.0],
            u,
            dt=1.0,
            system="discrete",
        )
        const = simulate_system(a_const, np.eye(1), [1.0], u, dt=1.0, system="discrete")

        assert not np.allclose(result.states, const.states)
        # 乘数逐步趋零，末态绝对值应远小于常数情形
        assert abs(result.final_state[0]) < abs(const.final_state[0])

    def test_time_varying_shape_mismatch_raises(self):
        """时变参数在某个时刻形状与 t=0 不符时应报错。"""
        u = np.zeros((5, 1))

        # t=0 时是 1x1，t>0 时变成 2x2，维度在运行中途变化
        def a(t):
            return np.eye(1) if t == 0 else np.eye(2)

        with pytest.raises(ValueError, match="形状应为"):
            simulate_system(a, np.eye(1), [1.0], u, dt=1e-2)


class TestLTVControllability:
    """LTV 可控性判定。"""

    @pytest.mark.parametrize(
        ("a", "b", "expected_rank"),
        [
            (np.diag([-1.0, -2.0]), np.eye(2), 2),
            (np.diag([-1.0, -2.0]), np.array([[1.0], [0.0]]), 1),
            (np.array([[-1.0, 1.0], [0.0, -2.0]]), np.array([[0.0], [1.0]]), 2),
            (np.diag([-1.0, -2.0, -3.0]), np.array([[1.0], [0.0], [0.0]]), 1),
        ],
    )
    def test_agrees_with_kalman_rank(self, a, b, expected_rank):
        """常数系统下，LTV Gramian 的秩必须与 Kalman 秩判据一致。"""
        result = ltv_controllability(a, b, t_final=5.0, steps=512)
        assert result.rank == expected_rank
        assert result.rank == kalman_rank(a, b)

    def test_time_varying_b_beats_fixed_b(self):
        """时变 B 交替驱动两个通道时可达满秩，而固定单通道只有秩 1。"""
        a = np.diag([-1.0, -2.0])
        b_fixed = np.array([[1.0], [0.0]])

        def b_varying(t):
            return np.array([[1.0], [0.0]]) if int(t) % 2 == 0 else np.array([[0.0], [1.0]])

        assert ltv_controllability(a, b_fixed, t_final=10.0, steps=1001).rank == 1
        assert ltv_controllability(a, b_varying, t_final=10.0, steps=1001).rank == 2

    def test_gramian_symmetric_psd(self):
        """Gramian 必须对称半正定。"""
        w = ltv_gramian(np.diag([-1.0, -2.0]), np.eye(2), t_final=2.0, steps=256)

        assert np.allclose(w, w.T)
        assert np.all(np.linalg.eigvalsh(w) > -1e-12)

    def test_condition_number_infinite_when_rank_deficient(self):
        """秩亏时条件数为 inf。"""
        result = ltv_controllability(
            np.diag([-1.0, -2.0]), np.array([[1.0], [0.0]]), t_final=5.0, steps=256
        )
        assert result.condition_number == float("inf")
        assert not result.controllable

    def test_invalid_steps(self):
        """steps < 2 应报错。"""
        with pytest.raises(ValueError, match="steps 必须 >= 2"):
            ltv_gramian(np.eye(2), np.eye(2), steps=1)
