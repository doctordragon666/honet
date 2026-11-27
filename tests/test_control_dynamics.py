"""超图网络控制框架的测试（Ma et al., Physica A 653 (2024) 130108）。

测试分四组：

1. **Laplacian 构造**——对照论文原文给定的 L⁽²⁾ / L⁽³⁾ 矩阵逐位验证；
2. **可控性判据**——Kalman 秩 / PBH / Gramian 三个判据交叉一致；
3. **控制律**——最小能量控制能把状态精确驱动到目标；
4. **论文四例**——Ex 2.1 / 2.2 / 3.1 / 3.2 的秩全部复现。

前两组是纯数学恒等式，不依赖论文；后两组把论文数值固化为回归基线。
"""

from __future__ import annotations

import numpy as np
import pytest

from honet.control import (
    HypergraphControlSystem,
    gramian,
    gramian_rank,
    hypergraph_laplacian,
    is_controllable,
    kalman_rank,
    minimum_energy_control,
    pbh_rank,
)

# ============================================================================
# 论文 Example 2.1 的超边定义与原文给定的 Laplacian
# ============================================================================

#: Ex 2.1：V(e) 与有向对 D(e)（结点用 1 基编号，与论文一致）
EDGES_21 = {
    "e1": ([1, 2, 3], [(1, 2), (2, 3)]),
    "e2": ([2, 3, 4], [(2, 3)]),
    "e3": ([3, 4, 5], [(3, 5), (4, 5)]),
}

#: 论文第 3 页原文给出的 L⁽²⁾
L2_PAPER = np.array(
    [
        [-0.5, 0.5, 0, 0, 0],
        [0, -1, 1, 0, 0],
        [0, 0, -0.5, 0, 0.5],
        [0, 0, 0, -0.5, 0.5],
        [0, 0, 0, 0, 0],
    ],
    dtype=float,
)

#: 论文第 3 页原文给出的 L⁽³⁾
L3_PAPER = np.array(
    [
        [-0.5, 0, 0.5, 0, 0],
        [0.5, -1, 0, 0.5, 0],
        [0, 0, -0.5, 0.5, 0],
        [0, 0, 0.5, -0.5, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=float,
)


class TestHypergraphLaplacian:
    """Laplacian 构造必须与论文原文逐位一致。"""

    def test_l2_matches_paper(self):
        """L⁽²⁾ 与论文第 3 页矩阵完全一致。"""
        mats = hypergraph_laplacian(EDGES_21, num_nodes=5)
        assert np.allclose(mats[2], L2_PAPER, atol=1e-12)

    def test_l3_matches_paper(self):
        """L⁽³⁾ 与论文第 3 页矩阵完全一致。

        这里最容易出错：m≥3 阶的项落在 ``(i, 第三点)`` 而不是 ``(i, j)``。
        """
        mats = hypergraph_laplacian(EDGES_21, num_nodes=5)
        assert np.allclose(mats[3], L3_PAPER, atol=1e-12)

    def test_row_sums_are_zero(self):
        """广义 Laplacian 的每行和为零（对角元由负行和定义）。"""
        mats = hypergraph_laplacian(EDGES_21, num_nodes=5)
        for mat in mats.values():
            assert np.allclose(mat.sum(axis=1), 0, atol=1e-12)

    def test_single_order_request(self):
        """请求单阶时只返回该阶矩阵，形状为 (N, N)。"""
        L2 = hypergraph_laplacian(EDGES_21, num_nodes=5, order=2)
        assert L2.shape == (5, 5)
        assert np.allclose(L2, L2_PAPER, atol=1e-12)

    def test_undirected_complete_pairs(self):
        """只给结点集时，全部有序对都当作有向对。"""
        # 单个三元超边，完全有向：每个有序对 (i,j) 都有一个第三点，故 L2[i,j]=1/2
        mats = hypergraph_laplacian({"e1": [0, 1, 2]}, num_nodes=3)
        assert np.allclose(mats[2][0, 1], 0.5)
        assert np.allclose(mats[2].sum(axis=1), 0)


class TestControllabilityCriteria:
    """三个判据必须互相一致。"""

    def test_kalman_rank_identity(self):
        """单位系统 + 置换输入：秩等于状态维数。"""
        A = np.eye(3)
        B = np.eye(3)
        assert kalman_rank(A, B) == 3

    def test_kalman_rank_insufficient_input(self):
        """只有一个输入通道时，秩不可能超过 1。"""
        A = np.eye(3)
        B = np.array([[1.0], [0.0], [0.0]])
        assert kalman_rank(A, B) == 1

    def test_pbh_agrees_with_kalman(self):
        """PBH 秩与 Kalman 秩在可控系统上一致。"""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.eye(2)
        assert pbh_rank(A, B) == kalman_rank(A, B) == 2

    def test_gramian_agrees_with_kalman(self):
        """Gramian 秩与 Kalman 秩在可控系统上一致。"""
        A = np.array([[0.0, 1.0], [-2.0, -3.0]])
        B = np.array([[0.0], [1.0]])
        assert gramian_rank(A, B, 5.0) == kalman_rank(A, B) == 2

    def test_gramian_is_symmetric_psd(self):
        """Gramian 必须对称半正定。"""
        A = np.array([[0.0, 1.0], [-2.0, -3.0]])
        B = np.array([[0.0], [1.0]])
        W = gramian(A, B, 3.0)
        assert np.allclose(W, W.T, atol=1e-12)
        assert np.linalg.eigvalsh(W).min() > -1e-12

    def test_is_controllable_negative(self):
        """输入矩阵秩不足时不可控。"""
        A = np.eye(2)
        B = np.array([[1.0], [0.0]])
        assert is_controllable(A, B) is False

    def test_dimension_mismatch_raises(self):
        """A 与 B 行数不匹配时报错。"""
        with pytest.raises(ValueError, match="维度不匹配"):
            kalman_rank(np.eye(3), np.eye(2))


class TestControlLaw:
    """最小能量控制律必须真正把状态驱动到目标。"""

    def test_steers_to_target(self):
        """可控系统上终点误差应远小于 1e-3。"""
        A = np.array([[0.0, 1.0], [-2.0, -3.0]])
        B = np.array([[0.0], [1.0]])
        result = minimum_energy_control(A, B, [1.0, 0.0], [0.0, 0.0], 5.0)
        assert result.final_error < 1e-3
        assert result.converged is True

    def test_trajectory_shapes(self):
        """轨迹各数组的行数与时间网格一致。"""
        A = np.eye(2)
        B = np.eye(2)
        result = minimum_energy_control(A, B, [1.0, 1.0], [0.0, 0.0], 2.0, num_points=50)
        assert result.times.shape == (50,)
        assert result.states.shape == (50, 2)
        assert result.inputs.shape[0] == 50

    def test_energy_is_positive(self):
        """控制能量必须为正（除非初态已是目标）。"""
        A = np.eye(2)
        B = np.eye(2)
        result = minimum_energy_control(A, B, [1.0, 2.0], [0.0, 0.0], 2.0)
        assert result.energy > 0

    def test_initial_state_preserved(self):
        """轨迹首点必须等于给定初态。"""
        A = np.eye(2)
        B = np.eye(2)
        x0 = [3.0, -1.0]
        result = minimum_energy_control(A, B, x0, [0.0, 0.0], 2.0)
        assert np.allclose(result.states[0], x0)

    def test_uncontrollable_raises_or_fails(self):
        """不可控系统的 Gramian 奇异，终点误差应当很大。"""
        A = np.eye(2)
        B = np.array([[1.0], [0.0]])  # 第二个状态永远无法影响
        result = minimum_energy_control(A, B, [0.0, 5.0], [0.0, 0.0], 2.0)
        # 伪逆会静默给出「尽力而为」的解，但终点误差必然不为零
        assert result.final_error > 1.0


class TestPaperExamples:
    """论文四个例子的秩必须全部复现。"""

    @staticmethod
    def _system_21() -> HypergraphControlSystem:
        mats = hypergraph_laplacian(EDGES_21, num_nodes=5)
        return HypergraphControlSystem(
            A=np.eye(2),
            L=mats[2] + mats[3],
            Lambda=np.eye(2),
            B=np.array([[0.0, 1.0], [1.0, 0.0]]),
            controlled_nodes=(1, 2, 4),  # 论文 1 基的 2, 3, 5
            coupling=1.0,
            order=3,
        )

    def test_example_2_1_rank_is_10(self):
        """Ex 2.1：超图网络可控，Kalman 秩 = 10（论文值）。"""
        Ac, Bc = self._system_21().to_blocks()
        assert kalman_rank(Ac, Bc) == 10

    def test_example_2_1_pbh_and_gramian(self):
        """Ex 2.1：PBH 与 Gramian 也应给出满秩 10。"""
        Ac, Bc = self._system_21().to_blocks()
        assert pbh_rank(Ac, Bc) == 10
        assert gramian_rank(Ac, Bc, 2.0) == 10

    def test_example_2_1_steering(self):
        """Ex 2.1：X(0)=[1..10] 能被驱动到原点（论文图 2）。"""
        Ac, Bc = self._system_21().to_blocks()
        result = minimum_energy_control(
            Ac, Bc, np.arange(1, 11, dtype=float), np.zeros(10), 2.0, num_points=1200
        )
        assert result.final_error < 1e-3

    def test_example_2_2_rank_is_9(self):
        """Ex 2.2：超图网络不可控，秩 = 9 < 12（论文值）。"""
        edges = {
            "e1": ([1, 2, 3], [(1, 2), (2, 3)]),
            "e2": ([2, 3, 4], [(2, 3), (2, 4), (3, 4)]),
            "e3": ([3, 5, 6], [(3, 5), (5, 6)]),
        }
        mats = hypergraph_laplacian(edges, num_nodes=6)
        sys = HypergraphControlSystem(
            A=np.array([[0.0, 1.0], [0.0, 0.0]]),
            L=mats[2] + mats[3],
            Lambda=np.diag([2.0, 0.0]),
            B=np.eye(2),
            controlled_nodes=(0, 3, 5),
            coupling=1.0,
            order=3,
        )
        Ac, Bc = sys.to_blocks()
        assert kalman_rank(Ac, Bc) == 9
        assert Ac.shape[0] == 12  # 满秩应为 12

    def test_example_3_1_simplicial_rank_is_8(self):
        """Ex 3.1：2-阶单纯复形可控，秩 = 8（论文值）。"""
        L1 = np.array([[2, -1, -1, 0], [-1, 2, -1, 0], [-1, -1, 3, -1], [0, 0, -1, 1]], dtype=float)
        L2 = np.array([[2, -1, -1, 0], [-1, 2, -1, 0], [-1, -1, 2, 0], [0, 0, 0, 0]], dtype=float)
        Ls = L1 + L2
        A = np.array([[1.0, 0.0], [0.0, 0.0]])
        # 单纯形约定：Ẋ = I⊗A - L⊗I
        Ac = np.kron(np.eye(4), A) - np.kron(Ls, np.eye(2))
        Bc = np.kron(np.diag([1.0, 1.0, 0.0, 1.0]), np.array([[1.0, 0.0], [1.0, 0.0]]))
        assert kalman_rank(Ac, Bc) == 8

    def test_example_3_1_steering(self):
        """Ex 3.1：X(0)=[1..8] 能被驱动到原点（论文图 6）。"""
        L1 = np.array([[2, -1, -1, 0], [-1, 2, -1, 0], [-1, -1, 3, -1], [0, 0, -1, 1]], dtype=float)
        L2 = np.array([[2, -1, -1, 0], [-1, 2, -1, 0], [-1, -1, 2, 0], [0, 0, 0, 0]], dtype=float)
        Ac = np.kron(np.eye(4), np.array([[1.0, 0.0], [0.0, 0.0]])) - np.kron(L1 + L2, np.eye(2))
        Bc = np.kron(np.diag([1.0, 1.0, 0.0, 1.0]), np.array([[1.0, 0.0], [1.0, 0.0]]))
        result = minimum_energy_control(
            Ac, Bc, np.arange(1, 9, dtype=float), np.zeros(8), 2.0, num_points=1200
        )
        assert result.final_error < 1e-3

    def test_example_3_2_rank_is_6(self):
        """Ex 3.2：同一单纯复形换输入后不可控，秩 = 6 < 8（论文值）。"""
        L1 = np.array([[2, -1, -1, 0], [-1, 2, -1, 0], [-1, -1, 3, -1], [0, 0, -1, 1]], dtype=float)
        L2 = np.array([[2, -1, -1, 0], [-1, 2, -1, 0], [-1, -1, 2, 0], [0, 0, 0, 0]], dtype=float)
        Ac = np.kron(np.eye(4), np.array([[1.0, 0.0], [0.0, 0.0]])) - np.kron(L1 + L2, np.eye(2))
        Bc = np.kron(np.diag([0.0, 0.0, 1.0, 1.0]), np.array([[1.0, 0.0], [0.0, 2.0]]))
        assert kalman_rank(Ac, Bc) == 6


class TestHigherOrderHelpsControllability:
    """论文核心论点：高阶交互可能让不可控的成对网络变可控。"""

    def test_pairwise_uncontrollable_but_hypergraph_controllable(self):
        """Ex 2.1 的成对网络秩只有 6 < 10，而超图版本是 10。

        这是论文 Remark 3 的定量体现。
        """
        # 论文给出的成对网络 Laplacian L̃（第 5 页）
        L_tilde = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0],
            ],
            dtype=float,
        )
        A = np.eye(2)
        Lam = np.eye(2)
        B = np.array([[0.0, 1.0], [1.0, 0.0]])
        delta = np.diag([0.0, 1.0, 1.0, 0.0, 1.0])
        Ac = np.kron(np.eye(5), A) + np.kron(L_tilde, Lam)
        Bc = np.kron(delta, B)
        assert kalman_rank(Ac, Bc) == 6

        # 同一结点集换成超图 Laplacian 后变为可控
        mats = hypergraph_laplacian(EDGES_21, num_nodes=5)
        sys = HypergraphControlSystem(
            A=A,
            L=mats[2] + mats[3],
            Lambda=Lam,
            B=B,
            controlled_nodes=(1, 2, 4),
            coupling=1.0,
            order=3,
        )
        Ac2, Bc2 = sys.to_blocks()
        assert kalman_rank(Ac2, Bc2) == 10


class TestControlSystemAssembly:
    """系统组装（式 (4)）的正确性。"""

    def test_block_shapes(self):
        """𝒜 是 (Nn, Nn)，ℬ 是 (Nn, Np)。"""
        mats = hypergraph_laplacian(EDGES_21, num_nodes=5)
        sys = HypergraphControlSystem(
            A=np.eye(2),
            L=mats[2] + mats[3],
            Lambda=np.eye(2),
            B=np.array([[0.0, 1.0], [1.0, 0.0]]),
            controlled_nodes=(1, 2, 4),
            coupling=1.0,
            order=3,
        )
        Ac, Bc = sys.to_blocks()
        assert Ac.shape == (10, 10)
        assert Bc.shape == (10, 10)

    def test_delta_vector(self):
        """δ 向量在受控位置为 1，其余为 0。"""
        mats = hypergraph_laplacian(EDGES_21, num_nodes=5)
        sys = HypergraphControlSystem(
            A=np.eye(2),
            L=mats[2] + mats[3],
            Lambda=np.eye(2),
            B=np.eye(2),
            controlled_nodes=(1, 2, 4),
        )
        assert np.array_equal(sys.delta, np.array([0.0, 1.0, 1.0, 0.0, 1.0]))
        assert sys.controlled == (1, 2, 4)

    def test_all_nodes_controlled_by_default(self):
        """不指定 controlled_nodes 时全部受控。"""
        sys = HypergraphControlSystem(
            A=np.eye(2),
            L=np.eye(3),
            Lambda=np.eye(2),
            B=np.eye(2),
        )
        assert np.array_equal(sys.delta, np.ones(3))

    def test_uncontrolled_has_no_input(self):
        """未受控结点对应的输入块必须全零。"""
        mats = hypergraph_laplacian(EDGES_21, num_nodes=5)
        sys = HypergraphControlSystem(
            A=np.eye(2),
            L=mats[2] + mats[3],
            Lambda=np.eye(2),
            B=np.eye(2),
            controlled_nodes=(0,),
            order=3,
        )
        _, Bc = sys.to_blocks()
        # 只有结点 0 受控：前 2 行非零，其余 8 行全零
        assert np.abs(Bc[2:, :]).max() == 0.0
        assert np.abs(Bc[:2, :]).max() == 1.0

    def test_flatten_roundtrip(self):
        """flatten / unflatten 必须互逆。"""
        sys = HypergraphControlSystem(A=np.eye(2), L=np.eye(3), Lambda=np.eye(2), B=np.eye(2))
        X = np.arange(6, dtype=float).reshape(3, 2)
        assert np.array_equal(sys.unflatten(sys.flatten(X)), X)
