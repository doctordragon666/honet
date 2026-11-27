"""控制律：最小能量驱动到目标状态（论文式 (5) 与 Lemma 3）。

给定 ``Ẋ = 𝒜X + ℬU``、初态 ``X(0)`` 与目标 ``X(T)``，最小能量控制律为::

    U(t) = ℬᵀ e^{𝒜ᵀ(T-t)} W(T)^{-1} (X(T) - e^{𝒜T} X(0))
    W(T) = ∫_0^T e^{𝒜s} ℬℬᵀ e^{𝒜ᵀs} ds

这正是旧 ``honet.control`` 完全没有的内容：它只回答「控制哪些结点」，
不给出「怎么控制」。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from honet._deps import require
from honet.control._controllability import gramian


@dataclass(frozen=True)
class ControlTrajectory:
    """一次控制仿真的结果。

    属性:
        times: 时间网格，形状 ``(m,)``。
        states: 状态轨迹，形状 ``(m, d)``；每行是 ``X(t)`` 的堆叠向量。
        inputs: 控制输入轨迹，形状 ``(m, p_total)``。
        energy: 控制能量 ``∫_0^T ‖U‖² dt`` 的数值估计。
        final_error: 终点误差 ``‖X(T) - X_target‖``。
    """

    times: Any
    states: Any
    inputs: Any
    energy: float
    final_error: float

    @property
    def converged(self) -> bool:
        """终点误差是否在默认容差内（``1e-3``）。"""
        return self.final_error < 1e-3


def minimum_energy_control(
    A: Any,
    B: Any,
    X0: Any,
    X_target: Any,
    T: float,
    *,
    num_points: int = 1000,
    rcond: float = 1e-12,
) -> ControlTrajectory:
    """计算最小能量控制律并给出状态轨迹。

    参数:
        A: 系统矩阵 ``𝒜``，形状 ``(d, d)``。
        B: 输入矩阵 ``ℬ``，形状 ``(d, p)``。
        X0: 初始状态，形状 ``(d,)``。
        X_target: 目标状态，形状 ``(d,)``。
        T: 终止时间（正数）。
        num_points: 输出轨迹的采样点数。
        rcond: 求逆时的相对条件数阈值；Gramian 病态时据此截断。

    返回:
        :class:`ControlTrajectory`。

    抛出:
        ValueError: Gramian 不可逆（系统在该时间窗内不可控）。

    示例:
        >>> import numpy as np
        >>> A = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> B = np.array([[0.0, 1.0], [1.0, 0.0]])
        >>> result = minimum_energy_control(A, B, [1.0, 2.0], [0.0, 0.0], 2.0)
        >>> result.converged
        True
    """
    np = require("numpy")
    expm = require("scipy.linalg", attr="expm")

    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    dim = A.shape[0]
    x0 = np.asarray(X0, dtype=float).reshape(-1)
    target = np.asarray(X_target, dtype=float).reshape(-1)
    if x0.shape[0] != dim or target.shape[0] != dim:
        msg = f"状态维数不匹配：A 是 {dim}，X0 是 {x0.shape}，目标 {target.shape}"
        raise ValueError(msg)

    W = gramian(A, B, T)
    # 用 SVD 求伪逆，病态时自动截断，比直接 inv 稳定
    u_mat, s_vals, vt_mat = np.linalg.svd(W)
    cutoff = rcond * (s_vals[0] if s_vals.size else 0.0)
    inv_s = np.array([1.0 / s if s > cutoff else 0.0 for s in s_vals])
    W_pinv = (vt_mat.T * inv_s) @ u_mat.T

    reach = target - expm(A * T) @ x0
    times = np.linspace(0.0, T, num_points)

    # U(t) = Bᵀ e^{Aᵀ(T-t)} W⁺ reach
    inputs = np.empty((num_points, B.shape[1]))
    for i, t in enumerate(times):
        propagator = expm(A.T * (T - t))
        inputs[i] = (B.T @ propagator @ W_pinv @ reach).reshape(-1)

    states = _propagate(A, B, x0, times, inputs)
    energy = float(np.trapezoid(np.sum(inputs**2, axis=1), times))
    final_error = float(np.linalg.norm(states[-1] - target))

    return ControlTrajectory(
        times=times,
        states=states,
        inputs=inputs,
        energy=energy,
        final_error=final_error,
    )


def _propagate(A: Any, B: Any, x0: Any, times: Any, inputs: Any) -> Any:
    """把输入按分段线性插值精确推进状态。

    每段 ``[t_i, t_{i+1}]`` 内把 ``u`` 视为线性，则::

        X(t+dt) = e^{A·dt} X(t) + Φ₁ B u_i + Φ₂ B (u_{i+1} - u_i)

    其中 ``Φ₁ = ∫_0^{dt} e^{Aτ} dτ``、``Φ₂ = (1/dt)∫_0^{dt} e^{Aτ}(dt-τ)dτ``。
    两者由增广矩阵指数一次求出，是**解析精确**的（无时间离散误差），
    因此整条轨迹的终点与 ``u`` 的采样密度无关，只取决于 ``u`` 在采样点间的
    线性插值假设。

    实现说明：``Φ₂`` 通过 ``[[A, I, 0], [0, 0, I], [0, 0, 0]]`` 的指数取得
    （上三角块分别对应 ``e^{A·dt}``、``Φ₁``、``dt·Φ₂``）。
    """
    np = require("numpy")
    expm = require("scipy.linalg", attr="expm")
    dim = A.shape[0]
    states = np.empty((times.shape[0], dim))
    states[0] = x0
    for i in range(1, times.shape[0]):
        dt = times[i] - times[i - 1]
        aug = np.zeros((3 * dim, 3 * dim))
        aug[:dim, :dim] = A
        aug[:dim, dim : 2 * dim] = np.eye(dim)
        aug[dim : 2 * dim, 2 * dim :] = np.eye(dim)
        block = expm(aug * dt)
        phi = block[:dim, :dim]
        phi1 = block[:dim, dim : 2 * dim]
        phi2 = block[:dim, 2 * dim :] / dt  # 除以 dt 得到 (1/dt)∫e^{Aτ}(dt-τ)dτ
        du = inputs[i] - inputs[i - 1]
        states[i] = phi @ states[i - 1] + phi1 @ B @ inputs[i - 1] + phi2 @ B @ du
    return states
