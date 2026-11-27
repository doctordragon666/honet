"""前向仿真：给定控制输入，计算系统轨迹（参考 nctpy ``sim_state_eq``）。

与 :mod:`honet.control._control_law` / :mod:`honet.control._optimal_control`
的区别是方向相反：那两个是「给定目标反解输入」，本模块是「给定输入正算
轨迹」。三者合起来才构成完整的开环控制闭环：

```
反解输入  →  minimum_energy_control / optimal_control
正算轨迹  →  simulate_system
```

实际用途：验证反解出的输入是否真能驱动到目标（端到端自检），或模拟外部
刺激（如脑刺激）引起的响应。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from honet._deps import require


@dataclass(frozen=True)
class SimulationResult:
    """前向仿真结果。

    属性:
        times: 时间网格，形状 ``(m,)``。
        states: 状态轨迹，形状 ``(m, N)``。
        inputs: 实际施加的输入，形状 ``(m, p)``。
    """

    times: Any
    states: Any
    inputs: Any

    @property
    def final_state(self) -> Any:
        """终点状态 ``x(T)``。"""
        return self.states[-1]


def simulate_system(
    A: Any,
    B: Any,
    x0: Any,
    U: Any,
    *,
    dt: float = 1e-3,
    system: str = "continuous",
) -> SimulationResult:
    """给定控制输入，仿真系统轨迹。

    参数:
        A: 系统矩阵，形状 ``(N, N)``。
        B: 输入矩阵，形状 ``(N, p)``。
        x0: 初始状态，形状 ``(N,)``。
        U: 控制输入，形状 ``(m, p)``；``m`` 为时间点数。
        dt: 连续时间下的采样间隔（决定 ``times``）。
        system: ``"continuous"`` 或 ``"discrete"``。

    返回:
        :class:`SimulationResult`。

    抛出:
        ValueError: 维度不匹配或 ``system`` 取值非法。

    示例:
        >>> import numpy as np
        >>> A = np.diag([-1.0, -2.0])
        >>> B = np.eye(2)
        >>> result = simulate_system(A, B, [1.0, 1.0], np.zeros((10, 2)))
        >>> result.final_state[0] < 1.0
        True

    说明:
        连续时间下用**零输入响应的解析解 + 零阶保持的输入积分**逐段推进::

            x(t+dt) = e^{A·dt} x(t) + (∫₀^{dt} e^{Aτ}dτ) B u(t)

        离散时间下直接迭代 ``x_{k+1} = A x_k + B u_k``。
    """
    np = require("numpy")
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    total = A.shape[0]
    if A.shape[0] != A.shape[1]:
        msg = f"A 必须是方阵，实际是 {A.shape}"
        raise ValueError(msg)
    if B.shape[0] != total:
        msg = f"B 的行数必须等于 A 的阶数：A {A.shape}, B {B.shape}"
        raise ValueError(msg)
    if system not in ("continuous", "discrete"):
        msg = f"system 必须是 'continuous' 或 'discrete'，实际是 {system!r}"
        raise ValueError(msg)

    start = np.asarray(x0, dtype=float).reshape(-1)
    if start.shape[0] != total:
        msg = f"x0 的长度应为 {total}，实际是 {start.shape[0]}"
        raise ValueError(msg)

    inputs = np.atleast_2d(np.asarray(U, dtype=float))
    if inputs.ndim == 1:
        inputs = inputs.reshape(-1, 1)
    if inputs.shape[1] != B.shape[1]:
        msg = f"U 的列数应等于 B 的列数 {B.shape[1]}，实际是 {inputs.shape[1]}"
        raise ValueError(msg)

    steps = inputs.shape[0]
    states = np.empty((steps, total))
    states[0] = start
    times = np.arange(steps, dtype=float) * dt

    if system == "discrete":
        for i in range(1, steps):
            states[i] = A @ states[i - 1] + B @ inputs[i - 1]
        return SimulationResult(times=times, states=states, inputs=inputs)

    expm = require("scipy.linalg", attr="expm")
    # 增广矩阵一次给出 e^{A·dt} 与 ∫₀^{dt} e^{Aτ}dτ
    aug = np.zeros((2 * total, 2 * total))
    aug[:total, :total] = A
    aug[:total, total:] = np.eye(total)
    block = expm(aug * dt)
    phi = block[:total, :total]
    integral = block[:total, total:]
    for i in range(1, steps):
        states[i] = phi @ states[i - 1] + integral @ B @ inputs[i - 1]

    return SimulationResult(times=times, states=states, inputs=inputs)


def simulate_impulse(
    A: Any,
    B: Any,
    x0: Any,
    impulse: Any,
    *,
    steps: int,
    dt: float = 1e-3,
    system: str = "continuous",
) -> SimulationResult:
    """施加恒定输入（阶跃刺激）并仿真。

    这是 :func:`simulate_system` 的便捷封装：把同一个输入向量重复 ``steps``
    次，适合模拟「持续刺激某几个结点」的场景。

    参数:
        A: 系统矩阵。
        B: 输入矩阵。
        x0: 初始状态。
        impulse: 恒定的输入向量，形状 ``(p,)``。
        steps: 时间点数。
        dt: 采样间隔。
        system: ``"continuous"`` 或 ``"discrete"``。

    返回:
        :class:`SimulationResult`。

    示例:
        >>> import numpy as np
        >>> A = np.diag([-1.0, -2.0])
        >>> B = np.eye(2)
        >>> result = simulate_impulse(A, B, [0.0, 0.0], [1.0, 0.0], steps=50)
        >>> result.final_state[0] > 0
        True
    """
    np = require("numpy")
    vector = np.asarray(impulse, dtype=float).reshape(-1)
    repeated = np.tile(vector, (int(steps), 1))
    return simulate_system(A, B, x0, repeated, dt=dt, system=system)
