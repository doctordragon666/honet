"""控制能量的积分与快速算法（参考 nctpy ``integrate_u`` / ``minimum_energy_fast``）。

两个函数解决不同问题：

* :func:`integrate_control_energy` —— 给定控制信号 ``u(t)``，算每个结点的
  能量贡献 ``∫ u_i(t)² dt``。用 Simpson 法则而非梯形法，因为稀疏控制集下
  ``u`` 可能剧烈变化，梯形法误差偏大。
* :func:`minimum_energy_fast` —— 用 Simpson 型 Gramian 数值积分快速估算最小
  能量，避免构造 ``expm`` 的大矩阵（nctpy 用 1000 步定步长）。

**已知差异（有意为之）**：nctpy 的 ``minimum_energy_fast`` 用定步长 Simpson
近似 Gramian，精度受步数限制；本实现的
:func:`honet.control.minimum_energy_control` 用精确的块指数解法。两者在
``nt`` 足够大时一致，测试里验证了这一点。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require


def integrate_control_energy(U: Any, dt: float = 1.0) -> Any:
    """对控制信号平方做 Simpson 积分，返回每个通道的能量。

    参数:
        U: 控制信号，形状 ``(m, p)``；``m`` 为时间点数，``p`` 为通道数。
        dt: 采样间隔。

    返回:
        长度 ``p`` 的数组，第 ``k`` 项是通道 ``k`` 的能量 ``∫ u_k(t)² dt``。

    示例:
        >>> import numpy as np
        >>> U = np.ones((101, 2))
        >>> energy = integrate_control_energy(U, dt=0.01)
        >>> np.allclose(energy, 1.0)
        True

    说明:
        用 ``scipy.integrate.simpson``（Simpson 1/3 法则）。若点数不足 3，
        退化为梯形法。nctpy 在同名函数里用字符串比较 scipy 版本号来选择
        ``simps`` / ``simpson``，在 scipy >= 1.14 上会因 ``simps`` 已移除而
        崩溃；这里直接调用现存的 ``simpson``。
    """
    np = require("numpy")
    simpson = require("scipy.integrate", attr="simpson")

    signals = np.atleast_2d(np.asarray(U, dtype=float))
    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)
    squared = np.square(signals)
    if squared.shape[0] < 3:
        # 点数太少时 Simpson 不适用，退化为梯形法
        return np.trapezoid(squared, dx=dt, axis=0)
    return simpson(squared, dx=dt, axis=0)


def minimum_energy_fast(
    A: Any,
    B: Any,
    x0: Any,
    xf: Any,
    T: float,
    *,
    steps: int = 1000,
) -> float:
    """用 Simpson 型 Gramian 数值积分快速估算最小控制能量。

    参数:
        A: 系统矩阵，形状 ``(N, N)``。
        B: 输入矩阵，形状 ``(N, p)``。
        x0: 初始状态，形状 ``(N,)``。
        xf: 目标状态，形状 ``(N,)``。
        T: 时间上界。
        steps: 积分步数（偶数）。步数越大越精确。

    返回:
        最小控制能量 ``reachᵀ W⁻¹ reach``。

    示例:
        >>> import numpy as np
        >>> A = np.diag([-1.0, -2.0])
        >>> B = np.eye(2)
        >>> energy = minimum_energy_fast(A, B, [1.0, 1.0], [0.0, 0.0], 2.0)
        >>> energy > 0
        True

    说明:
        Gramian 用 Simpson 1/3 法则累加::

            W ≈ dt/3 · (B Bᵀ + 4Σ_odd + 2Σ_even + e^{AT}BBᵀe^{AᵀT})

        这是 nctpy ``minimum_energy_fast`` 的做法。精确解请用
        :func:`honet.control.minimum_energy_control`（块指数法）。
    """
    np = require("numpy")
    expm = require("scipy.linalg", attr="expm")

    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    total = A.shape[0]
    start = np.asarray(x0, dtype=float).reshape(-1)
    target = np.asarray(xf, dtype=float).reshape(-1)

    nt = int(steps)
    if nt % 2 != 0:
        nt += 1
    dt = T / nt

    step = expm(A * dt)
    accumulated = np.eye(total)
    BB = B @ B.T
    gram = np.zeros((total, total))

    for _ in range(nt // 2):
        # 奇数项权重 4
        accumulated = accumulated @ step
        odd = accumulated @ B
        # 偶数项权重 2
        accumulated = accumulated @ step
        even = accumulated @ B
        gram = gram + 4.0 * (odd @ odd.T) + 2.0 * (even @ even.T)

    # 最后一个奇数项
    accumulated = accumulated @ step
    last = accumulated @ B
    gram = gram + 4.0 * (last @ last.T)

    terminal = expm(A * T)
    endpoint = terminal @ B
    gram = (gram + BB + endpoint @ endpoint.T) * dt / 3.0

    reach = target - terminal @ start
    return float(reach @ np.linalg.solve(gram, reach))


def minimum_energy_exact(
    A: Any,
    B: Any,
    x0: Any,
    xf: Any,
    T: float,
) -> float:
    """用精确 Gramian（块指数法）计算最小控制能量。

    参数:
        A: 系统矩阵。
        B: 输入矩阵。
        x0: 初始状态。
        xf: 目标状态。
        T: 时间上界。

    返回:
        最小控制能量。

    示例:
        >>> import numpy as np
        >>> A = np.diag([-1.0, -2.0])
        >>> energy = minimum_energy_exact(A, np.eye(2), [1.0, 1.0], [0.0, 0.0], 2.0)
        >>> energy > 0
        True
    """
    np = require("numpy")
    expm = require("scipy.linalg", attr="expm")

    from honet.control._controllability import gramian

    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    start = np.asarray(x0, dtype=float).reshape(-1)
    target = np.asarray(xf, dtype=float).reshape(-1)

    W = gramian(A, B, T)
    reach = target - expm(A * T) @ start
    return float(reach @ np.linalg.solve(W, reach))
