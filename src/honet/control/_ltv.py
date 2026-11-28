"""线性时变（LTV）系统的可控性判定（私有）。

常数系统的 Kalman 秩判据对 LTV 系统**不适用**：``[B, AB, A²B, …]`` 只在
``A`` 为常数时才有意义。LTV 系统的可控性由**可控性 Gramian** 的秩决定：

.. math::

    W_c(t_0, t_f) = \\int_{t_0}^{t_f} \\Phi(t_f, \\tau) B(\\tau) B(\\tau)^T
                    \\Phi(t_f, \\tau)^T \\, d\\tau

其中 ``Φ(t_f, τ)`` 是状态转移矩阵。``W_c`` 满秩即系统在 ``[t_0, t_f]`` 上完全
可控。``W_c`` 用数值积分（中点法）估计，状态转移矩阵按时间网格递推：

.. math::

    \\Phi(t_{k+1}, t_0) = e^{A(t_k) \\Delta t} \\Phi(t_k, t_0)

这是冻结系数近似，与 :func:`honet.control.simulate_system` 的 LTV 处理口径一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from honet._deps import require
from honet.control._timevarying import resolve_matrix


@dataclass(frozen=True)
class LTVControllabilityResult:
    """LTV 可控性分析结果。

    属性:
        gramian: 可控性 Gramian，形状 ``(N, N)``。
        rank: Gramian 的数值秩。
        singular_values: Gramian 的奇异值（降序）。
        controllable: 秩是否等于系统阶数。
        condition_number: 最大/最小奇异值之比；秩亏时为 ``inf``。
        times: 用于积分的时刻网格。
    """

    gramian: Any
    rank: int
    singular_values: Any
    controllable: bool
    condition_number: float
    times: Any


def _transition_matrices(
    A: Any,
    times: list[float],
    *,
    total: int,
    a_varying: bool,
) -> list[Any]:
    """递推各时刻的累积状态转移矩阵 ``Φ(t_k, t_0)``。"""
    np = require("numpy")
    expm = require("scipy.linalg", attr="expm")

    phi_list = [np.eye(total)]
    for index in range(1, len(times)):
        dt = times[index] - times[index - 1]
        a_now = (
            resolve_matrix(A, times[index - 1], rows=total, cols=total, name="A")
            if a_varying
            else A
        )
        phi_list.append(expm(a_now * dt) @ phi_list[-1])
    return phi_list


def ltv_gramian(
    A: Any,
    B: Any,
    *,
    t0: float = 0.0,
    t_final: float = 1.0,
    steps: int = 256,
) -> Any:
    """数值估计 LTV 系统的可控性 Gramian ``W_c(t0, t_final)``。

    参数:
        A: 系统矩阵，或 ``Callable[[float], ndarray]``。
        B: 输入矩阵，或 ``Callable[[float], ndarray]``。
        t0: 起始时刻。
        t_final: 终止时刻。
        steps: 积分采样点数（中点法），越大越精确。

    返回:
        形状 ``(N, N)`` 的对称半正定矩阵。

    抛出:
        ValueError: 维度不符或 ``steps < 2``。

    示例:
        >>> import numpy as np
        >>> W = ltv_gramian(np.diag([-1.0]), np.eye(1), t_final=1.0)
        >>> W.shape
        (1, 1)
        >>> float(W[0, 0]) > 0
        True
    """
    np = require("numpy")
    from honet.control._timevarying import is_time_varying, sample_grid

    if steps < 2:
        msg = f"steps 必须 >= 2，收到 {steps}"
        raise ValueError(msg)

    a_varying = is_time_varying(A)
    b_varying = is_time_varying(B)

    raw_a = np.atleast_2d(np.asarray(A(0.0) if a_varying else A, dtype=float))
    total = raw_a.shape[0]
    if raw_a.shape[0] != raw_a.shape[1]:
        msg = f"A 必须是方阵，实际是 {raw_a.shape}"
        raise ValueError(msg)
    raw_b = np.atleast_2d(np.asarray(B(0.0) if b_varying else B, dtype=float))
    if raw_b.shape[0] != total:
        msg = f"B 的行数必须等于 A 的阶数：A 阶 {total}, B {raw_b.shape}"
        raise ValueError(msg)
    columns = raw_b.shape[1]

    grid = sample_grid(t0, t_final, steps)
    phi_list = _transition_matrices(A, grid, total=total, a_varying=a_varying)

    gramian = np.zeros((total, total))
    # 中点法：用区间中点的 Φ 与 B 估计该区间的贡献
    for index in range(len(grid) - 1):
        dt = grid[index + 1] - grid[index]
        mid = 0.5 * (grid[index] + grid[index + 1])
        b_now = resolve_matrix(B, mid, rows=total, cols=columns, name="B") if b_varying else B
        # Φ(mid, t0) ≈ Φ(t_k, t0)，误差 O(dt)
        phi_mid = phi_list[index]
        # 从 mid 到 t_final 的转移矩阵：Φ(t_f, mid) = Φ(t_f, t0) Φ(mid, t0)^{-1}
        # 对稳定系统 Φ(t_f,t0) 可能接近奇异，这里改用「从 mid 正向推进到 t_f」
        # 的近似：用剩余的累积转移矩阵右乘逆，等价于按中点切分
        phi_final = phi_list[-1]
        try:
            phi_from_mid = phi_final @ np.linalg.inv(phi_mid)
        except np.linalg.LinAlgError:
            phi_from_mid = phi_final @ np.linalg.pinv(phi_mid)
        reach = phi_from_mid @ b_now
        gramian += dt * (reach @ reach.T)

    # 对称化，消除浮点累积的不对称
    return 0.5 * (gramian + gramian.T)


def ltv_controllability(
    A: Any,
    B: Any,
    *,
    t0: float = 0.0,
    t_final: float = 1.0,
    steps: int = 256,
    rcond: float = 1e-10,
) -> LTVControllabilityResult:
    """判定 LTV 系统在 ``[t0, t_final]`` 上是否完全可控。

    参数:
        A: 系统矩阵，或 ``Callable[[float], ndarray]``。
        B: 输入矩阵，或 ``Callable[[float], ndarray]``。
        t0: 起始时刻。
        t_final: 终止时刻。
        steps: 积分采样点数。
        rcond: 判定秩时的相对阈值（与 ``numpy.linalg.matrix_rank`` 同义）。

    返回:
        :class:`LTVControllabilityResult`。

    示例:
        >>> import numpy as np
        >>> result = ltv_controllability(np.diag([-1.0, -2.0]), np.eye(2), t_final=1.0)
        >>> result.controllable
        True
        >>> result.rank
        2

        只驱动一个通道时秩亏：

        >>> B1 = np.array([[1.0], [0.0]])
        >>> ltv_controllability(np.diag([-1.0, -2.0]), B1, t_final=1.0).controllable
        False

    说明:
        常数 ``A``、``B`` 时本函数同样可用，结果应与 Kalman 秩判据一致；
        两者的一致性由测试保证。
    """
    np = require("numpy")
    from honet.control._timevarying import sample_grid

    gramian = ltv_gramian(A, B, t0=t0, t_final=t_final, steps=steps)
    singular_values = np.linalg.svd(gramian, compute_uv=False)
    rank = int(np.linalg.matrix_rank(gramian, tol=rcond * singular_values[0]))
    total = gramian.shape[0]

    if singular_values[-1] > 0:
        condition = float(singular_values[0] / singular_values[-1])
    else:
        condition = float("inf")

    return LTVControllabilityResult(
        gramian=gramian,
        rank=rank,
        singular_values=singular_values,
        controllable=rank == total,
        condition_number=condition,
        times=np.asarray(sample_grid(t0, t_final, steps)),
    )
