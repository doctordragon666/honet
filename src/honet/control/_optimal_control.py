"""带状态轨迹约束的最优控制（Hamiltonian 状态-协态法）。

与 :mod:`honet.control._control_law` 的最小能量控制不同，这里允许对**状态
轨迹**加惩罚：目标函数是::

    J = ∫₀ᵀ [ ρ‖u(t)‖² + ‖S(x(t) - x_r)‖² ] dt

即「控制能量 + 偏离参考状态的代价」的加权和。``ρ`` 越小，控制能量权重越低，
系统越倾向于用大输入换取状态贴合；``ρ`` 越大则反过来。

求解方法是构造状态-协态联合系统。对连续时间，一阶最优条件给出::

    d/dt [x; λ] = M [x; λ] + c
    M = [[A,        -BBᵀ/(2ρ)],
         [-2S,      -Aᵀ       ]]
    c = [0; 2S x_r]

其中 ``λ`` 是协态向量。由于 ``x(0)`` 与 ``x(T)`` 都已知（两端边值问题），
先解出 ``λ(0)``，再正向积分整条轨迹。

参考: nctpy ``energies.get_control_inputs``；理论出处见
Gu et al., *Nature Communications* 6:8414, 2015。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from honet._deps import require


@dataclass(frozen=True)
class OptimalControlResult:
    """带状态约束最优控制的结果。

    属性:
        times: 时间网格，形状 ``(m,)``。
        states: 状态轨迹，形状 ``(m, N)``。
        inputs: 控制输入轨迹，形状 ``(m, N)``。
        energy: 控制能量 ``∫‖u‖²dt``。
        final_error: 终点误差 ``‖x(T) - x_f‖``。
        boundary_error: 边值问题的残差（应接近 0，用于诊断数值问题）。
    """

    times: Any
    states: Any
    inputs: Any
    energy: float
    final_error: float
    boundary_error: float

    @property
    def converged(self) -> bool:
        """终点误差是否在默认容差内（``1e-3``）。"""
        return self.final_error < 1e-3


def optimal_control(
    A: Any,
    B: Any,
    x0: Any,
    xf: Any,
    T: float,
    *,
    rho: float = 1.0,
    S: Any = None,
    xr: Any = "zero",
    system: str = "continuous",
    dt: float = 1e-3,
) -> OptimalControlResult:
    """求解带状态轨迹约束的最优控制问题。

    参数:
        A: 系统矩阵，形状 ``(N, N)``。
        B: 控制输入矩阵，形状 ``(N, N)``；对角元标记哪些结点受控及权重。
        x0: 初始状态，形状 ``(N,)``。
        xf: 目标状态，形状 ``(N,)``。
        T: 时间上界（正数）。
        rho: 混合参数，``ρ → 0`` 时接近纯最小能量控制。必须为正。
        S: 状态约束矩阵，形状 ``(N, N)``。``None`` 表示约束全部结点。
        xr: 参考状态。可取 ``"zero"`` / ``"x0"`` / ``"xf"`` / ``"midpoint"``
            或显式数组。
        system: ``"continuous"`` 或 ``"discrete"``。
        dt: 连续时间下的积分步长。

    返回:
        :class:`OptimalControlResult`。

    抛出:
        ValueError: 参数非法（``rho <= 0``、维度不匹配、``system`` 取值非法）。

    示例:
        >>> import numpy as np
        >>> A = np.diag([-1.0, -2.0])
        >>> B = np.eye(2)
        >>> result = optimal_control(A, B, [1.0, 1.0], [0.0, 0.0], 2.0, rho=1.0)
        >>> result.final_error < 1e-2
        True

    说明:
        当 ``S = 0`` 时退化为最小能量控制，此时结果应与
        :func:`honet.control.minimum_energy_control` 一致（能量意义下最优）。
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
    if rho <= 0:
        msg = f"rho 必须为正，实际是 {rho}"
        raise ValueError(msg)
    if system not in ("continuous", "discrete"):
        msg = f"system 必须是 'continuous' 或 'discrete'，实际是 {system!r}"
        raise ValueError(msg)

    start = _as_vector(x0, total, "x0")
    target = _as_vector(xf, total, "xf")
    constraint = np.eye(total) if S is None else np.atleast_2d(np.asarray(S, dtype=float))
    reference = _resolve_reference(xr, start, target, total)

    if system == "continuous":
        return _solve_continuous(A, B, start, target, T, rho, constraint, reference, dt)
    return _solve_discrete(A, B, start, target, T, rho, constraint, reference)


def _as_vector(value: Any, total: int, name: str) -> Any:
    """把输入规整成 ``(N,)`` 的浮点向量。"""
    np = require("numpy")
    vec = np.asarray(value, dtype=float).reshape(-1)
    if vec.shape[0] != total:
        msg = f"{name} 的长度应为 {total}，实际是 {vec.shape[0]}"
        raise ValueError(msg)
    return vec


def _resolve_reference(xr: Any, x0: Any, xf: Any, total: int) -> Any:
    """把 ``xr`` 的各种写法解析成具体向量。"""
    np = require("numpy")
    if isinstance(xr, str):
        mapping = {
            "zero": lambda: np.zeros(total),
            "x0": lambda: x0.copy(),
            "xf": lambda: xf.copy(),
            "midpoint": lambda: x0 + (xf - x0) * 0.5,
        }
        if xr not in mapping:
            msg = f"未知的 xr 取值：{xr!r}（可用 'zero' / 'x0' / 'xf' / 'midpoint'）"
            raise ValueError(msg)
        return mapping[xr]()
    return _as_vector(xr, total, "xr")


def _solve_continuous(
    A: Any,
    B: Any,
    x0: Any,
    xf: Any,
    T: float,
    rho: float,
    S: Any,
    xr: Any,
    dt: float,
) -> OptimalControlResult:
    """连续时间：解两端边值问题，再正向积分状态-协态系统。"""
    np = require("numpy")
    expm = require("scipy.linalg", attr="expm")
    total = A.shape[0]

    # 状态-协态联合矩阵
    M = np.zeros((2 * total, 2 * total))
    M[:total, :total] = A
    M[:total, total:] = -(B @ B.T) / (2.0 * rho)
    M[total:, :total] = -2.0 * S
    M[total:, total:] = -A.T

    # 参考状态带来的常数项
    offset = np.zeros(2 * total)
    offset[total:] = 2.0 * S @ xr
    constant = np.linalg.solve(M, offset)

    E = expm(M * T)
    idx = np.arange(total)
    E11 = E[np.ix_(idx, idx)]
    E12 = E[np.ix_(idx, idx + total)]

    # 由两端边值解出初始协态
    right = xf - E11 @ x0 - (E11 - np.eye(total)) @ constant[:total] - E12 @ constant[total:]
    try:
        lam0 = np.linalg.solve(E12, right)
    except np.linalg.LinAlgError as exc:
        msg = (
            "无法解出初始协态：E12 奇异，说明该系统在 [0, T] 内不可控或 "
            "T 过小；请增大 T 或检查 B 是否覆盖可控子空间"
        )
        raise ValueError(msg) from exc

    steps = int(round(T / dt)) + 1
    times = np.linspace(0.0, T, steps)

    # 用增广矩阵做精确的常系数线性推进（含常数项）
    aug = np.zeros((2 * total + 1, 2 * total + 1))
    aug[: 2 * total, : 2 * total] = M * dt
    aug[: 2 * total, 2 * total] = constant * dt
    step = expm(aug)

    z = np.zeros((2 * total, steps))
    z[:, 0] = np.concatenate([x0, lam0])
    for i in range(1, steps):
        z[:, i] = step[: 2 * total, : 2 * total] @ z[:, i - 1] + step[: 2 * total, 2 * total]

    states = z[:total, :].T
    inputs = (-(B.T @ z[total:, :]) / (2.0 * rho)).T
    energy = float(np.trapezoid(np.sum(inputs**2, axis=1), times))
    final_error = float(np.linalg.norm(states[-1] - xf))
    boundary_error = float(np.linalg.norm(E12 @ lam0 - right))

    return OptimalControlResult(
        times=times,
        states=states,
        inputs=inputs,
        energy=energy,
        final_error=final_error,
        boundary_error=boundary_error,
    )


def _solve_discrete(
    A: Any,
    B: Any,
    x0: Any,
    xf: Any,
    T: float,
    rho: float,
    S: Any,
    xr: Any,
) -> OptimalControlResult:
    """离散时间：把所有时刻的约束拼成一个大型稀疏线性系统一次求解。

    未知量按 ``[x_0..x_{T-1}, λ_1..λ_T]`` 排列，边值条件给出 ``x_0 = x0``
    与 ``x_T = xf``，状态方程与协态方程给出其余约束。
    """
    np = require("numpy")
    sparse = require("scipy", attr="sparse")
    spsolve = require("scipy.sparse.linalg", attr="spsolve")

    total = A.shape[0]
    horizon = int(round(T))
    if horizon < 2:
        msg = f"离散时间系统的 T 必须 >= 2，实际是 {T}"
        raise ValueError(msg)

    C = (B @ B.T) / (2.0 * rho)
    D = 2.0 * S
    identity = np.eye(total)

    # 未知量分块：[x_0 .. x_{T-1}] 共 T 块，[λ_1 .. λ_T] 共 T 块
    n_blocks = 2 * horizon
    size = n_blocks * total
    M = np.zeros((size, size))
    b = np.zeros(size)

    def xs(k: int) -> Any:
        """第 k 个状态块（0 基）的行下标。"""
        return slice(k * total, (k + 1) * total)

    def ls(k: int) -> Any:
        """第 k 个协态块（对应 λ_{k+1}）的行下标。"""
        return slice((horizon + k) * total, (horizon + k + 1) * total)

    row = 0
    # 1) 初始条件：x_0 = x0
    M[row : row + total, xs(0)] = identity
    b[row : row + total] = x0
    row += total

    # 2) 状态方程：x_{k+1} = A x_k + B u_k，而 u_k = -Bᵀλ_{k+1}/(2ρ)
    #    => x_{k+1} - A x_k + C λ_{k+1} = 0   (k = 0 .. T-2)
    for k in range(horizon - 1):
        M[row : row + total, xs(k + 1)] = identity
        M[row : row + total, xs(k)] = -A
        M[row : row + total, ls(k)] = C
        row += total

    # 3) 终点条件：x_{T-1} = xf
    M[row : row + total, xs(horizon - 1)] = identity
    b[row : row + total] = xf
    row += total

    # 4) 协态方程：λ_k = -2S(x_k - xr) + Aᵀλ_{k+1}   (k = 1 .. T-1)
    for k in range(1, horizon):
        M[row : row + total, ls(k - 1)] = identity
        M[row : row + total, xs(k)] = D
        M[row : row + total, ls(k)] = -A.T
        b[row : row + total] = D @ xr
        row += total

    solution = spsolve(sparse.csc_matrix(M), sparse.csc_matrix(b.reshape(-1, 1)))
    values = np.asarray(solution).reshape(-1)

    states = np.vstack([values[xs(k)] for k in range(horizon)])
    costates = np.vstack([values[ls(k)] for k in range(horizon)])
    inputs = -(B.T @ costates.T).T / (2.0 * rho)

    times = np.arange(horizon, dtype=float)
    energy = float(np.sum(np.sum(inputs**2, axis=1)))
    final_error = float(np.linalg.norm(states[-1] - xf))
    residual = float(np.linalg.norm(M @ values - b))

    return OptimalControlResult(
        times=times,
        states=states,
        inputs=inputs,
        energy=energy,
        final_error=final_error,
        boundary_error=residual,
    )
