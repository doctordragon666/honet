"""连续/离散时间线性系统的稳态与能量分析工具。

本模块补上 :mod:`honet.control._controllability` 缺少的两块能力（参考 nctpy）：

* **无限时域 Gramian**——有限时域 Gramian 需要积分，无限时域则退化为求解
  Lyapunov 方程（仅对稳定系统有解）；
* **矩阵归一化**——把邻接矩阵缩放成适合线性动力学的形式，连续时间与离散
  时间的归一化方式不同。

nctpy 的 ``matrix_normalization`` 用「谱半径 + 常数」归一化，本实现保持同一
口径以便数值对照：``A_norm = A / (c + |λ|_max)``，连续时间再减去单位阵。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require


def matrix_normalization(A: Any, system: str, c: float = 1.0) -> Any:
    """把邻接矩阵归一化为适合线性动力学分析的形式。

    参数:
        A: 邻接矩阵（结构连接矩阵），形状 ``(N, N)``。
        system: ``"continuous"`` 或 ``"discrete"``。
        c: 归一化常数，默认 1.0。

    返回:
        归一化后的矩阵。离散时间：``A / (c + λmax)``；
        连续时间：``A / (c + λmax) - I`` （保证稳定）。

    抛出:
        ValueError: ``system`` 取值非法。

    示例:
        >>> import numpy as np
        >>> A = np.array([[0.0, 1.0], [1.0, 0.0]])
        >>> A_norm = matrix_normalization(A, "continuous")
        >>> np.max(np.real(np.linalg.eigvals(A_norm))) < 0
        True
    """
    np = require("numpy")
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.shape[0] != A.shape[1]:
        msg = f"A 必须是方阵，实际是 {A.shape}"
        raise ValueError(msg)
    if system not in ("continuous", "discrete"):
        msg = f"system 必须是 'continuous' 或 'discrete'，实际是 {system!r}"
        raise ValueError(msg)

    spectral = float(np.abs(np.linalg.eigvals(A)).max())
    normalized = A / (c + spectral)
    if system == "continuous":
        normalized = normalized - np.eye(A.shape[0])
    return normalized


def is_stable(A: Any, system: str) -> bool:
    """判断线性系统是否稳定。

    参数:
        A: 系统矩阵。
        system: ``"continuous"`` （要求全部特征值实部 < 0）或
            ``"discrete"`` （要求全部特征值模 < 1）。

    返回:
        稳定返回 ``True``。
    """
    np = require("numpy")
    eigenvalues = np.linalg.eigvals(np.atleast_2d(np.asarray(A, dtype=float)))
    if system == "continuous":
        return bool(np.max(np.real(eigenvalues)) < 0)
    if system == "discrete":
        return bool(np.max(np.abs(eigenvalues)) < 1)
    msg = f"system 必须是 'continuous' 或 'discrete'，实际是 {system!r}"
    raise ValueError(msg)


def infinite_gramian(A: Any, B: Any, system: str) -> Any:
    """计算无限时域可控性 Gramian。

    连续时间（稳定系统）::

        W = ∫_0^∞ e^{As} BBᵀ e^{Aᵀs} ds   ⟺   A W + W Aᵀ + BBᵀ = 0

    离散时间（稳定系统）::

        W = Σ_{k=0}^∞ A^k BBᵀ (Aᵀ)^k   ⟺   W - A W Aᵀ = BBᵀ

    两者都通过 Lyapunov 方程求解，无需数值积分。

    参数:
        A: 系统矩阵，形状 ``(N, N)``。
        B: 输入矩阵，形状 ``(N, p)``。
        system: ``"continuous"`` 或 ``"discrete"``。

    返回:
        Gramian 矩阵，形状 ``(N, N)``。

    抛出:
        ValueError: 系统不稳定（无限时域 Gramian 不存在）。

    示例:
        >>> import numpy as np
        >>> A = np.array([[-1.0, 0.0], [0.0, -1.0]])
        >>> W = infinite_gramian(A, np.eye(2), "continuous")
        >>> np.allclose(W, 0.5 * np.eye(2))
        True
    """
    np = require("numpy")
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if not is_stable(A, system):
        msg = (
            f"系统不稳定（system={system!r}），无限时域 Gramian 不存在；"
            f"请改用有限时域 gramian(A, B, T)"
        )
        raise ValueError(msg)

    BB = B @ B.T
    if system == "continuous":
        lyapunov = require("scipy.linalg", attr="solve_continuous_lyapunov")
        return lyapunov(A, -BB)
    lyapunov = require("scipy.linalg", attr="solve_discrete_lyapunov")
    return lyapunov(A, BB)


def gramian_auto(A: Any, B: Any, T: float | None, system: str) -> Any:
    """按 ``T`` 自动选择有限时域或无限时域 Gramian。

    参数:
        A: 系统矩阵。
        B: 输入矩阵。
        T: 时间上界；``None`` 或 ``float("inf")`` 表示无限时域。
        system: ``"continuous"`` 或 ``"discrete"``。

    返回:
        Gramian 矩阵。

    示例:
        >>> import numpy as np
        >>> A = np.diag([-1.0, -2.0])
        >>> W = gramian_auto(A, np.eye(2), None, "continuous")
        >>> W.shape
        (2, 2)
    """
    from honet.control._controllability import gramian

    if T is None or T == float("inf"):
        return infinite_gramian(A, B, system)
    if system == "discrete":
        np = require("numpy")
        A = np.atleast_2d(np.asarray(A, dtype=float))
        B = np.atleast_2d(np.asarray(B, dtype=float))
        total = np.zeros((A.shape[0], A.shape[0]))
        power = np.eye(A.shape[0])
        for _ in range(int(T)):
            total = total + power @ B @ B.T @ power.T
            power = power @ A
        return total
    return gramian(A, B, float(T))
