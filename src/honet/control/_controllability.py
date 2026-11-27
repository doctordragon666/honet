"""线性系统的可控性判据（论文 Lemma 1 / Lemma 3）。

三个判据互为依据，用于交叉验证：

* **Kalman 秩准则**：``rank[B, AB, ..., A^{Nn-1}B] = Nn``；
* **PBH 秩准则**：对 ``A`` 的每个特征值 ``λ`` 都有 ``rank[λI - A, B] = Nn``；
* **Gramian 判据**：有限时间可控性 Gramian ``W(T)`` 可逆。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require


def kalman_rank(A: Any, B: Any) -> int:
    """计算 Kalman 可控性矩阵 ``[B, AB, ..., A^{d-1}B]`` 的秩。

    参数:
        A: 系统矩阵，形状 ``(d, d)``。
        B: 输入矩阵，形状 ``(d, p)``。

    返回:
        可控性矩阵的秩（整数），满秩 ``d`` 即系统可控。

    示例:
        >>> import numpy as np
        >>> A = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> B = np.array([[0.0, 1.0], [1.0, 0.0]])
        >>> kalman_rank(A, B)
        2
    """
    np = require("numpy")
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if B.shape[0] != A.shape[0]:
        msg = f"维度不匹配：A 是 {A.shape}，B 是 {B.shape}"
        raise ValueError(msg)
    blocks = [B]
    current = B
    for _ in range(1, A.shape[0]):
        current = A @ current
        blocks.append(current)
    return int(np.linalg.matrix_rank(np.hstack(blocks)))


def is_controllable(A: Any, B: Any, *, tol: float = 1e-10) -> bool:
    """用 Kalman 秩准则判断系统是否可控。

    参数:
        A: 系统矩阵。
        B: 输入矩阵。
        tol: 秩判定的相对容差（按矩阵最大奇异值缩放）。

    返回:
        可控返回 ``True``。
    """
    np = require("numpy")
    A = np.atleast_2d(np.asarray(A, dtype=float))
    rank = kalman_rank(A, B)
    return rank == A.shape[0]


def pbh_rank(A: Any, B: Any) -> int:
    """PBH 判据：返回 ``min_λ rank[λI - A, B]``（满 ``d`` 即可控）。

    参数:
        A: 系统矩阵，形状 ``(d, d)``。
        B: 输入矩阵，形状 ``(d, p)``。

    返回:
        所有特征值处 ``[λI - A, B]`` 秩的最小值。

    示例:
        >>> import numpy as np
        >>> A = np.diag([1.0, 2.0])
        >>> B = np.array([[1.0], [1.0]])
        >>> pbh_rank(A, B)
        2
    """
    np = require("numpy")
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    dim = A.shape[0]
    eye = np.eye(dim)
    worst = dim
    for lam in np.linalg.eigvals(A):
        stacked = np.hstack([lam * eye - A, B])
        worst = min(worst, int(np.linalg.matrix_rank(stacked)))
    return worst


def gramian(A: Any, B: Any, T: float) -> Any:
    """计算有限时间可控性 Gramian ``W(T) = ∫_0^T e^{As} BBᵀ e^{Aᵀs} ds``。

    用块指数技巧精确求解（无需数值积分）::

        expm([[-A, BBᵀ], [0, Aᵀ]] · T) = [[·, H], [·, ·]]
        W(T) = e^{AT} · H

    参数:
        A: 系统矩阵，形状 ``(d, d)``。
        B: 输入矩阵，形状 ``(d, p)``。
        T: 时间上界（正数）。

    返回:
        对称半正定矩阵，形状 ``(d, d)``。
    """
    np = require("numpy")
    expm = require("scipy.linalg", attr="expm")
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    dim = A.shape[0]
    block = np.zeros((2 * dim, 2 * dim))
    block[:dim, :dim] = -A
    block[:dim, dim:] = B @ B.T
    block[dim:, dim:] = A.T
    mid = expm(block * T)[:dim, dim:]
    return expm(A * T) @ mid


def gramian_rank(A: Any, B: Any, T: float) -> int:
    """返回 Gramian 的数值秩（满秩即有限时间可控）。"""
    np = require("numpy")
    return int(np.linalg.matrix_rank(gramian(A, B, T)))
