"""网络可控性度量：平均可控性与模态可控性（参考 nctpy / Gu et al. 2015）。

这两个指标回答的是「**哪些结点是好的控制入口**」，与拓扑层的 dilation 分析
互补：后者看结构可达性，这里看能量代价与模态可驱动性。

* **平均可控性**（average controllability）：把某个结点作为唯一控制入口时，
  驱动系统到随机目标所需的平均能量。等于无限时域 Gramian 的对角元。
* **模态可控性**（modal controllability）：某结点驱动系统「难控模态」的能力，
  对每个模态按 ``1 - λᵢ²`` 加权。

参考: Gu, Pasqualetti, Cieslak, Telesford, Yu, Kahn, Medaglia, Vettel,
Miller, Grafton & Bassett, *Nature Communications* 6:8414, 2015.
"""

from __future__ import annotations

from typing import Any

from honet._deps import require
from honet.control._steady_state import infinite_gramian, is_stable


def average_controllability(A: Any, system: str = "continuous") -> Any:
    """计算每个结点的平均可控性。

    参数:
        A: 归一化后的系统矩阵，形状 ``(N, N)``。
        system: ``"continuous"`` 或 ``"discrete"``。

    返回:
        长度 ``N`` 的数组，第 ``i`` 项是结点 ``i`` 作为唯一控制入口时的
        平均可控性。数值越大越适合作为控制入口。

    抛出:
        ValueError: 系统不稳定（连续时间要求谱实部为负，离散时间要求模 < 1）。

    示例:
        >>> import numpy as np
        >>> A = np.diag([-1.0, -2.0])
        >>> ac = average_controllability(A, "continuous")
        >>> np.allclose(ac, [0.5, 0.25])
        True

    说明:
        连续时间下平均可控性等于无限时域 Gramian 的对角元
        （``W = ∫₀^∞ e^{As}BBᵀe^{Aᵀs}ds``，取 ``B = I`` 后逐结点取对角）。

        离散时间下用 Schur 分解的闭式解::

            ac_i = Σ_j |U_ij|² / (1 - λ_j²)

        其中 ``A = U T Uᵀ`` 是实 Schur 分解。
    """
    np = require("numpy")
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.shape[0] != A.shape[1]:
        msg = f"A 必须是方阵，实际是 {A.shape}"
        raise ValueError(msg)
    if system not in ("continuous", "discrete"):
        msg = f"system 必须是 'continuous' 或 'discrete'，实际是 {system!r}"
        raise ValueError(msg)
    if not is_stable(A, system):
        msg = (
            f"系统不稳定（system={system!r}），平均可控性无定义；请先用 matrix_normalization 归一化"
        )
        raise ValueError(msg)

    if system == "continuous":
        W = infinite_gramian(A, np.eye(A.shape[0]), "continuous")
        return np.diag(W)

    # 离散时间：实 Schur 分解的闭式解
    schur = require("scipy.linalg", attr="schur")
    T_mat, U = schur(A, output="real")
    eigvals = np.diag(T_mat)
    # ac_i = Σ_j |U_ij|² / (1 - λ_j²)
    # 注意分母只依赖单个模态的 λ_j，不能写成 1 - λ_a·λ_b（那是另一种量）。
    modal_weight = np.multiply(U, U).T  # modal_weight[j, i] = |U_ij|²
    denominator = 1.0 - np.square(eigvals)[:, None]  # denominator[j, :] = 1 - λ_j²
    return np.sum(np.divide(modal_weight, denominator), axis=0)


def modal_controllability(A: Any) -> Any:
    """计算每个结点的模态可控性（离散时间系统）。

    参数:
        A: 归一化后的离散时间系统矩阵，形状 ``(N, N)``。

    返回:
        长度 ``N`` 的数组，第 ``i`` 项是结点 ``i`` 的模态可控性。

    示例:
        >>> import numpy as np
        >>> A = np.diag([0.5, 0.2])
        >>> phi = modal_controllability(A)
        >>> phi.shape
        (2,)

    说明:
        公式为 ``φ_i = Σ_j |U_ij|² (1 - λ_j²)``，其中 ``A = U T Uᵀ`` 是实
        Schur 分解。权重 ``1 - λ²`` 强调「难控模态」（``λ`` 接近 ±1）。

        与 nctpy 一致，本函数只支持离散时间（``A`` 的特征值模应 < 1）；
        连续时间变体需要先把 ``A`` 离散化。
    """
    np = require("numpy")
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.shape[0] != A.shape[1]:
        msg = f"A 必须是方阵，实际是 {A.shape}"
        raise ValueError(msg)

    schur = require("scipy.linalg", attr="schur")
    T_mat, U = schur(A, output="real")
    eigvals = np.diag(T_mat)
    # φ_i = Σ_j |U_ij|² (1 - λ_j²)
    modal_weight = np.multiply(U, U)  # |U_ij|²
    weights = 1.0 - np.square(eigvals)
    return modal_weight @ weights


def control_energy_ranking(A: Any, system: str = "continuous") -> list[tuple[int, float]]:
    """按平均可控性从高到低给出结点排名。

    参数:
        A: 归一化后的系统矩阵。
        system: ``"continuous"`` 或 ``"discrete"``。

    返回:
        ``[(结点下标, 平均可控性), ...]``，按可控性降序。

    示例:
        >>> import numpy as np
        >>> A = np.diag([-1.0, -3.0])
        >>> ranking = control_energy_ranking(A)
        >>> ranking[0][0]
        0
    """
    ac = average_controllability(A, system)
    order = sorted(range(len(ac)), key=lambda i: ac[i], reverse=True)
    return [(int(i), float(ac[i])) for i in order]
