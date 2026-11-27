"""批量控制任务：一次算多个状态转移的控制能量（参考 nctpy pipelines）。

在脑网络控制文献里，常见的分析不是「一次转移」，而是「所有状态对之间的
转移」——例如把 65 个脑区按功能分成若干系统，考察从系统 A 转到系统 B 需要
多少控制能量。这类分析需要：

1. 把一个「标签向量」展开成所有状态对的初态/终态矩阵
   （nctpy ``expand_states``）；
2. 对每个转移算控制能量，汇总成矩阵。

本模块把这两步封装成 :func:`control_energy_matrix`，避免调用方写嵌套循环。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from honet._deps import require


@dataclass(frozen=True)
class BatchEnergyResult:
    """批量控制能量结果。

    属性:
        energies: 控制能量矩阵，形状 ``(n_states, n_states)``；
            ``energies[i, j]`` 是从状态 ``i`` 转移到状态 ``j`` 的能量。
        labels: 状态标签列表，长度 ``n_states``。
        transitions: ``[(i, j), ...]``，与矩阵下三角展开顺序一致。
    """

    energies: Any
    labels: list[str]
    transitions: list[tuple[int, int]]

    def as_dict(self) -> dict[tuple[str, str], float]:
        """把能量矩阵转成 ``{(源标签, 目标标签): 能量}`` 的字典。"""
        result: dict[tuple[str, str], float] = {}
        for (i, j), value in zip(self.transitions, self.energies.ravel(), strict=True):
            result[(self.labels[i], self.labels[j])] = float(value)
        return result

    def hardest(self) -> tuple[str, str, float]:
        """返回能量最高的转移（最难驱动的那一对）。"""
        flat = [
            (value, self.labels[i], self.labels[j])
            for (i, j), value in zip(self.transitions, self.energies.ravel(), strict=True)
        ]
        value, source, target = max(flat, key=lambda item: item[0])
        return source, target, value

    def easiest(self) -> tuple[str, str, float]:
        """返回能量最低的转移（最容易驱动的那一对）。"""
        flat = [
            (value, self.labels[i], self.labels[j])
            for (i, j), value in zip(self.transitions, self.energies.ravel(), strict=True)
        ]
        value, source, target = min(flat, key=lambda item: item[0])
        return source, target, value


def expand_states(labels: Any) -> tuple[Any, Any, list[str]]:
    """把标签向量展开成所有状态对的初态/终态矩阵。

    参数:
        labels: 长度 ``N`` 的标签序列，每个元素标记该结点所属的状态。
            例如 ``["A", "A", "B", "B"]`` 表示前两个结点属状态 A、后两个属 B。

    返回:
        ``(x0_matrix, xf_matrix, unique_labels)`` 三元组：

        * ``x0_matrix``：形状 ``(N, n_states)`` 的布尔矩阵，第 ``k`` 列标记
          状态 ``k`` 包含哪些结点；
        * ``xf_matrix``：同上，用于目标状态；
        * ``unique_labels``：去重后的标签列表（排序后）。

    示例:
        >>> x0, xf, names = expand_states(["A", "A", "B", "B"])
        >>> names
        ['A', 'B']
        >>> x0[:, 0].tolist()
        [True, True, False, False]
        >>> xf[:, 1].tolist()
        [False, False, True, True]
    """
    np = require("numpy")
    items = list(labels)
    if not items:
        msg = "labels 不能为空"
        raise ValueError(msg)
    unique = sorted({str(item) for item in items})
    total = len(items)
    matrix = np.zeros((total, len(unique)), dtype=bool)
    for column, name in enumerate(unique):
        for row, item in enumerate(items):
            matrix[row, column] = str(item) == name
    return matrix, matrix.copy(), unique


def control_energy_matrix(
    A: Any,
    B: Any,
    labels: Any,
    T: float,
    *,
    normalize_state: bool = True,
) -> BatchEnergyResult:
    """计算所有状态对之间转移的最小控制能量。

    参数:
        A: 系统矩阵，形状 ``(N, N)``（通常已用 ``matrix_normalization`` 归一化）。
        B: 控制输入矩阵，形状 ``(N, N)``。
        labels: 长度 ``N`` 的状态标签序列。
        T: 时间上界。
        normalize_state: 是否把每个状态的初态/终态向量单位化。默认 ``True``
            （与 nctpy 的做法一致），使不同规模的系统之间可比。

    返回:
        :class:`BatchEnergyResult`。

    抛出:
        ValueError: 维度不匹配。

    示例:
        >>> import numpy as np
        >>> A = np.diag([-1.0, -2.0, -3.0, -4.0])
        >>> B = np.eye(4)
        >>> result = control_energy_matrix(A, B, ["A", "A", "B", "B"], 1.0)
        >>> result.energies.shape
        (2, 2)

    说明:
        每个转移的能量用最小能量公式 ``reachᵀ W(T)⁻¹ reach`` 计算，其中
        ``reach = x_f - e^{AT} x_0``。这比逐条反解输入再积分快得多，且
        数值上等价（测试已验证）。
    """
    np = require("numpy")
    expm = require("scipy.linalg", attr="expm")

    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if A.shape[0] != A.shape[1]:
        msg = f"A 必须是方阵，实际是 {A.shape}"
        raise ValueError(msg)
    if B.shape[0] != A.shape[0]:
        msg = f"B 的行数必须等于 A 的阶数：A {A.shape}, B {B.shape}"
        raise ValueError(msg)

    x0_matrix, xf_matrix, unique = expand_states(labels)
    if x0_matrix.shape[0] != A.shape[0]:
        msg = f"labels 的长度应为 {A.shape[0]}，实际是 {x0_matrix.shape[0]}"
        raise ValueError(msg)

    from honet.control._controllability import gramian

    W = gramian(A, B, T)
    # 用 SVD 求伪逆：不可控方向上的分量自动截断，避免 inv 放大数值噪声
    u_mat, s_vals, vt_mat = np.linalg.svd(W)
    cutoff = 1e-12 * (s_vals[0] if s_vals.size else 0.0)
    inv_s = np.array([1.0 / s if s > cutoff else 0.0 for s in s_vals])
    W_pinv = (vt_mat.T * inv_s) @ u_mat.T
    propagator = expm(A * T)

    count = len(unique)
    energies = np.zeros((count, count))
    transitions: list[tuple[int, int]] = []
    for i in range(count):
        start = x0_matrix[:, i].astype(float)
        if normalize_state:
            norm = np.linalg.norm(start)
            if norm > 0:
                start = start / norm
        for j in range(count):
            target = xf_matrix[:, j].astype(float)
            if normalize_state:
                norm = np.linalg.norm(target)
                if norm > 0:
                    target = target / norm
            reach = target - propagator @ start
            energies[i, j] = float(reach @ W_pinv @ reach)
            transitions.append((i, j))

    return BatchEnergyResult(energies=energies, labels=unique, transitions=transitions)
