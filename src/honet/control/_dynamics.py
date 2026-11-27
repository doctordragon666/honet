"""超图网络动力学模型（论文式 (4)）。

把「结点自身动力学 + 高阶耦合 + 外部控制输入」组装成一个线性系统::

    Ẋ(t) = 𝒜 X(t) + ℬ U(t)
    𝒜 = I_N ⊗ A + c/(k-1) · L ⊗ Λ
    ℬ = Δ ⊗ B

其中 ``A`` 是结点自身动力学（``n×n``）、``Λ`` 是内耦合矩阵、``L`` 是
m-阶广义 Laplacian、``Δ = diag(δ)`` 标记哪些结点受外部控制、``B`` 是
输入矩阵。这正是当前 ``honet.control`` 缺失的部分——旧实现只给出「控制
哪些结点」，没有动力学与耦合。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from honet._deps import require


@dataclass(frozen=True)
class HypergraphControlSystem:
    """超图网络控制系统 ``(𝒜, ℬ)``。

    属性:
        A: 结点自身动力学矩阵，形状 ``(n, n)``。
        L: 广义 Laplacian，形状 ``(N, N)``。
        Lambda: 内耦合矩阵，形状 ``(n, n)``。
        B: 输入矩阵，形状 ``(n, p)``。
        controlled_nodes: 受外部控制的结点下标；``None`` 表示全部受控。
        coupling: 耦合强度 ``c``。
        order: 超边阶数 ``k``（用于 ``c/(k-1)`` 归一化）。
        num_nodes: 结点数 ``N``；``None`` 时取 ``L.shape[0]``。
    """

    A: Any
    L: Any
    Lambda: Any
    B: Any
    controlled_nodes: tuple[int, ...] | None = None
    coupling: float = 1.0
    order: int = 3
    num_nodes: int | None = None

    @property
    def N(self) -> int:
        """结点数。"""
        return int(self.num_nodes if self.num_nodes is not None else self.L.shape[0])

    @property
    def n(self) -> int:
        """单个结点的状态维数。"""
        return int(self.A.shape[0])

    @property
    def p(self) -> int:
        """每个受控结点的输入通道数。"""
        return int(self.B.shape[1])

    @property
    def delta(self) -> Any:
        """控制指示向量 ``δ``：受控结点为 1，其余为 0。"""
        np = require("numpy")
        if self.controlled_nodes is None:
            return np.ones(self.N)
        mask = np.zeros(self.N)
        mask[list(self.controlled_nodes)] = 1.0
        return mask

    @property
    def controlled(self) -> tuple[int, ...]:
        """实际受控的结点下标元组。"""
        if self.controlled_nodes is None:
            return tuple(range(self.N))
        return tuple(self.controlled_nodes)

    def coupling_term(self) -> Any:
        """返回 ``c/(k-1) · L ⊗ Λ``，即高阶耦合项。"""
        np = require("numpy")
        scale = self.coupling / (self.order - 1)
        return scale * np.kron(self.L, self.Lambda)

    def state_matrix(self) -> Any:
        """返回系统矩阵 ``𝒜 = I_N ⊗ A + c/(k-1)·L ⊗ Λ``。"""
        np = require("numpy")
        return np.kron(np.eye(self.N), self.A) + self.coupling_term()

    def input_matrix(self) -> Any:
        """返回输入矩阵 ``ℬ = Δ ⊗ B``。"""
        np = require("numpy")
        return np.kron(np.diag(self.delta), self.B)

    def to_blocks(self) -> tuple[Any, Any]:
        """返回 ``(𝒜, ℬ)`` 二元组，便于直接喂给求解器。"""
        return self.state_matrix(), self.input_matrix()

    def flatten(self, X: Any) -> Any:
        """把 ``(N, n)`` 的逐结点状态堆叠成 ``(N·n,)`` 的列向量。"""
        np = require("numpy")
        return np.asarray(X, dtype=float).reshape(-1)

    def unflatten(self, X: Any) -> Any:
        """把 ``(N·n,)`` 的列向量还原成 ``(N, n)`` 的逐结点状态。"""
        np = require("numpy")
        return np.asarray(X, dtype=float).reshape(self.N, self.n)
