"""单纯形控制：选择并牵制哪些 2-单纯形（周进等, 中国科学: 信息科学 2024）。

与超图控制层的区别：

* 超图控制层（:mod:`honet.control._dynamics`）用**广义 Laplacian** ``L`` 描述
  高阶耦合，回答「整个超图是否可控、如何驱动到目标」；
* 单纯形控制层（本模块）处理**牵制控制**：在单纯复形里选出若干 2-单纯形，
  加大其 1-单纯形耦合权重，使全网更快同步。核心是选择指标 ``Δ^s``。

选择指标（论文式 (18)）::

    Δ^s = Σ_{d=1..D} Σ_{(s,t) ∈ 该 d-单纯形} (v_s - v_t)²
        = 2 · Σ_{无序对 (s,t)} (v_s - v_t)²

其中 ``v`` 是 ``L`` 的 Fiedler 向量（最小非零特征值对应的特征向量）。``Δ^s``
越大，牵制该单纯形对 ``λ₂`` 的提升越多，同步越快。

**求和口径是最容易踩的坑**：``d=1`` 部分（3 条 1-单纯形）与 ``d=2`` 部分
（该 2-单纯形自身）各含同样的 3 个无序对，因此必须乘 2。只写一重求和得到的
恰好是论文值的一半——排序仍对、数值全错。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from honet._deps import require


@dataclass(frozen=True)
class SimplexRanking:
    """一个 2-单纯形的牵制优先级。

    属性:
        simplex: 该 2-单纯形的三个结点（升序）。
        delta: 选择指标 ``Δ^s``。
        rank: 排名（1 为最该牵制）。
    """

    simplex: tuple[int, int, int]
    delta: float
    rank: int


def generalized_laplacian(
    edges: Any,
    simplices: Any,
    num_nodes: int,
    *,
    c1: float = 1.0,
    c2: float = 1.0,
) -> Any:
    """构造单纯复形的广义 Laplacian ``L = c1·L⁽¹⁾ + c2·L⁽²⁾``。

    参数:
        edges: 1-单纯形（边）列表，每项是 ``(i, j)``。
        simplices: 2-单纯形（三角形）列表，每项是 ``(i, j, k)``。
        num_nodes: 结点数 ``N``。
        c1: 1-阶权重。
        c2: 2-阶权重。

    返回:
        对称矩阵，形状 ``(N, N)``，每行和为 0。

    约定:
        ``L⁽¹⁾`` 是图 Laplacian；``L⁽²⁾_ii = 2×(含 i 的三角形数)``、
        ``L⁽²⁾_ij = -(i, j 共处三角形数)``。索引均为 0 基。

    示例:
        >>> L = generalized_laplacian([(0, 1)], [(0, 1, 2)], 3)
        >>> abs(L.sum(axis=1)).max() < 1e-12
        True
    """
    np = require("numpy")
    total = int(num_nodes)
    adjacency = np.zeros((total, total), dtype=float)
    for i, j in edges:
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    laplacian1 = np.diag(adjacency.sum(axis=1)) - adjacency

    triangle_count = np.zeros(total, dtype=float)
    pair_count = np.zeros((total, total), dtype=float)
    for simplex in simplices:
        members = list(simplex)
        for node in members:
            triangle_count[node] += 1.0
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pair_count[members[a], members[b]] += 1.0
                pair_count[members[b], members[a]] += 1.0
    laplacian2 = np.diag(2.0 * triangle_count) - pair_count

    combined = c1 * laplacian1 + c2 * laplacian2
    return (combined + combined.T) / 2.0


def fiedler_vector(L: Any) -> Any:
    """返回 Laplacian 的 Fiedler 向量（最小非零特征值对应的特征向量）。

    参数:
        L: 对称 Laplacian，形状 ``(N, N)``。

    返回:
        单位化的 Fiedler 向量，形状 ``(N,)``。符号已归一（绝对值最大分量为正），
        便于跨实现比对。

    说明:
        若 ``λ₂`` 是重根，返回的是该特征子空间里的任意一个向量；调用方应先
        检查重数（见 :func:`algebraic_connectivity` 的返回值）。
    """
    np = require("numpy")
    values, vectors = np.linalg.eigh(np.asarray(L, dtype=float))
    # eigh 已按升序排列；跳过零特征值取第一个非零
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    idx = 1 if values.shape[0] > 1 else 0
    vec = vectors[:, idx]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    pivot = np.argmax(np.abs(vec))
    if vec[pivot] < 0:
        vec = -vec
    return vec


def algebraic_connectivity(L: Any) -> tuple[float, int]:
    """返回 ``(λ₂, λ₂ 的重数)``。

    参数:
        L: 对称 Laplacian。

    返回:
        ``(λ₂, 重数)`` 二元组。重数 > 1 说明 Fiedler 向量不唯一，``Δ^s``
        的排序会依赖选取（论文用 ``min_{j<=r}`` 处理这种情况）。

    示例:
        >>> L = generalized_laplacian([(0, 1), (1, 2)], [(0, 1, 2)], 3)
        >>> lam2, mult = algebraic_connectivity(L)
        >>> lam2 > 0 and mult >= 1
        True
    """
    np = require("numpy")
    values = np.sort(np.linalg.eigvalsh(np.asarray(L, dtype=float)))
    if values.shape[0] < 2:
        return 0.0, 1
    lam2 = float(values[1])
    multiplicity = int(np.sum(np.abs(values - lam2) < 1e-6))
    return lam2, multiplicity


def simplex_delta(simplex: Any, fiedler: Any) -> float:
    """计算一个单纯形的选择指标 ``Δ^s``。

    参数:
        simplex: 单纯形的结点下标序列（2-单纯形给三个，1-单纯形给两个）。
        fiedler: Fiedler 向量。

    返回:
        ``Δ^s = 2 · Σ_{无序对} (v_s - v_t)²``。

    示例:
        >>> import numpy as np
        >>> v = np.array([1.0, 0.0, -1.0])
        >>> simplex_delta((0, 1, 2), v)
        8.0
    """
    np = require("numpy")
    members = list(simplex)
    vec = np.asarray(fiedler, dtype=float)
    total = 0.0
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            diff = vec[members[a]] - vec[members[b]]
            total += float(diff * diff)
    return 2.0 * total


def rank_simplices(
    simplices: Any,
    fiedler: Any,
) -> list[SimplexRanking]:
    """按 ``Δ^s`` 从大到小对单纯形排序（越大越该牵制）。

    参数:
        simplices: 待排序的单纯形列表。
        fiedler: Fiedler 向量。

    返回:
        :class:`SimplexRanking` 列表，按 ``delta`` 降序，``rank`` 从 1 开始。

    示例:
        >>> import numpy as np
        >>> v = np.array([1.0, 0.0, -1.0, 0.0])
        >>> ranked = rank_simplices([(0, 1, 2), (0, 1, 3)], v)
        >>> ranked[0].rank
        1
        >>> ranked[0].delta > ranked[1].delta
        True
    """
    scored = [
        (tuple(sorted(int(x) for x in simplex)), simplex_delta(simplex, fiedler))
        for simplex in simplices
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [
        SimplexRanking(simplex=simplex, delta=delta, rank=index + 1)
        for index, (simplex, delta) in enumerate(scored)
    ]


def pinning_matrix(
    simplices: Any,
    num_nodes: int,
    *,
    weight: float = 1.0,
    include_simplex: bool = False,
) -> Any:
    """构造牵制矩阵 ``E``：在被牵制单纯形的 1-单纯形上加权重。

    论文做法是把被控 2-单纯形的三条 1-单纯形权重 ``1 → α``（``β`` 不变），
    等价于在 Laplacian 上叠加 ``α-1`` 倍的「该三角形的完全图 Laplacian」。

    **与定理 3 的系数关系**：定理 3 的一阶近似 ``Δλ₂ ≈ ε·Δ^s`` 对应的是
    1-单纯形与 2-单纯形权重**同时** ``+ε``；若只改 1-单纯形（论文图 2 的
    设定，``β`` 保持 1），则为 ``Δλ₂ ≈ ε·Δ^s / 2``。本函数的 ``weight``
    对应后者，需要前者时把 ``include_simplex`` 设为 ``True``。

    参数:
        simplices: 被牵制的单纯形列表。
        num_nodes: 结点数。
        weight: 叠加强度 ``α - 1``。
        include_simplex: 是否同时给 2-单纯形自身加权重（``β`` 也变）。
            为 ``True`` 时每对再额外加 ``weight``，使总效果翻倍。

    返回:
        对称矩阵，形状 ``(N, N)``，可直接加到 ``L`` 上。

    示例:
        >>> E = pinning_matrix([(0, 1, 2)], 3, weight=1.0)
        >>> E.shape
        (3, 3)
        >>> abs(E.sum(axis=1)).max() < 1e-12
        True
    """
    np = require("numpy")
    total = int(num_nodes)
    result = np.zeros((total, total), dtype=float)
    # 2-单纯形自身贡献的额外份数：include_simplex 时与 1-单纯形等量
    factor = 2.0 if include_simplex else 1.0
    for simplex in simplices:
        members = [int(x) for x in simplex]
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                result[i, j] -= factor * weight
                result[j, i] -= factor * weight
                result[i, i] += factor * weight
                result[j, j] += factor * weight
    return result
