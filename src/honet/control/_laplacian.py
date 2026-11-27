"""从超图结构构造 m-阶广义 Laplacian（论文 Definition 2/3）。

控制框架的耦合项完全由这里产出的矩阵决定，因此它是整个子包的基础。

数学约定（Ma et al., Physica A 653 (2024) 130108, Definition 2/3）::

    sigma_{i1 i2 ... ik} = 1/(k-1)  若超边 e 的结点集为 {i1..ik} 且 (i1,i2) ∈ D(e)
    L^(m)_ij = sum_e sigma_e(i,j)    (i != j)
    L^(m)_ii = -sum_{j != i} L^(m)_ij

其中 ``m`` 是「压缩阶」：对一条 k-元超边，压缩到 m 阶 Laplacian 时，把
k 个结点中除 i、j 之外的 k-2 个结点遍历求和，每个组合贡献 ``1/(k-1)``。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require


def hypergraph_laplacian(
    hyperedges: Any,
    num_nodes: int | None = None,
    *,
    order: int | None = None,
) -> Any:
    """构造超图的 m-阶广义 Laplacian 矩阵。

    参数:
        hyperedges: 超边定义。支持三种形式：

            * ``Hypergraph`` —— 直接取 ``H.edges`` 的成员集合（无向，全部
              结点对都视为有向对）；
            * ``{边名: (结点集, 有向对集)}`` —— 显式给出有向对 ``D(e)``，
              与论文 Definition 2 一致（有向超图）；
            * ``{边名: 结点集}`` —— 仅给结点集，等价于把其中所有有序对当作
              有向对（完全有向）。
        num_nodes: 结点总数。为 ``None`` 时按结点出现顺序自动编号。
        order: 压缩阶 ``m``。为 ``None`` 时返回**所有阶**的 Laplacian 字典
            ``{m: L^(m)}``；给出整数时只返回该阶矩阵。

    返回:
        ``order`` 为 ``None`` 时返回 ``{m: ndarray}``；否则返回 ``ndarray``，
        形状 ``(num_nodes, num_nodes)``，每行和为 0。

    示例:
        >>> import numpy as np
        >>> edges = {"e1": ({0, 1, 2}, {(0, 1), (1, 2)})}
        >>> L2 = hypergraph_laplacian(edges, num_nodes=3, order=2)
        >>> np.allclose(L2.sum(axis=1), 0)
        True
        >>> L2[0, 1] > 0
        True
    """
    np = require("numpy")

    node_index, edge_specs = _normalize_edges(hyperedges, num_nodes)
    total = len(node_index)

    # 每个压缩阶一个累加矩阵；对角线最后统一用「负行和」补齐
    accumulators: dict[int, Any] = {}
    for nodes, directed_pairs in edge_specs:
        k = len(nodes)
        if k < 2:
            continue
        weight = 1.0 / (k - 1)
        for source, target in directed_pairs:
            others = [v for v in nodes if v != source and v != target]
            # m=2 阶：每个「第三点」都对 (i, j) 累加一次（即 T2(i,j) += fac）
            for _ in others:
                _accumulate(accumulators, 2, total, np, node_index, source, target, weight, order)
            # m>=3 阶：项落在 (i, 第三点) 而非 (i, j)，且每个 others 元素贡献一次
            for third in others:
                _accumulate(accumulators, 3, total, np, node_index, source, third, weight, order)

    for m, mat in accumulators.items():
        row_sums = mat.sum(axis=1)
        mat[np.diag_indices_from(mat)] = -row_sums
        accumulators[m] = mat

    if order is not None:
        return accumulators.get(order, np.zeros((total, total), dtype=float))
    return accumulators


def _normalize_edges(
    hyperedges: Any,
    num_nodes: int | None,
) -> tuple[dict[Any, int], list[tuple[list[Any], list[tuple[Any, Any]]]]]:
    """把各种超边写法归一成 ``(结点→下标, [(结点列表, 有向对列表)])``。"""
    if hasattr(hyperedges, "edges") and hasattr(hyperedges, "nodes"):
        nodes = list(hyperedges.nodes)
        index = {node: i for i, node in enumerate(nodes)}
        specs: list[tuple[list[Any], list[tuple[Any, Any]]]] = []
        for edge in hyperedges.edges:
            members = list(hyperedges.edges[edge])
            # 无向超图：所有有序对都视为有向对
            pairs = [(u, v) for u in members for v in members if u != v]
            specs.append((members, pairs))
        return index, specs

    raw = list(hyperedges.items()) if isinstance(hyperedges, dict) else list(hyperedges)
    seen: list[Any] = []
    for _, spec in raw:
        members = spec[0] if isinstance(spec, tuple) else spec
        for node in members:
            if node not in seen:
                seen.append(node)
    if num_nodes is not None:
        while len(seen) < num_nodes:
            seen.append(len(seen))
        seen = seen[:num_nodes]
    index = {node: i for i, node in enumerate(seen)}

    specs = []
    for _, spec in raw:
        if isinstance(spec, tuple):
            members, pairs = spec[0], spec[1]
            pairs = list(pairs)
        else:
            members = list(spec)
            pairs = [(u, v) for u in members for v in members if u != v]
        specs.append((list(members), pairs))
    return index, specs


def _accumulate(
    accumulators: dict[int, Any],
    order: int,
    total: int,
    np: Any,
    node_index: dict[Any, int],
    source: Any,
    target: Any,
    weight: float,
    wanted: int | None,
) -> None:
    """把 ``weight`` 累加到 ``accumulators[order]`` 的 ``(source, target)`` 项。"""
    if wanted is not None and order != wanted:
        return
    mat = accumulators.get(order)
    if mat is None:
        mat = np.zeros((total, total), dtype=float)
        accumulators[order] = mat
    mat[node_index[source], node_index[target]] += weight


def _subsets(items: list[Any]) -> list[list[Any]]:
    """返回 ``items`` 的全部子集（按大小排序，保证压缩阶从小到大）。"""
    result: list[list[Any]] = [[]]
    for item in items:
        result.extend([existing + [item] for existing in result])
    return result
