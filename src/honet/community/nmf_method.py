"""非负矩阵分解（NMF）社区划分。"""

from __future__ import annotations

from typing import Any

from honet.community._methods import nmf_method as _impl


def nmf_method(
    hypergraph: Any,
    min_components: int = 2,
    max_components: int = 6,
    seed: int = 42,
) -> list[list[Any]] | None:
    """对关联矩阵做非负矩阵分解，按系数矩阵的最大分量给结点打社区标签。

    做法：对 ``n_components`` 从 ``min_components`` 到 ``max_components-1``
    逐一分解关联矩阵，用 ``W.argmax(axis=1)`` 得到每个结点的社区标签，再用
    超图模块度挑最优的划分。

    参数:
        hypergraph: hypernetx 超图对象。
        min_components: 尝试的最小社区数（含）。
        max_components: 尝试的最大社区数（不含）。
        seed: NMF 的 ``random_state``，默认 42（与旧实现一致）。

    返回:
        模块度最高的分区，``list[list]``，空社区会被剔除。若尝试范围内没有
        任何可用的社区数（例如结点数太少），返回 ``None``。

    注意:
        旧实现固定循环 ``range(3, min(7, len(nodes)))``，结点数不超过 3 时循环
        体一次都不执行、直接返回 ``None``，这是个实际缺陷。这里把上下界参数化
        并默认从 2 起，使 3 个及以上结点的超图都能得到结果。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["c", "d"]})
        >>> partition = nmf_method(H)
        >>> sorted(n for c in partition for n in c) == sorted(H.nodes)
        True
    """
    return _impl(
        hypergraph,
        min_components=min_components,
        max_components=max_components,
        seed=seed,
    )
