"""随机 k-均匀有向超图生成（实现）。

每条超边先抽 ``k`` 个结点作为 tail，再从剩余结点里抽 ``1..k-1`` 个作为 head，
因此 head 与 tail 不相交、head 非空。生成的边按 ``(head, tail)`` 去重，
保证不出现重复超边。
"""

from __future__ import annotations

import random


def random_uniform_hypergraph(
    n_local: int, k_local: int, m_local: int
) -> list[dict[str, list[int]]]:
    """生成随机 k-均匀有向超图。

    参数:
        n_local: 结点总数（编号 ``0..n_local-1``）。
        k_local: 每条超边 tail 的大小，必须满足 ``1 <= k_local < n_local``，
            否则剩余结点为空，循环无法推进。
        m_local: 需要生成的超边条数。

    返回:
        长度为 ``m_local`` 的列表，每项形如 ``{"head": [...], "tail": [...]}``，
        两个列表均按结点下标升序。

    抛出:
        ValueError: ``k_local`` 不在 ``1 <= k_local < n_local`` 范围内
            （旧实现会在这里死循环，见公开函数的说明）。
    """
    if not 1 <= k_local < n_local:
        msg = f"k_local 必须满足 1 <= k_local < n_local，收到 n_local={n_local}, k_local={k_local}"
        raise ValueError(msg)

    edge_set: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    edge_list_local: list[dict[str, list[int]]] = []
    while len(edge_list_local) < m_local:
        tail = tuple(sorted(random.sample(range(n_local), k_local)))
        remaining = sorted(set(range(n_local)) - set(tail))
        hsize = random.randint(1, min(k_local - 1, len(remaining)))
        head = tuple(sorted(random.sample(remaining, hsize)))
        edge = (head, tail)
        if edge not in edge_set:
            edge_set.add(edge)
            edge_list_local.append({"head": list(head), "tail": list(tail)})
    return edge_list_local
