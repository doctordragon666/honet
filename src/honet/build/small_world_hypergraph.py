"""生成小世界超图。

对应旧库 hypertrade ``honbuild.small_world_hypergraph``。先在环上按“每个结点
取其后 k 个邻居”生成规则点阵，再以 ``rewire_prob`` 的概率把 head 换成随机
远端点，制造长程捷径，从而得到高聚类 + 短平均距离的小世界结构。
"""

from __future__ import annotations

import random
from typing import Any

from honet.build._common import head_tail_hypergraph


def small_world_hypergraph(
    n: int, k: int, m: int, rewire_prob: float = 0.1, seed: int | None = 0
) -> Any:
    """生成小世界超图。

    参数:
        n: 环上结点数，编号 ``0..n-1``。
        k: 每个结点的邻居数；head 从中心点的 ``-2k..2k`` 邻域中取。
        m: 超边总数。
        rewire_prob: 重连概率，以该概率把 head 换成远离中心的随机结点。
        seed: 随机种子，固定时结果可复现。

    返回:
        ``hypernetx.Hypergraph``，超边 uid 为 ``e0``、``e1``……；每条超边的
        ``head`` / ``tail`` 原样保留为超边属性。

    抛出:
        ValueError: ``n < 2``、``k < 1``、``k >= n``、``m < 0``，或
            ``rewire_prob`` 不在 ``[0, 1]`` 内。
        ImportError: 未安装 hypernetx（需要 ``honet[analysis]``）。

    备注:
        **与旧实现的差异**：

        1. 旧代码用全局 ``random``，本实现用独立的 ``random.Random(seed)``，
           不污染全局状态；固定 ``seed`` 下可复现。
        2. 旧实现返回 head/tail 的**列表**，本实现返回 ``hypernetx.Hypergraph``
           （head/tail 仍作为超边属性可查）。
        3. 旧代码在 ``k >= n`` 时 ``remaining`` 恒为空、``while`` 循环会**死循环**；
           本实现改为提前抛 ``ValueError``。

    示例:
        >>> H = small_world_hypergraph(n=6, k=2, m=5, rewire_prob=0.2, seed=7)
        >>> H.shape[1]
        5
    """
    if n < 2:
        msg = f"n 必须 >= 2，实际为 {n}"
        raise ValueError(msg)
    if k < 1:
        msg = f"k 必须 >= 1，实际为 {k}"
        raise ValueError(msg)
    if k >= n:
        msg = f"k({k}) 必须小于 n({n})，否则邻域会占满全部结点"
        raise ValueError(msg)
    if m < 0:
        msg = f"m 必须 >= 0，实际为 {m}"
        raise ValueError(msg)
    if not 0.0 <= rewire_prob <= 1.0:
        msg = f"rewire_prob 必须在 [0, 1] 内，实际为 {rewire_prob}"
        raise ValueError(msg)

    rng = random.Random(seed)
    edge_list: list[dict[str, list[int]]] = []

    # 绕环逐点生成超边，直到凑够 m 条
    total_edges = 0
    while total_edges < m:
        for center in range(n):
            tail = [(center + offset) % n for offset in range(1, k + 1)]
            remaining = [node for node in range(n) if node not in set(tail) and node != center]
            if not remaining:
                continue

            if rng.random() < rewire_prob:
                # 长程捷径：在剩余结点里随机取一个
                head = [rng.choice(remaining)]
            else:
                # 优先取环上离中心较近的结点
                nearby = [(center + offset) % n for offset in range(-2 * k, 2 * k + 1)]
                nearby_remaining = [node for node in nearby if node in remaining]
                head = [rng.choice(nearby_remaining or remaining)]

            edge_list.append({"head": head, "tail": tail})
            total_edges += 1
            if total_edges >= m:
                break

    return head_tail_hypergraph(edge_list, name="small_world")
