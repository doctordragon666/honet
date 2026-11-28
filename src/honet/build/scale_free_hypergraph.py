"""按优先连接（preferential attachment）生成无标度超图。

对应旧库 hypertrade ``honbuild.scale_free_hypergraph``。口径：结点度为
``(degree + 1) ** gamma`` 时更可能被选进新的超边，从而形成幂律度分布。
"""

from __future__ import annotations

import random
from typing import Any

from honet.build._common import head_tail_hypergraph, numpy, sample_head


def scale_free_hypergraph(n: int, k: int, m: int, gamma: float = 2.5, seed: int | None = 0) -> Any:
    """用优先连接生成无标度超图。

    参数:
        n: 结点数，编号 ``0..n-1``。
        k: 每条超边 tail 的结点数；head 取 ``1..k-1`` 个。
        m: 超边总数。
        gamma: 优先连接的幂指数，越大越偏向高度结点。
        seed: 随机种子，固定时结果可复现。

    返回:
        ``hypernetx.Hypergraph``，超边 uid 为 ``e0``、``e1``……；每条超边的
        ``head`` / ``tail`` 原样保留为超边属性。

    抛出:
        ValueError: ``n < 2``、``k < 2``、``k > n``、``m < 0``，或
            ``gamma <= 0`` （否则概率分布退化）。
        ImportError: 未安装 hypernetx / numpy（需要 ``honet[analysis]``）。

    备注:
        **与旧实现的差异**：

        1. 旧代码用全局 ``random`` 与 ``np.random``，本实现改用
           ``random.Random(seed)`` 与 ``np.random.default_rng(seed)`` 两个独立
           随机源，不污染全局状态；固定 ``seed`` 下可复现。
        2. 旧实现返回 ``{"head": [...], "tail": [...]}`` 的**列表**；本实现返回
           可直接使用的 ``hypernetx.Hypergraph`` （head/tail 仍作为超边属性可查）。
        3. 旧代码在 ``n <= k`` 时 ``random.sample`` 会抛
           ``ValueError: Sample larger than population``，本实现提前校验并给出
           带参数名的提示。

    示例:
        >>> H = scale_free_hypergraph(n=6, k=2, m=4, seed=3)
        >>> H.shape[1]
        4
    """
    if n < 2:
        msg = f"n 必须 >= 2，实际为 {n}"
        raise ValueError(msg)
    if k < 2:
        msg = f"k 必须 >= 2，实际为 {k}"
        raise ValueError(msg)
    if k > n:
        msg = f"k({k}) 不能大于 n({n})，否则无法无放回抽样"
        raise ValueError(msg)
    if m < 0:
        msg = f"m 必须 >= 0，实际为 {m}"
        raise ValueError(msg)
    if gamma <= 0:
        msg = f"gamma 必须 > 0，实际为 {gamma}"
        raise ValueError(msg)

    rng = random.Random(seed)
    np_rng = numpy().random.default_rng(seed)
    degrees = [0] * n
    edge_list: list[dict[str, list[int]]] = []

    def _record(tail: list[int], head: list[int]) -> None:
        """登记一条超边并更新两端结点的度。"""
        edge_list.append({"head": head, "tail": tail})
        for node in [*tail, *head]:
            degrees[node] += 1

    # 先用少量随机超边起头，保证每个结点都有被选中的机会
    for _ in range(min(5, m)):
        tail = rng.sample(range(n), k)
        chosen = set(tail)
        remaining = [node for node in range(n) if node not in chosen]
        _record(tail, sample_head(rng, remaining, k))

    while len(edge_list) < m:
        probabilities = [(degree + 1) ** gamma for degree in degrees]
        total = sum(probabilities)
        tail = [
            int(node)
            for node in np_rng.choice(
                n, size=k, replace=False, p=[p / total for p in probabilities]
            )
        ]

        chosen = set(tail)
        remaining = [node for node in range(n) if node not in chosen]
        if not remaining:
            continue
        remaining_probabilities = [(degrees[node] + 1) ** gamma for node in remaining]
        remaining_total = sum(remaining_probabilities)
        hsize = rng.randint(1, min(k - 1, len(remaining)))
        head = [
            int(node)
            for node in np_rng.choice(
                remaining,
                size=hsize,
                replace=False,
                p=[p / remaining_total for p in remaining_probabilities],
            )
        ]
        _record(tail, head)

    return head_tail_hypergraph(edge_list, name="scale_free")
