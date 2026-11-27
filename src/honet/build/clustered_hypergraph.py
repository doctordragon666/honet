"""生成带社区结构的聚簇超图。

对应旧库 hypertrade ``honbuild.clustered_hypergraph``。结点被均分成
``num_clusters`` 个簇，约 80% 的超边在簇内抽取，其余 20% 作为跨簇桥接，
从而同时具备高内聚与少量长程连接。
"""

from __future__ import annotations

import random
from typing import Any

from honet.build._common import head_tail_hypergraph, sample_head


def clustered_hypergraph(
    n: int, k: int, m: int, num_clusters: int = 5, seed: int | None = 0
) -> Any:
    """生成带社区结构的超图。

    参数:
        n: 结点数，编号 ``0..n-1``。
        k: 每条超边 tail 的结点数；head 取 ``1..k-1`` 个。
        m: 超边总数。
        num_clusters: 簇数；结点按顺序均分，除不尽的余数归入最后一个簇。
        seed: 随机种子，固定时结果可复现。

    返回:
        ``hypernetx.Hypergraph``，超边 uid 为 ``e0``、``e1``……；每条超边的
        ``head`` / ``tail`` 原样保留为超边属性。

    抛出:
        ValueError: ``n < 2``、``k < 2``、``k > n``、``m < 0``，或
            ``num_clusters < 1``。
        ImportError: 未安装 hypernetx（需要 ``honet[analysis]``）。

    备注:
        **与旧实现的差异**：

        1. 旧代码用全局 ``random``，本实现用独立的 ``random.Random(seed)``，
           不污染全局状态；固定 ``seed`` 下可复现。
        2. 旧实现返回 head/tail 的**列表**，本实现返回 ``hypernetx.Hypergraph``
           （head/tail 仍作为超边属性可查）。
        3. 旧代码在 ``n < k`` 时抛 ``ValueError: Sample larger than population``，
           本实现提前校验并给出带参数名的提示。

    示例:
        >>> H = clustered_hypergraph(n=6, k=2, m=4, num_clusters=2, seed=5)
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
    if num_clusters < 1:
        msg = f"num_clusters 必须 >= 1，实际为 {num_clusters}"
        raise ValueError(msg)

    rng = random.Random(seed)
    cluster_size = n // num_clusters
    clusters = [
        list(range(i * cluster_size, min((i + 1) * cluster_size, n))) for i in range(num_clusters)
    ]
    # 除不尽的余数结点归入最后一个簇
    for node in range(num_clusters * cluster_size, n):
        clusters[-1].append(node)

    intra_prob = 0.8
    edge_list: list[dict[str, list[int]]] = []

    for _ in range(m):
        if rng.random() < intra_prob:
            cluster = rng.choice(clusters)
            if len(cluster) >= k + 1:
                tail = rng.sample(cluster, k)
                chosen = set(tail)
                remaining = [node for node in cluster if node not in chosen]
            else:
                # 簇太小，退回全图抽样
                tail = rng.sample(range(n), k)
                chosen = set(tail)
                remaining = [node for node in range(n) if node not in chosen]
        else:
            # 跨簇桥接边
            tail = rng.sample(range(n), k)
            chosen = set(tail)
            remaining = [node for node in range(n) if node not in chosen]

        edge_list.append({"head": sample_head(rng, remaining, k), "tail": tail})

    return head_tail_hypergraph(edge_list, name="clustered")
