"""生成均匀随机的 hypernetx 超图。

对应旧库 hypertrade ``honbuild.generate_random_hypernet``：结点固定为
``0..n_nodes-1``，每条超边的大小在 ``[2, max_edge_size]`` 上均匀抽取，
成员在结点集上无放回抽样。用于给算法压测提供可控规模的输入。
"""

from __future__ import annotations

import random
from typing import Any

from honet.build._common import hypernetx


def generate_random_hypernet(
    n_nodes: int = 100, n_edges: int = 200, max_edge_size: int = 4, seed: int | None = 0
) -> Any:
    """生成随机超图。

    参数:
        n_nodes: 结点数，结点编号为 ``0..n_nodes-1``。
        n_edges: 超边数。
        max_edge_size: 超边大小的上界（含）；下界固定为 2。
        seed: 随机种子。给定时结果可复现；为 ``None`` 时不固定种子。

    返回:
        ``hypernetx.Hypergraph``，超边 uid 为 ``0..n_edges-1``。

    抛出:
        ValueError: ``max_edge_size < 2``，或 ``n_nodes < max_edge_size``
            （超边大小可能超过结点总数，无法无放回抽样）。
        ImportError: 未安装 hypernetx（需要 ``honet[analysis]``）。

    备注:
        **与旧实现的差异**：旧代码用模块级 ``random.seed(seed)``，会污染调用方
        的全局随机状态；本实现改用独立的 ``random.Random(seed)`` 实例，抽取
        序列完全相同但不再影响全局。另外旧实现在 ``n_nodes < max_edge_size``
        时会抛 ``ValueError: Sample larger than population``，本实现提前给出
        带参数名的 ``ValueError``。

        注意：结点是“按需出现”的——某个编号若从未被任何超边抽中，就不会出现在
        ``H.nodes`` 里，因此 ``H.shape[0]`` 可能小于 ``n_nodes``（``H.shape[1]``
        恒等于 ``n_edges``）。这一点与旧实现一致。

    示例:
        >>> H = generate_random_hypernet(n_nodes=5, n_edges=4, max_edge_size=3, seed=1)
        >>> H.shape[1]
        4
    """
    if max_edge_size < 2:
        msg = f"max_edge_size 必须 >= 2，实际为 {max_edge_size}"
        raise ValueError(msg)
    if n_nodes < max_edge_size:
        msg = f"n_nodes({n_nodes}) 必须 >= max_edge_size({max_edge_size})，否则无法无放回抽样"
        raise ValueError(msg)

    rng = random.Random(seed)
    hyperedge_list = [
        rng.sample(range(n_nodes), k=rng.randint(2, max_edge_size)) for _ in range(n_edges)
    ]
    return hypernetx().Hypergraph(hyperedge_list)
