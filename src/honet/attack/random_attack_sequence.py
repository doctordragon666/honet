"""随机攻击序列。

按目标把结点或超边随机打乱，得到「依次被移除」的顺序。随机序列用于给
度攻击、中心性攻击提供基线：如果随机攻击下的鲁棒性与定向攻击接近，
说明网络结构本身没有可被利用的不均衡性。

与旧实现（``hypertrade.honattack.random_attack_sequence``）的差异：
旧代码在 ``target='node'`` 分支里实际返回的是**按度升序**排好的结点列表
（注释与实现错位，随机逻辑只走到了超边分支），这里改为真正的随机打乱；
另外新增可选的 ``seed`` 参数，便于复现实验。
"""

from __future__ import annotations

import random
from typing import Any

#: 支持的攻击目标
_TARGETS = ("node", "edge")


def random_attack_sequence(
    hg: Any,
    target: str = "node",
    seed: int | None = None,
) -> list[Any]:
    """生成随机攻击序列。

    参数:
        hg: hypernetx 超图对象。
        target: 攻击目标，``"node"`` 打乱结点，``"edge"`` 打乱超边。
        seed: 随机种子；为 ``None`` 时使用全局随机状态。

    返回:
        打乱后的结点（或超边）列表，长度与 ``len(hg.nodes)``
        （或 ``len(hg.edges)``）相同。

    抛出:
        ValueError: ``target`` 不是 ``"node"`` 或 ``"edge"``。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> sorted(random_attack_sequence(H, seed=0)) == sorted(H.nodes)
        True
    """
    if target not in _TARGETS:
        msg = f"target 只能是 {_TARGETS} 之一，收到 {target!r}"
        raise ValueError(msg)

    items = list(hg.nodes) if target == "node" else list(hg.edges)
    rng = random.Random(seed) if seed is not None else random
    rng.shuffle(items)
    return items
