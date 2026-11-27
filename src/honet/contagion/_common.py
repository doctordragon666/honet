"""传播动力学子包的内部共享工具（私有，不对外导出）。

这里只放三件与模型无关的小事：统一播种、裁剪用户给定的结点集合、
按旧实现口径挑选初始感染者。
"""

from __future__ import annotations

import random
from typing import Any

from honet._deps import require


def seed_everything(seed: int | None) -> None:
    """同时给 ``random`` 与 ``numpy.random`` 播种，保证随机过程可复现。

    参数:
        seed: 随机种子；为 ``None`` 时不做任何事（沿用调用方已有的随机状态）。
    """
    if seed is None:
        return
    random.seed(int(seed))
    require("numpy").random.seed(int(seed))


def normalize_nodes(H: Any, nodes: Any) -> list[Any]:
    """把用户给定的结点集合裁剪成「超图中真实存在」的结点列表。

    与旧实现 ``set(H.nodes).intersection(nodes)`` 等价，但保持 ``H.nodes`` 的
    迭代顺序，使结果可复现。

    参数:
        H: hypernetx 超图对象。
        nodes: 候选结点（可为 ``None``）。

    返回:
        属于 ``H`` 的结点列表；``nodes`` 为 ``None`` 时返回空列表。
    """
    if nodes is None:
        return []
    wanted = set(nodes)
    return [node for node in H.nodes if node in wanted]


def choose_initial_infecteds(
    H: Any,
    initial_infecteds: Any,
    rho: float | None,
) -> list[Any]:
    """确定初始感染者列表（``initial_infecteds`` 优先，其次按比例 ``rho`` 抽样）。

    参数:
        H: hypernetx 超图对象。
        initial_infecteds: 显式给定的初始感染者；``None`` 表示随机抽样。
        rho: 初始感染比例，仅当 ``initial_infecteds`` 为 ``None`` 时生效；
            两者都为 ``None`` 时取 1 个结点（与旧实现一致）。

    返回:
        初始感染者列表。

    抛出:
        ValueError: ``initial_infecteds`` 与 ``rho`` 同时给出（旧实现抛的是
            裸 ``Exception``，这里收窄为 ``ValueError``，便于调用方捕获）。

    备注:
        ``rho > 0`` 时至少返回 1 个结点。旧实现直接取
        ``int(round(len(H.nodes) * rho))``，在 4 个结点的小图上会算出 0 个
        初始感染者，进而让 hypernetx 的 Gillespie 分支
        （``random.expovariate(0)``）抛 ``ZeroDivisionError``；这里补上最小
        为 1 的保护，使小图也能正常模拟。``rho`` 为 0 或 ``None`` 时行为不变。
    """
    if rho is not None and initial_infecteds is not None:
        msg = "initial_infecteds 与 rho 不能同时指定"
        raise ValueError(msg)
    if initial_infecteds is not None:
        return normalize_nodes(H, initial_infecteds)

    candidates = list(H.nodes)
    if not candidates:
        return []
    count = 1 if rho is None else int(round(len(candidates) * rho))
    if rho:
        # rho 为正却算出 0 个感染者时补到 1 个，避免空初始状态
        count = max(1, count)
    count = max(0, min(count, len(candidates)))
    return random.sample(candidates, count)
