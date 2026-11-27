"""加权超图的可约性（含代表层集合）。

可约性回答「一个多层超图能否只用少数几个代表性层（超边规模）无损描述」：
把超边按规模分层，若层间信息高度冗余，则可约性接近 1。

本实现是对 HyperGraphX 官方 ``reducibility`` 的**加权修正版**：在官方无权
熵的基础上，额外叠加每层的「权重异质性成本」——层内权重越不均，该层越
难以被其它层代表。

参考文献：Kirkley, A., Felippe, H., & Battiston, F. PRL 135, 247401 (2025).
"""

from __future__ import annotations

import math
from typing import Any

from honet._deps import require

#: 允许的优化方式（与 HyperGraphX 官方一致）
_OPTIMIZATIONS = frozenset({"exact", "greedy"})
#: 允许的熵估计口径（与 HyperGraphX 官方一致）
_ENTROPY_METHODS = frozenset({"count", "project"})


def _layer_weight_costs(hg: Any, layers: dict[int, set[Any]]) -> dict[int, float]:
    """计算每层的权重异质性成本 ``W_l``。

    参数:
        hg: ``hypergraphx.Hypergraph``，需支持 ``get_weight(edge)``。
        layers: ``{超边规模: 该规模的边集合}``，由 HyperGraphX 的私有函数
            ``_get_layers`` 给出。

    返回:
        ``{超边规模: W_l}``，其中 ``W_l = E_l · [ln E_l − H(p)] ≥ 0``，
        层内权重全相等时取 0；自然对数单位与官方熵估计保持一致。
    """
    costs: dict[int, float] = {}
    for size, edges in layers.items():
        keys = list(edges)
        count = len(keys)
        if count <= 1:
            # 0/1 条边无从谈异质性
            costs[size] = 0.0
            continue

        weights = [float(hg.get_weight(edge)) for edge in keys]
        total = sum(weights)
        if total <= 0.0:
            costs[size] = 0.0
            continue

        p = [weight / total for weight in weights]
        entropy = -sum(pi * math.log(pi) for pi in p if pi > 0.0)
        costs[size] = count * (math.log(count) - entropy)
    return costs


def sub_reducibility(
    hg: Any,
    partition: Any = None,
    optimization: str = "exact",
    entropy_method: str = "count",
) -> tuple[float, tuple[int, ...]]:
    """计算（加权）超图的可约性及代表层集合。

    参数:
        hg: ``hypergraphx.Hypergraph``。加权时权重自动参与；无权时权重全为 1，
            退化为官方结果。
        partition: 结点分区（``list`` / ``dict`` / ``None``），用于多尺度可约性。
        optimization: ``"exact"``（枚举全部代表层组合）或 ``"greedy"``（贪心）。
        entropy_method: ``"count"`` 或 ``"project"``，对应官方两种熵估计。

    返回:
        ``(reducibility, representative_layers)``：可约性取值在 ``[0, 1]``，
        代表层是升序排列的超边规模元组。

    抛出:
        ImportError: 未安装 hypergraphx。
        hypergraphx.exceptions.InvalidParameterError: 参数非法，或输入不是
            无向 ``Hypergraph``（由 HyperGraphX 抛出）。

    说明:
        **依赖 hypergraphx 的内部接口**：``_get_entropies_count`` /
        ``_get_entropies_project`` / ``_get_layers`` / ``_powerset`` 都是下划线
        开头的私有函数，上游版本变化时可能改名或改语义而失效。这是为了在
        官方无权熵之上叠加权重成本所付出的代价；升级 hypergraphx 后请重跑
        本子包测试。

    示例:
        >>> import hypergraphx as hgx
        >>> G = hgx.Hypergraph([("a", "b"), ("a", "b", "c")], weighted=True, weights=[1.0, 1.0])
        >>> value, reps = sub_reducibility(G, optimization="greedy")
        >>> 0.0 <= value <= 1.0
        True
    """
    np = require("numpy")
    require("hypergraphx")
    reducibility_module = require("hypergraphx.measures.reducibility")
    invalid_parameter = require("hypergraphx.exceptions", attr="InvalidParameterError")

    if entropy_method not in _ENTROPY_METHODS:
        raise invalid_parameter('entropy_method must be either "count" or "project".')
    if optimization not in _OPTIMIZATIONS:
        raise invalid_parameter('optimization must be either "exact" or "greedy".')

    # 1) 复用官方无权熵：Q 为层自身成本，M[k][l] 为高层 k 对低层 l 的条件成本
    if entropy_method == "project":
        M, Q = reducibility_module._get_entropies_project(hg, partition)
    else:
        M, Q = reducibility_module._get_entropies_count(hg, partition)

    # 2) 加权修正：把各层权重异质性成本 W_l 叠加到 Q 与 M 上
    weight_costs = _layer_weight_costs(hg, reducibility_module._get_layers(hg))
    for layer in Q:
        cost = weight_costs.get(layer, 0.0)
        Q[layer] = Q[layer] + cost
        for higher in M:
            # M[k][l] 只对 k > l 存在
            if higher > layer:
                M[higher][layer] += cost

    max_layer = max(Q.keys())

    # 3) 最小化总信息成本，寻找代表层集合（与官方逻辑一致）
    if optimization == "exact":
        costs_by_reps: dict[tuple[int, ...], float] = {}
        for reps in reducibility_module._powerset(list(Q.keys())):
            chosen = set(reps)
            if max_layer not in chosen:
                # 最高层必须作为代表层（它没有更高层可以代表它）
                continue
            total = sum(Q[r] for r in chosen)
            for layer in set(Q.keys()) - chosen:
                higher = [r for r in chosen if r > layer]
                best = higher[int(np.argmin([M[r][layer] for r in higher]))]
                total += M[best][layer]
            costs_by_reps[tuple(sorted(chosen))] = total
        best_reps = min(costs_by_reps, key=costs_by_reps.get)
        best_cost = costs_by_reps[best_reps]
    else:
        rep_history: list[list[int]] = []
        cost_history: list[float] = []
        chosen = {max_layer}
        remaining = set(Q.keys()) - chosen
        rep_history.append(sorted(chosen))
        cost_history.append(Q[max_layer] + sum(M[max_layer][layer] for layer in remaining))
        while remaining:
            candidates: list[int] = []
            scores: list[float] = []
            for candidate in remaining:
                trial = chosen | {candidate}
                still_out = remaining - {candidate}
                total = sum(Q[r] for r in trial)
                for layer in still_out:
                    higher = [r for r in trial if r > layer]
                    best = higher[int(np.argmin([M[r][layer] for r in higher]))]
                    total += M[best][layer]
                candidates.append(candidate)
                scores.append(total)
            winner = int(np.argmin(scores))
            chosen.add(candidates[winner])
            remaining.remove(candidates[winner])
            rep_history.append(sorted(chosen))
            cost_history.append(scores[winner])
        best_index = int(np.argmin(cost_history))
        best_reps = tuple(rep_history[best_index])
        best_cost = cost_history[best_index]

    # 4) 归一化到 [0, 1]：完全不可约（每层都必需）时为 0，单层即可代表时为 1
    baseline = sum(Q[layer] for layer in Q)
    if baseline == Q[max_layer]:
        return 1.0, tuple(best_reps)
    return float((baseline - best_cost) / (baseline - Q[max_layer])), tuple(best_reps)
