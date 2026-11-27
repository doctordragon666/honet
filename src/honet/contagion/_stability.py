"""传播稳定性判定与相图单点分类（私有，不对外导出）。

公开的 :func:`honet.contagion.is_stable`、:func:`honet.contagion.classify_point`
与 :func:`honet.contagion.sis_phase` 共用本模块，避免公开模块之间互相 import
（原因见 :mod:`honet.contagion._tau` 的模块说明）。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require
from honet.contagion._degree_stats import avg_degree_by_edge_size
from honet.contagion._tau import build_tau


def stability_check(H: Any, tau: dict[int, float], gamma: float) -> bool:
    """判断 SIS 传播是否稳定（感染衰减而非暴发）。

    参数:
        H: hypernetx 超图对象。
        tau: ``{超边大小: 感染率}``。
        gamma: 恢复率。

    返回:
        ``True`` 表示稳定；没有结点或图中没有大小为 2 的超边时返回 ``True``。
    """
    hnx = require("hypernetx")
    stats = hnx.dist_stats(H)
    k_list = stats["node degree list"]
    n = len(k_list)
    if n == 0:
        return True

    k_mean = sum(k_list) / n
    per_size = avg_degree_by_edge_size(H)
    # 用「大小为 2 的超边下的平均度」作为 <k>_2
    k_mean_sq = per_size.get(2, 0.0)
    if k_mean_sq == 0:
        return True

    rhs = gamma * k_mean / k_mean_sq

    lhs = 0.0
    for m, count in per_size.items():
        if m < 2:
            continue
        beta_m = tau.get(m, 0.0)
        if beta_m == 0:
            continue
        lhs += (m - 1) * beta_m * count / k_mean

    return lhs < rhs


def classify(
    H: Any, beta2: float, beta_higher: float, gamma: float, kind: str, function: Any
) -> bool:
    """生成 ``tau`` 后判定单个相图点是否稳定。

    参数:
        H: hypernetx 超图对象。
        beta2: 大小为 2 的超边的感染率。
        beta_higher: 其余超边大小的感染率参数。
        gamma: 恢复率。
        kind: ``tau`` 的生成方式。
        function: 自定义 ``tau`` 生成函数；``None`` 时用 :func:`build_tau`。

    返回:
        ``True`` 表示该点稳定。
    """
    betas = [float(beta2), float(beta_higher)]
    if function is not None:
        tau = function(H, beta=betas, type=kind)
    else:
        tau = build_tau(H, betas, kind)
    return stability_check(H, tau, gamma)
