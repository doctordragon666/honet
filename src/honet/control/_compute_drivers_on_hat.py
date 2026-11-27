"""按结点规模自动选择算法，在已构建好的 HAT 超图上计算驱动结点集合。"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from honet._deps import require
from honet.control._dilation import (
    detect_dilation_and_drivers,
    detect_dilation_with_given_drivers,
)
from honet.control._greedy_accessibility import greedy_accessibility
from honet.control._reachability import hypergraph_reachable

#: 结点数不超过该值时用暴力枚举（精确但指数级）
BRUTE_FORCE_LIMIT = 8
#: 结点数不超过该值时在匹配结果上再跑贪心补点
GREEDY_LIMIT = 30


def _brute_force(HG: Any, node_count: int) -> tuple[bool, set[Any], str]:
    """枚举最小可行驱动集合；返回 ``(dilation_exists, 驱动集合, 算法名)``。"""
    all_nodes = set(range(node_count))
    for size in range(1, node_count + 1):
        for subset in combinations(range(node_count), size):
            drivers = set(subset)
            if detect_dilation_with_given_drivers(HG, drivers):
                continue
            if (hypergraph_reachable(HG, drivers) & all_nodes) == all_nodes:
                return False, drivers, "brute_force"
    # 没有任何子集可行：说明超图完全不可控（存在无法消除的 dilation）
    return True, set(), "brute_force"


def _compute_drivers_on_hat(HG: Any, out_csv: str | None = "drivers.csv") -> dict[str, Any]:
    """在已构建好的有向 HAT 超图上计算驱动结点集合。

    算法按结点数自动选择：

    * ``nnodes <= 8``：暴力枚举最小可行驱动集合（精确）；
    * ``8 < nnodes <= 30``：最大匹配给出初始驱动集，再用贪心补齐可达性
      （``matching_plus_greedy``）；
    * ``nnodes > 30``：只用最大匹配结果（``matching_only``）。

    参数:
        HG: 已添加结点字段的有向 HAT 超图（建议先跑 :func:`honet.control.add_node_fields`）。
        out_csv: 结果 CSV 路径；``None`` 或空串表示不写文件。

    返回:
        ``{"algorithm": 算法名, "dilation_exists": 是否完全不可控,
        "drivers": 驱动结点下标升序列表}``。

    注意:
        旧实现在「匹配给出的初始驱动集仍存在 dilation」时只打印一句
        「超图完全不可控」，不改变返回结构；这里保留该行为（不打印），
        但 ``dilation_exists`` 标记的是「暴力分支找不到可行集合」这一含义，
        与匹配分支的判定口径不同，详见 :func:`honet.control.compute_drivers_from_hypernetx`。
    """
    pd = require("pandas")
    node_count = HG.nnodes

    if node_count <= BRUTE_FORCE_LIMIT:
        dilation_exists, drivers_final, algorithm = _brute_force(HG, node_count)
    else:
        _, _, driver_nodes, _ = detect_dilation_and_drivers(HG, node_count)
        if detect_dilation_with_given_drivers(HG, driver_nodes):
            # 匹配结果仍存在 dilation：该超图无法完全控制，但仍返回匹配得到的候选集
            dilation_exists = True
        else:
            dilation_exists = False
        initial_drivers = set(driver_nodes)
        if node_count <= GREEDY_LIMIT:
            algorithm = "matching_plus_greedy"
            drivers_final = greedy_accessibility(HG, initial_drivers)
        else:
            algorithm = "matching_only"
            drivers_final = initial_drivers

    drivers_sorted = sorted(drivers_final)
    if drivers_sorted:
        df = pd.DataFrame({"driver_node": drivers_sorted})
        df["algorithm"] = algorithm
        df["dilation_exists"] = dilation_exists
    else:
        df = pd.DataFrame(columns=["driver_node", "algorithm", "dilation_exists"])
    if out_csv:
        df.to_csv(out_csv, index=False)
    return {"algorithm": algorithm, "dilation_exists": dilation_exists, "drivers": drivers_sorted}
