"""暴力搜索最小驱动结点集合。

枚举 ``1..n`` 的全部结点子集，按大小递增，返回第一组同时满足
「无 dilation」与「全系统可达」的集合，因此结果是最小基数意义下的精确解。
复杂度 ``O(2^n)``，只适合小超图（``nnodes <= 8`` 量级）。
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from honet.control._dilation import detect_dilation_with_given_drivers
from honet.control._reachability import hypergraph_reachable


def brute_force_min_driver_sets(
    HG: Any, verbose: bool = False
) -> tuple[int | None, list[set[int]]]:
    """暴力搜索最小驱动结点集合。

    参数:
        HG: HAT 有向超图。
        verbose: 是否打印搜索过程。

    返回:
        ``(最小规模, 该规模下全部可行的驱动结点集合列表)``；无可行集合时返回
        ``(None, [])`` （表示存在无法通过驱动结点消除的 dilation）。

    注意:
        返回的集合元素是结点下标（``0..nnodes-1``），不是原始结点名。
        同一规模下的多个可行集合会**全部**返回，这一点与
        :func:`honet.control.compute_and_export_drivers` 只取第一个不同。

    示例:
        >>> size, sets = brute_force_min_driver_sets(HG)  # doctest: +SKIP
        >>> size, sorted(sorted(s) for s in sets)  # doctest: +SKIP
        (2, [[0, 1]])
    """
    n = HG.nnodes
    all_sys_nodes = set(range(n))

    if verbose:
        print("开始暴力搜索……\n")

    for k in range(1, n + 1):
        if verbose:
            print(f"=== 尝试驱动集合规模 k = {k} ===")

        found_k: list[set[int]] = []
        for subset in combinations(range(n), k):
            drivers = set(subset)
            if detect_dilation_with_given_drivers(HG, drivers):
                continue
            if (hypergraph_reachable(HG, drivers) & all_sys_nodes) == all_sys_nodes:
                found_k.append(drivers)
                if verbose:
                    print("  可行驱动集合:", sorted(drivers))

        if found_k:
            if verbose:
                print(f"\n>>> 最小驱动规模 = {k}")
                for s in found_k:
                    print("    ", sorted(s))
            return k, found_k

    if verbose:
        print("未找到可行驱动集合：或存在 dilation，或可达性无法满足。")
    return None, []
