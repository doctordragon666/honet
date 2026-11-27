"""性能基线 / 优化效果对比脚本。

在固定随机超图上测量各指标函数的耗时与调用次数，用于验证优化收益。
输出为 markdown 表格，可直接贴进报告。

用法：
    uv run python docs/bench_metrics.py            # 默认规模
    uv run python docs/bench_metrics.py --big      # 更大规模
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import hypernetx as hnx
import numpy as np


def build_graph(n_nodes: int, n_edges: int, edge_size: int, seed: int = 0) -> hnx.Hypergraph:
    """构造一个随机超图：每条超边随机取 edge_size 个不同结点。"""
    rng = np.random.default_rng(seed)
    edges = {}
    for i in range(n_edges):
        members = rng.choice(n_nodes, size=edge_size, replace=False)
        edges[f"e{i}"] = [f"v{j}" for j in members]
    return hnx.Hypergraph(edges)


def timeit(fn: Callable[[], object], repeat: int = 3) -> float:
    """返回最佳耗时（秒），降低噪声。"""
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--big", action="store_true")
    args = parser.parse_args()

    if args.big:
        n_nodes, n_edges, size = 600, 1500, 5
    else:
        n_nodes, n_edges, size = 200, 500, 4

    import honet
    from honet import _cache

    cases: list[tuple[str, Callable[[], object]]] = [
        ("weighted_degree", lambda H: honet.metrics.weighted_degree(H)),
        ("clustering_coefficient", lambda H: honet.metrics.clustering_coefficient(H)),
        ("excess_average_degree", lambda H: honet.metrics.excess_average_degree(H)),
        ("betweenness_centrality", lambda H: honet.metrics.betweenness_centrality(H)),
        ("calc_clustering_coefficient_g", lambda H: honet.metrics.calc_clustering_coefficient_g(H)),
        ("calc_entropy", lambda H: honet.metrics.calc_entropy(H)),
        ("calc_natural_connectivity", lambda H: honet.metrics.calc_natural_connectivity(H)),
    ]

    print(f"超图规模：{n_nodes} 结点 / {n_edges} 超边 / 每条 {size} 成员\n")
    print("| 函数 | 冷启动 (ms) | 命中缓存 (ms) | 加速 |")
    print("| --- | ---: | ---: | ---: |")

    for name, fn in cases:
        # 冷启动：清空全部缓存后单次调用
        H_cold = build_graph(n_nodes, n_edges, size)
        _cache.clear()
        start = time.perf_counter()
        fn(H_cold)
        cold = time.perf_counter() - start

        # 命中缓存：同一对象重复调用
        start = time.perf_counter()
        fn(H_cold)
        warm = time.perf_counter() - start

        speedup = cold / warm if warm > 0 else float("inf")
        print(f"| `{name}` | {cold * 1000:.2f} | {warm * 1000:.2f} | {speedup:.1f}× |")


if __name__ == "__main__":
    main()
