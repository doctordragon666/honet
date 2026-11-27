"""验证「多指标共享一次拓扑索引」的实际收益。

场景：对同一个超图连续计算多个指标（真实分析流程的常见形态）。

两种模式对比：
* **无共享**：每个指标计算前清空缓存，等价于「各函数各自重建结构」；
* **共享**：同一对象连续调用全部指标，拓扑索引只建一次。

同时校验两种模式给出的数值完全一致。
"""

from __future__ import annotations

import time

import hypernetx as hnx
import numpy as np

from honet import _cache

METRIC_NAMES = (
    "weighted_degree",
    "clustering_coefficient",
    "excess_average_degree",
    "calc_entropy",
    "get_avg_degree_by_edge_size",
    "calc_clustering_coefficient_g",
)


def build_graph(n_nodes: int, n_edges: int, edge_size: int, seed: int = 0) -> hnx.Hypergraph:
    rng = np.random.default_rng(seed)
    edges = {}
    for i in range(n_edges):
        members = rng.choice(n_nodes, size=edge_size, replace=False)
        edges[f"e{i}"] = [f"v{j}" for j in members]
    return hnx.Hypergraph(edges)


def run_all(H) -> dict[str, object]:
    """依次计算全套指标。"""
    import honet

    return {name: getattr(honet.metrics, name)(H) for name in METRIC_NAMES}


def run_all_without_sharing(H) -> dict[str, object]:
    """每个指标计算前清空缓存，模拟「各函数各自重建」。"""
    import honet

    results = {}
    for name in METRIC_NAMES:
        _cache.clear()
        results[name] = getattr(honet.metrics, name)(H)
    return results


N_NODES, N_EDGES, SIZE = 300, 800, 4

# --- 无共享 ---
H1 = build_graph(N_NODES, N_EDGES, SIZE)
_cache.clear()
start = time.perf_counter()
no_share_results = run_all_without_sharing(H1)
no_share = time.perf_counter() - start

# --- 共享索引 ---
H2 = build_graph(N_NODES, N_EDGES, SIZE)
_cache.clear()
start = time.perf_counter()
shared_results = run_all(H2)
shared = time.perf_counter() - start

# --- 一致性 ---
same = all(no_share_results[name] == shared_results[name] for name in METRIC_NAMES)

print(f"超图规模：{N_NODES} 结点 / {N_EDGES} 超边 / 每条 {SIZE} 成员")
print(f"无共享（每次重建）：{no_share * 1000:8.2f} ms")
print(f"共享索引：          {shared * 1000:8.2f} ms")
print(f"加速：{no_share / shared:.1f}×")
print(f"全部指标结果一致：{same}")
print(f"缓存规模：{_cache.stats()}")
