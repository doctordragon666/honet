"""测量转换函数的耗时（验证「减少转换成本」的收益）。"""

from __future__ import annotations

import time

import hypernetx as hnx
import numpy as np

from honet import _cache


def build_graph(n_nodes: int, n_edges: int, edge_size: int, seed: int = 0) -> hnx.Hypergraph:
    rng = np.random.default_rng(seed)
    edges = {}
    for i in range(n_edges):
        members = rng.choice(n_nodes, size=edge_size, replace=False)
        edges[f"e{i}"] = [f"v{j}" for j in members]
    return hnx.Hypergraph(edges)


def bench(label: str, fn, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        _cache.clear()
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    print(f"{label:<46} {best * 1000:8.2f} ms")
    return best


H = build_graph(300, 800, 4)
print(f"超图规模：300 结点 / 800 超边\n")

import honet

bench("convert_hypernetx_to_nxgraph", lambda: honet.convert.convert_hypernetx_to_nxgraph(H))
bench(
    "convert_hypernetx_to_hypergraphx",
    lambda: honet.convert.convert_hypernetx_to_hypergraphx(H),
)
bench("convert_hypernetx_to_xgi", lambda: honet.convert.convert_hypernetx_to_xgi(H))
bench("get_hyperedge_weights", lambda: honet.convert.get_hyperedge_weights(H))
