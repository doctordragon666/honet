"""缓存与拓扑索引的测试。

覆盖三件事：
1. 缓存命中确实避免重算（用调用计数验证）；
2. 超图结构变化后缓存必须失效（不能返回过期结果）；
3. 指纹只取决于拓扑，与对象身份、插入顺序无关。
"""

from __future__ import annotations

import gc

import hypernetx as hnx
import pytest

from honet import _cache
from honet._topology import topology


@pytest.fixture(autouse=True)
def clean_cache():
    """每个用例前后清空缓存，避免相互干扰。"""
    _cache.clear()
    yield
    _cache.clear()


def make_graph() -> hnx.Hypergraph:
    return hnx.Hypergraph(
        {
            "e1": ["a", "b", "c"],
            "e2": ["b", "c"],
            "e3": ["c", "d"],
        }
    )


class TestFingerprint:
    """指纹只取决于拓扑。"""

    def test_same_structure_same_fingerprint(self):
        """同样结构、不同对象，指纹相同。"""
        a = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        b = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        assert _cache.fingerprint(a) == _cache.fingerprint(b)

    def test_insertion_order_irrelevant(self):
        """成员与超边的插入顺序不影响指纹。"""
        a = hnx.Hypergraph({"e1": ["a", "b", "c"]})
        b = hnx.Hypergraph({"e1": ["c", "a", "b"]})
        assert _cache.fingerprint(a) == _cache.fingerprint(b)

    def test_different_structure_differs(self):
        """结构不同则指纹不同。"""
        a = hnx.Hypergraph({"e1": ["a", "b"]})
        b = hnx.Hypergraph({"e1": ["a", "c"]})
        assert _cache.fingerprint(a) != _cache.fingerprint(b)

    def test_fingerprint_is_sha256_hex(self):
        assert len(_cache.fingerprint(make_graph())) == 64


class TestStructure:
    """第一级结构特征。"""

    def test_structure_triple(self):
        H = make_graph()
        assert _cache.structure(H) == (3, 4, 7)

    def test_structure_changes_after_remove(self):
        H = make_graph()
        before = _cache.structure(H)
        H.remove_edges("e3")
        assert _cache.structure(H) != before


class TestCacheHit:
    """命中缓存时不重复计算。"""

    def test_cached_calls_compute_once(self):
        """同一对象重复取值，compute 只执行一次。"""
        H = make_graph()
        calls = []

        def compute():
            calls.append(1)
            return "value"

        assert _cache.cached(H, "k", compute) == "value"
        assert _cache.cached(H, "k", compute) == "value"
        assert len(calls) == 1

    def test_different_keys_are_independent(self):
        H = make_graph()
        calls = {"a": 0, "b": 0}

        def make(name):
            def compute():
                calls[name] += 1
                return name

            return compute

        _cache.cached(H, "a", make("a"))
        _cache.cached(H, "b", make("b"))
        assert calls == {"a": 1, "b": 1}

    def test_structure_change_invalidates(self):
        """超图被修改后必须重算，不能返回旧值。"""
        H = make_graph()
        calls = []

        def compute():
            calls.append(1)
            return len(H.edges)

        assert _cache.cached(H, "k", compute) == 3
        H.remove_edges("e3")
        assert _cache.cached(H, "k", compute) == 2
        assert len(calls) == 2

    def test_lru_eviction_keeps_limit(self):
        """条目数不超过上限。"""
        H = make_graph()
        for i in range(_cache._MAX_ENTRIES + 10):
            _cache.cached(H, f"k{i}", lambda i=i: i)
        assert _cache.stats()["entries"] <= _cache._MAX_ENTRIES


class TestTopology:
    """拓扑索引的内容与复用。"""

    def test_reused_for_same_object(self):
        """同一对象两次取索引，返回同一个实例。"""
        H = make_graph()
        assert topology(H) is topology(H)

    def test_rebuilt_after_change(self):
        """结构变化后重建索引。"""
        H = make_graph()
        first = topology(H)
        H.remove_edges("e3")
        second = topology(H)
        assert first is not second
        assert len(second.edges) == 2

    def test_contents(self):
        H = make_graph()
        top = topology(H)
        assert top.members["e1"] == ("a", "b", "c")
        assert top.degrees == {"a": 1, "b": 2, "c": 3, "d": 1}
        assert top.sizes["e2"] == 2

    def test_neighbors(self):
        H = make_graph()
        top = topology(H)
        assert top.neighbors_of("a") == frozenset({"b", "c"})
        assert top.neighbors_of("d") == frozenset({"c"})

    def test_neighbors_cached_after_first_call(self):
        """邻居表按需填充，第二次访问直接命中。"""
        H = make_graph()
        top = topology(H)
        assert "a" not in top.neighbors
        top.neighbors_of("a")
        assert "a" in top.neighbors

    def test_pairs_sharing_edge(self):
        H = make_graph()
        pairs = topology(H).pairs_sharing_edge()
        assert pairs[frozenset({"b", "c"})] == 2


class TestMemory:
    """缓存不会因为对象回收而泄漏。"""

    def test_entries_dropped_after_collect(self):
        H = make_graph()
        _cache.cached(H, "k", lambda: 1)
        assert _cache.stats()["objects_with_values"] == 1

        del H
        gc.collect()
        assert _cache.stats()["objects_with_values"] == 0


class TestMetricsUseCache:
    """公开指标函数确实接入了缓存。"""

    @pytest.mark.parametrize(
        "name",
        [
            "weighted_degree",
            "clustering_coefficient",
            "excess_average_degree",
            "betweenness_centrality",
            "calc_entropy",
            "calc_natural_connectivity",
            "calc_clustering_coefficient_g",
        ],
    )
    def test_metric_result_is_stable_and_cached(self, name):
        """重复调用返回一致结果，且缓存中留有条目。"""
        import honet

        H = make_graph()
        fn = getattr(honet.metrics, name)
        first = fn(H)
        second = fn(H)
        assert first == second
        assert _cache.stats()["entries"] >= 1

    def test_metric_reflects_structure_change(self):
        """指标结果随结构变化而更新，不会读到旧缓存。"""
        import honet

        H = make_graph()
        before = honet.metrics.clustering_coefficient(H)
        H.remove_edges("e1")
        after = honet.metrics.clustering_coefficient(H)
        assert before != after
