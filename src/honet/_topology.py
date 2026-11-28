"""一次性构建的超图拓扑索引。

多个指标函数都需要「结点 → 所属超边」「超边 → 成员」「结点 → 邻居」这几张
表。若每个函数各自遍历 hypernetx 对象，就会反复做同样的转换。本模块把
这些转换集中到一次构建，并按内容指纹缓存，使同一超图上的多次分析共享同一份
索引。

索引内容：

* ``members``：``{超边: 成员元组}``
* ``memberships``：``{结点: 所属超边元组}``
* ``degrees``：``{结点: 度}``，即成员数之和
* ``sizes``：``{超边: 成员数}``
* ``neighbors`` ：``{结点: 邻居结点 frozenset}`` （延迟构建，按需填充）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from honet import _cache


@dataclass
class Topology:
    """超图的邻接结构快照，只构建一次，供多个指标复用。"""

    members: dict[Any, tuple[Any, ...]]
    memberships: dict[Any, tuple[Any, ...]]
    degrees: dict[Any, int]
    sizes: dict[Any, int]
    nodes: tuple[Any, ...]
    edges: tuple[Any, ...]
    #: 邻居表按需填充（不是每个指标都需要）
    neighbors: dict[Any, frozenset[Any]] = field(default_factory=dict)

    def neighbors_of(self, node: Any) -> frozenset[Any]:
        """返回与 ``node`` 共享至少一条超边的其他结点（首次访问时计算并缓存）。"""
        cached = self.neighbors.get(node)
        if cached is not None:
            return cached
        found: set[Any] = set()
        for edge in self.memberships.get(node, ()):
            found.update(self.members[edge])
        found.discard(node)
        value = frozenset(found)
        self.neighbors[node] = value
        return value

    def pairs_sharing_edge(self) -> dict[frozenset[Any], int]:
        """返回 ``{结点对: 共享超边数}``，供聚类系数等「邻居对」类指标复用。"""
        pairs: dict[frozenset[Any], int] = {}
        for members in self.members.values():
            ordered = sorted(members, key=repr)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    key = frozenset((ordered[i], ordered[j]))
                    pairs[key] = pairs.get(key, 0) + 1
        return pairs


def _build(H: Any) -> Topology:
    """从 hypernetx 超图一次性抽取全部结构信息。

    性能要点：``H.edges[e]`` 每次访问都会重建成员列表（实测 500 条边约 4ms），
    而 ``H.edges.incidence_dict`` 是一次性返回的普通字典（同规模近 0ms）。
    因此这里优先走 ``incidence_dict``，只在不可用时退回逐边访问。
    """
    nodes = tuple(H.nodes)

    try:
        raw = H.edges.incidence_dict
        if not isinstance(raw, dict):  # 某些版本是方法
            raw = raw()
        members = {edge: tuple(mem) for edge, mem in raw.items()}
    except Exception:  # noqa: BLE001 - 退回逐边访问
        members = {edge: tuple(H.edges[edge]) for edge in H.edges}

    edges = tuple(members)

    memberships_lists: dict[Any, list[Any]] = {node: [] for node in nodes}
    sizes: dict[Any, int] = {}
    degrees: dict[Any, int] = {node: 0 for node in nodes}

    for edge, edge_members in members.items():
        sizes[edge] = len(edge_members)
        for node in edge_members:
            memberships_lists.setdefault(node, []).append(edge)
            degrees[node] = degrees.get(node, 0) + 1

    memberships = {node: tuple(edges_) for node, edges_ in memberships_lists.items()}
    return Topology(
        members=members,
        memberships=memberships,
        degrees=degrees,
        sizes=sizes,
        nodes=nodes,
        edges=edges,
    )


def topology(H: Any) -> Topology:
    """返回 ``H`` 的拓扑索引，命中缓存时不重建。

    缓存键带版本后缀：索引结构变更时递增，避免读到旧格式。
    """
    return _cache.cached(H, "topology:v1", lambda: _build(H))
