"""超边膨胀（dilation）检测。

判据是「二分图最大匹配是否覆盖全部系统结点」：把每条超边的 head 侧与系统结点
做二分匹配，若匹配不到全部结点，说明存在 dilation（多条超边争夺同一批头结点，
或某些结点从不作为任何超边的头），此时仅靠驱动结点无法完全控制系统。

驱动结点的作用等价于额外加一个「只指向该结点的源超边」，因此
:func:`detect_dilation_with_given_drivers` 会在星形图上补 ``u<i> -> n<i>``
这样的边再求匹配。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require
from honet.control._star_graph import build_star_graph


def _bipartite_matching(G_star: Any, left_prefixes: tuple[str, ...]) -> dict[Any, Any]:
    """在星形图上求「左部（超边/驱动结点）→ 右部（系统结点）」的最大匹配。

    只保留左部以 ``left_prefixes`` 开头、右部以 ``n`` 开头的边，这样有向图里
    「尾结点 → 边结点」那一半不会被误当作可匹配关系（与旧实现一致）。

    参数:
        G_star: :func:`honet.control.star_graph` 的返回值。
        left_prefixes: 允许出现在左部的结点名前缀，如 ``("e",)`` 或 ``("e", "u")``。

    返回:
        networkx 的匹配字典，键包含左右两侧的结点名。
    """
    nx = require("networkx")
    from networkx.algorithms import bipartite

    G_bip = nx.Graph()
    left_side = set()
    for u, v in G_star.edges():
        if (
            isinstance(u, str)
            and isinstance(v, str)
            and u.startswith(left_prefixes)
            and v.startswith("n")
        ):
            G_bip.add_edge(u, v)
            left_side.add(u)
    return bipartite.hopcroft_karp_matching(G_bip, left_side)


def _matched_node_ids(matching: dict[Any, Any]) -> set[int]:
    """从匹配字典里取出「已匹配上的系统结点」下标集合。"""
    return {int(v[1:]) for v in matching if isinstance(v, str) and v.startswith("n")}


def detect_dilation_and_drivers(
    HG_local: Any, num_nodes: int
) -> tuple[bool, set[int], set[int], dict]:
    """用最大匹配检测 dilation，并据此给出候选驱动结点。

    参数:
        HG_local: HAT 有向超图。
        num_nodes: 系统结点总数（一般取 ``HG_local.nnodes``）。

    返回:
        ``(dilation_exists, matched_heads, driver_nodes, matching)``：
        是否存在 dilation、匹配上的头结点下标集合、驱动结点集合
        （``全部结点 - matched_heads``）、以及原始匹配字典。
    """
    G_star = build_star_graph(HG_local)
    matching = _bipartite_matching(G_star, ("e",))
    matched_heads = _matched_node_ids(matching)
    all_nodes = set(range(HG_local.nnodes))
    driver_nodes = all_nodes - matched_heads
    dilation_exists = len(matched_heads) < num_nodes
    return dilation_exists, matched_heads, driver_nodes, matching


def detect_dilation_with_given_drivers(HG_local: Any, drivers: Any) -> bool:
    """在给定驱动结点集合下检测 dilation 是否仍然存在。

    参数:
        HG_local: HAT 有向超图。
        drivers: 候选驱动结点下标集合（可迭代）。

    返回:
        ``True`` 表示仍存在 dilation（该驱动集合无法完全控制系统）。
    """
    G_star = build_star_graph(HG_local)
    for driver in drivers:
        u_node = f"u{driver}"
        if not G_star.has_node(u_node):
            G_star.add_node(u_node)
        G_star.add_edge(u_node, f"n{driver}")

    matching = _bipartite_matching(G_star, ("e", "u"))
    matched_heads = _matched_node_ids(matching)
    return len(matched_heads) < HG_local.nnodes
