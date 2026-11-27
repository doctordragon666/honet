"""基于最大匹配的 dilation 检测与候选驱动结点识别。"""

from __future__ import annotations

from typing import Any

from honet.control._dilation import detect_dilation_and_drivers as _impl


def detect_dilation_and_drivers(
    HG_local: Any, num_nodes: int
) -> tuple[bool, set[int], set[int], dict]:
    """通过二分图最大匹配检测 dilation 并识别驱动结点。

    做法：构造星形图后，把「超边 → 头结点」的边交给 Hopcroft-Karp 求最大匹配，
    匹配不上的结点必须由驱动结点补上，因此
    ``driver_nodes = 全部结点 - 匹配上的头结点``。

    参数:
        HG_local: HAT 有向超图。
        num_nodes: 系统结点总数，一般传 ``HG_local.nnodes``。

    返回:
        ``(dilation_exists, matched_heads, driver_nodes, matching)``：

        * ``dilation_exists``: ``len(matched_heads) < num_nodes`` 时为 ``True``；
        * ``matched_heads``: 匹配上的头结点下标集合；
        * ``driver_nodes``: 建议的驱动结点下标集合；
        * ``matching``: networkx 的原始匹配字典（便于调用方复用）。

    注意:
        ``matching`` 的键同时包含左右两侧结点名（如 ``{"e0": "n1", "n1": "e0"}``），
        这是 networkx 的返回格式，与旧实现一致。
    """
    return _impl(HG_local, num_nodes)
