"""按指定方法计算社区划分并返回模块度。"""

from __future__ import annotations

from typing import Any

from honet.community._methods import extract_community_partition as _impl
from honet.community.method import Method


def extract_community_partition(
    hnx_g: Any,
    method: Method = Method.KUMAR,
    only_qh: bool = False,
) -> dict[str, Any] | float:
    """用指定方法划分超图，返回模块度与分区。

    参数:
        hnx_g: hypernetx 超图对象。
        method: 划分方法，取值见 :class:`honet.community.Method`。默认
            ``Method.KUMAR``。
        only_qh: 为 ``True`` 时只返回模块度。

    返回:
        ``only_qh=True`` 时返回模块度 ``float``；否则返回字典::

            {"qh": 模块度, "communities": 分区, "nodemap": {结点: 社区下标}}

    抛出:
        ValueError: ``method`` 不是 :class:`Method` 的成员。
        RuntimeError: 所选方法没有产出任何划分（例如 NMF 在可用社区数范围内
            找不到解，或递归/层次方法遇到退化输入）。

    注意:
        旧实现里 NMF 与递归方法可能返回 ``None``，随后 ``modularity(None)``
        会抛出一个含义不明的异常；这里改为显式报错并说明是哪个方法没有结果。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["c", "d"]})
        >>> result = extract_community_partition(H)
        >>> set(result) == {"qh", "communities", "nodemap"}
        True
    """
    return _impl(hnx_g, method=method, only_qh=only_qh)
