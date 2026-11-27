"""比较多张超图在同一划分方法下的结果。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from honet.community._methods import compare_graphs as _impl
from honet.community.method import Method


def compare_graphs(
    H_lst: Iterable[Any],
    method: Method = Method.KUMAR,
) -> dict[Any, dict[str, Any] | float]:
    """对多张超图逐一施加同一种划分方法。

    参数:
        H_lst: 超图列表，也可以是 ``{标签: 超图}`` 形式的字典。
        method: 划分方法，默认 ``Method.KUMAR``。

    返回:
        ``{标签: extract_community_partition 的返回值}``。标签的取法：

        * 传入字典时用字典的键；
        * 传入序列时用元素下标（``0``、``1``……）。

    注意:
        旧实现直接把超图对象当作字典的键（``result_dict[H] = ...``）。在
        hypernetx 2.4.3 中 ``Hypergraph`` 不可哈希，这样做会抛
        ``TypeError: unhashable type: 'Hypergraph'``，所以这里改为用下标或调用方
        提供的键作标签——这是**与旧库的语义差异**。需要按名字索引时请传字典。

    示例:
        >>> import hypernetx as hnx
        >>> A = hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["c", "d"]})
        >>> B = hnx.Hypergraph({"e1": ["x", "y"], "e2": ["y", "z"]})
        >>> result = compare_graphs({"A": A, "B": B})
        >>> sorted(result)
        ['A', 'B']
    """
    return _impl(H_lst, method=method)
