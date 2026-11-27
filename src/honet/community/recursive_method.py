"""递归二分：对模块度仍可提升的社区继续拆分。"""

from __future__ import annotations

from typing import Any

from honet.community._methods import recursive_method as _impl


def recursive_method(hnx_g: Any, max_depth: int = 10) -> list[set[Any]]:
    """以 Kumar 划分为起点，自顶向下递归拆分社区直到模块度不再提升。

    每一步取出当前划分里的每个社区，用「完全落在该社区内的超边」构造诱导子
    超图，在其上跑一次 Kumar；若把该社区替换成子划分后整体模块度**严格上升**，
    就接受这次拆分并继续处理新产生的子社区，否则停止。

    参数:
        hnx_g: hypernetx 超图对象。
        max_depth: 最大递归层数，防止病态输入下无限递归。

    返回:
        分区，``list[set]``。结果至少包含 Kumar 给出的初始划分，模块度不会低于
        初始划分。

    注意:
        旧实现的 ``Recursive_Method`` 函数体只有 ``pass``，返回 ``None``，实际
        未实现。这里是按旧实现 docstring 描述的思路重新实现的版本，因此**语义
        与旧库不同**（旧库调用它只会得到 ``None``）。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["c", "d"]})
        >>> sorted(n for c in recursive_method(H) for n in c) == sorted(H.nodes)
        True
    """
    return _impl(hnx_g, max_depth=max_depth)
