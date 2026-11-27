"""检查一份历史社区划分能否用于当前网络。"""

from __future__ import annotations

from typing import Any

from honet.community._methods import check_partition as _impl


def check_partition(hypergraph: Any, prev_partition: Any) -> list[Any]:
    """找出当前网络中「没有被历史划分覆盖」的结点。

    用途：把上一年的划分结果迁移到今年的网络时，先确认划分覆盖了今年的全部
    结点；返回的列表非空，说明这些结点是新增的，需要重新分配社区。

    参数:
        hypergraph: 当前的高阶网络对象（只需提供 ``nodes``）。
        prev_partition: 历史划分，格式为 ``{结点: 社区}``。

    返回:
        当前网络中不在 ``prev_partition`` 里的结点列表，顺序与
        ``hypergraph.nodes`` 一致。``prev_partition`` 不是字典时返回空列表
        （旧实现会打印一条警告，这里改为不产生输出）。

    注意:
        旧实现只保留「在当前网络中」的键，再用它反查缺失结点，因此对历史划分
        里多出来的旧结点是容忍的。这里保持该语义。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"]})
        >>> check_partition(H, {"a": 0, "b": 0})
        ['c']
    """
    return _impl(hypergraph, prev_partition)
