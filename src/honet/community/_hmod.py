"""hypernetx 超图模块度算法的兼容封装（私有，不对外导出）。

旧库直接从 ``hypernetx.algorithms.hypergraph_modularity`` 导入 ``kumar`` /
``last_step`` / ``modularity`` / ``part2dict``。在 hypernetx 2.4.3 中该模块已
搬到 ``hypernetx.algorithms.clustering.hypergraph_modularity``，顶层路径不再
存在。这里把新路径集中到一处，公开模块只调用本文件的四个函数，避免版本
差异散落到各实现里。

本模块是实现细节（文件名下划线开头），不导出到 ``honet.community`` 的公开
命名空间。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require


def _algorithms() -> Any:
    """返回 hypernetx 的超图模块度算法模块。

    抛出:
        ImportError: 未安装 hypernetx（需要 ``honet[analysis]``）。
    """
    return require("hypernetx.algorithms.clustering.hypergraph_modularity")


def modularity(H: Any, partition: Any) -> float:
    """计算分区 ``partition`` 在超图 ``H`` 上的超图模块度。

    参数:
        H: hypernetx 超图对象。
        partition: 分区，``list[set]`` 或 ``list[list]``，必须覆盖 ``H`` 的
            全部结点，否则 hypernetx 内部查表会抛 ``KeyError``。

    返回:
        模块度，统一为 Python ``float``。
    """
    return float(_algorithms().modularity(H, partition))


def part2dict(partition: Any) -> dict[Any, int]:
    """把分区转换成 ``{结点: 社区下标}``。"""
    return _algorithms().part2dict(partition)


def kumar(H: Any) -> list[set[Any]]:
    """用 Kumar 算法划分超图，返回 ``list[set]``。"""
    return _algorithms().kumar(H)


def last_step(H: Any, partition: Any) -> list[set[Any]]:
    """以 ``partition`` 为初值做 Last-Step 局部优化，返回 ``list[set]``。"""
    return _algorithms().last_step(H, partition)
