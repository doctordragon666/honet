"""批量计算多个高阶网络的整体指标，输出成表格。

把「一堆超图（通常按年份排列）」压成一张 ``pandas.DataFrame``：行是时间键，
列是指标名，便于直接画趋势图（见 :func:`honet.plot.draw_hnxs_global`）。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require
from honet.metrics._reuse import package_function


def hnxs_global(hnxs: dict[Any, Any]) -> Any:
    """计算多个高阶网络的整体性质。

    参数:
        hnxs: ``{时间键: 超图对象}``，超图可以是 hypernetx 或 hypergraphx 对象。

    返回:
        ``pandas.DataFrame``，索引为 ``hnxs`` 的键（如年份），列为指标名。
        当前只统计一项指标「高阶网络可约性」（加权口径），与旧实现一致——
        旧代码里其余指标都已被注释掉。

    说明:
        **与旧实现的差异**：旧代码用 ``isinstance(H_hnx, hnx.Hypergraph)`` 判断
        输入，非 hypernetx 对象会被原样转交下游；这里改由
        :func:`honet.metrics.calc_reducibility` 统一处理（既接受 hypernetx 也
        接受 hypergraphx 超图）。单元格取 ``no_layer=True`` 的标量口径，
        与旧实现一致。

    示例:
        >>> import hypernetx as hnx
        >>> H1 = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["a", "b", "c"]})
        >>> H2 = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> table = hnxs_global({2020: H1, 2021: H2})
        >>> list(table.columns)
        ['高阶网络可约性']
    """
    pd = require("pandas")
    reducibility = package_function("honet.metrics", "calc_reducibility")

    rows = {
        key: {"高阶网络可约性": reducibility(graph, weighted=True)} for key, graph in hnxs.items()
    }
    return pd.DataFrame(rows).T
