"""超图的星形图（二分图）表示。

把超图展开成「系统结点 ↔ 超边结点」的二分图，是 dilation 检测与匹配算法的
前置结构。有向超边按「尾结点 → 边结点 → 头结点」连边。
"""

from __future__ import annotations

from typing import Any

from honet.control._star_graph import build_star_graph


def star_graph(HG: Any) -> Any:
    """构建超图的星形图（二分图）表示。

    参数:
        HG: HAT 超图对象。

    返回:
        ``networkx.DiGraph``（有向超图）或 ``networkx.Graph``（无向超图）。
        结点名 ``n<i>`` 是系统结点（属性 ``bipartite=0``），``e<j>`` 是超边结点
        （属性 ``bipartite=1``，``j`` 为边表行号）。

    示例:
        >>> import pandas as pd
        >>> from HAT import Hypergraph
        >>> HG = Hypergraph(edges=pd.DataFrame({"Nodes": [[0, 1], [1, 2]]}))
        >>> sorted(star_graph(HG).nodes)
        ['e0', 'e1', 'n0', 'n1', 'n2']
    """
    return build_star_graph(HG)
