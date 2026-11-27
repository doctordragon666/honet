"""超边层次聚类后投影回结点，得到社区划分。"""

from __future__ import annotations

from typing import Any

from honet.community._methods import hierarchy_method as _impl


def hierarchy_method(hnx_g: Any, cut_height: float = 0.5) -> list[list[Any]]:
    """先对超边做层次聚类，再用多数票把超边簇投影成结点社区。

    步骤：

    1. 转成 hypergraphx 超图并取最大连通分量（孤立结点没有超边，无法参与
       超边聚类，稍后单独成簇）；
    2. 以 Jaccard 距离为边相似度，构建「超边为结点」的加权图；
    3. 对距离矩阵做 average-linkage 层次聚类得到树状图；
    4. 用 :func:`hypergraphx.communities.get_num_hyperlink_communties` 在
       ``cut_height`` 处切树，得到超边簇数，再 ``fcluster`` 取超边标签；
    5. 每个结点归属到「它所在超边中最常见的那个超边簇」。

    参数:
        hnx_g: hypernetx 超图对象。
        cut_height: 树状图的切割距离阈值（不是社区数量）。取值越大社区越少。
            默认 0.5。

    返回:
        分区，``list[list]``，覆盖 ``hnx_g`` 的全部结点。未被任何超边覆盖的
        结点各自单独成簇。

    注意:
        旧实现有两处缺陷，这里已修正，因此**语义与旧库不同**：

        * 它把「超边数量」当作 ``cut_height`` 传给
          ``get_num_hyperlink_communties``，而该参数实际是距离阈值，导致切出
          的社区数恒为 1；
        * 它用 ``fcluster`` 得到的**超边**标签去索引 ``list(hnx_g.nodes)``，
          把超边下标当成结点下标，当结点数与超边数不同时会张冠李戴或
          ``IndexError``。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["c", "d"]})
        >>> partition = hierarchy_method(H)
        >>> sorted(n for c in partition for n in c) == sorted(H.nodes)
        True
    """
    return _impl(hnx_g, cut_height=cut_height)
