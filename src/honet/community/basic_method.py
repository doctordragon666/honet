"""把超图投影成普通图后用 Louvain 检测社区。"""

from __future__ import annotations

from typing import Any

from honet.community._methods import basic_method as _impl


def basic_method(hnx_g: Any, seed: int = 42) -> list[set[Any]]:
    """用 s=1 的辅助矩阵（两节图 / 团投影）把超图降为普通图，再跑 Louvain。

    超图的 ``auxiliary_matrix(s=1)`` 等价于把每条超边展开成团（clique）后得到
    的加权邻接矩阵，因此该方法是「先团投影、再用经典图社区发现」。

    参数:
        hnx_g: hypernetx 超图对象。
        seed: 传给 Louvain 的随机种子，默认 42（与旧实现一致，保证可复现）。

    返回:
        分区，每个元素是一个结点集合。集合内的结点是原始结点标签（不是整数
        下标）。

    注意:
        旧实现用 ``auxiliary_matrix(s=1, index=False)`` 取矩阵，再假设矩阵的
        行列顺序就是 ``list(hnx_g.nodes)``。该假设在 hypernetx 2.4.3 中成立，
        但属于隐式约定；这里改用 ``index=True`` 显式取回结点索引数组，按索引
        反查标签，不再依赖顺序假设。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["c", "d"]})
        >>> sorted(sorted(c) for c in basic_method(H))
        [['a', 'b'], ['c', 'd']]
    """
    return _impl(hnx_g, seed=seed)
