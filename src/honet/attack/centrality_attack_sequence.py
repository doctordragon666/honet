"""基于中心性的攻击序列。

中心性刻画结点在「最短路」与「连通分界」上的位置，比度更能反映移除后的
全局影响：

* 介数中心性 —— 基于结点—超边二部图的最短路介数，度量结点作为「桥梁」
  的程度；
* 特征向量中心性 —— 基于归一化拉普拉斯矩阵的 Fiedler 向量，绝对值越大
  越靠近网络的分界位置，按绝对值降序移除。

度量实现直接复用 ``honet.metrics`` 下的单文件模块，避免重复实现。
"""

from __future__ import annotations

from typing import Any

from honet.metrics.nodemetrics import betweenness_centrality, eigenvector_centrality

#: 支持的中心性指标
_METRICS = ("betweenness", "eigenvector")


def centrality_attack_sequence(hg: Any, metric: str = "betweenness") -> list[Any]:
    """生成基于中心性的攻击序列（降序）。

    参数:
        hg: hypernetx 超图对象。
        metric: 中心性指标，``"betweenness"`` 用介数中心性，
            ``"eigenvector"`` 用特征向量中心性（按绝对值排序）。

    返回:
        结点标识列表，按中心性从大到小排列。

    抛出:
        ValueError: ``metric`` 不是 ``"betweenness"`` 或 ``"eigenvector"``。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> centrality_attack_sequence(H)[0]
        'b'
    """
    if metric == "betweenness":
        scores = betweenness_centrality(hg)
        return [node for node, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
    if metric == "eigenvector":
        scores = eigenvector_centrality(hg)
        # Fiedler 向量有正负号，重要性看绝对值，因此按绝对值降序
        return [node for node, _ in sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)]

    msg = f"metric 只能是 {_METRICS} 之一，收到 {metric!r}"
    raise ValueError(msg)
