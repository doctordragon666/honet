"""普通图（成对网络）的攻击序列。

面向 NetworkX 的 ``Graph``：一次算出度、余平均度、聚类系数、介数中心性、
特征向量中心性五种排序，返回 ``{策略名称: 降序结点列表}``，便于在普通图
上做多策略攻击对比（超图请用本子包其余函数）。

NetworkX 通过 ``honet._deps.require`` 获取，不在模块顶层直接 import。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require


def basic_attack_sequence(g: Any) -> dict[str, list[Any]]:
    """生成普通图的五类攻击序列（均按指标降序）。

    参数:
        g: NetworkX 无向图对象。

    返回:
        ``{策略名称: 结点列表}``，键依次为 ``"度"``、``"余平均度"``、
        ``"聚类系数"``、``"介数"``、``"特征向量中心性"``。

    示例:
        >>> import networkx as nx
        >>> g = nx.path_graph(3)
        >>> list(basic_attack_sequence(g))
        ['度', '余平均度', '聚类系数', '介数', '特征向量中心性']
    """
    nx = require("networkx")

    scores = {
        "度": dict(g.degree()),
        # 余平均度：邻居的平均度，等价于超图版本的高阶推广
        "余平均度": nx.average_neighbor_degree(g),
        "聚类系数": nx.clustering(g),
        "介数": nx.betweenness_centrality(g),
        "特征向量中心性": nx.eigenvector_centrality(g),
    }

    return {
        name: [node for node, _ in sorted(values.items(), key=lambda kv: kv[1], reverse=True)]
        for name, values in scores.items()
    }
