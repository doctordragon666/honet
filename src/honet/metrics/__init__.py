"""honet.metrics —— 超图指标（结点 / 超边 / 全局）。

本子包的全部符号都是惰性导出的：只有真正访问某个符号时才会 import
对应的实现模块，因此 ``import honet`` 不会因此拉入可选依赖。

实现按**粒度**收敛到三个文件（辅助函数不计）：

* :mod:`honet.metrics.nodemetrics` —— 结点级，每个结点返回一个分值：
  :func:`weighted_degree`、:func:`clustering_coefficient`、
  :func:`excess_average_degree`、:func:`eigenvector_centrality`、
  :func:`betweenness_centrality`、:func:`temporal_closeness_centrality`；
* :mod:`honet.metrics.edgemetrics` —— 超边级，按超边属性分组：
  :func:`get_avg_degree_by_edge_size`；
* :mod:`honet.metrics.globalmetrics` —— 全局级，整张超图压成一个标量或一张表：
  :func:`calc_entropy`、:func:`calc_information_gain`、
  :func:`calc_clustering_coefficient_g`、:func:`calc_natural_connectivity`、
  :func:`assortativity_test`、:func:`calc_assortativity`、
  :func:`calc_reducibility`、:func:`sub_reducibility`，
  以及批量汇总的 :func:`hnxs_global`。

另有私有辅助模块：:mod:`honet.metrics._assortativity`（同配系数核心计算）、
:mod:`honet.metrics._reuse`（跨模块取用公开函数的统一入口）。

依赖分布：可约性两个函数（``calc_reducibility`` / ``sub_reducibility``）
与同配系数的零模型需要 ``hypergraphx``；其余函数只需要
``honet[analysis]`` （numpy / scipy / hypernetx / networkx / pandas）。
"""

from __future__ import annotations

from honet._lazy import lazy_getattr

#: ``{导出名: "实现模块:属性"}``
_EXPORTS = {
    "assortativity_test": "honet.metrics.globalmetrics:assortativity_test",
    "betweenness_centrality": "honet.metrics.nodemetrics:betweenness_centrality",
    "calc_assortativity": "honet.metrics.globalmetrics:calc_assortativity",
    "calc_clustering_coefficient_g": ("honet.metrics.globalmetrics:calc_clustering_coefficient_g"),
    "calc_entropy": "honet.metrics.globalmetrics:calc_entropy",
    "calc_information_gain": "honet.metrics.globalmetrics:calc_information_gain",
    "calc_natural_connectivity": "honet.metrics.globalmetrics:calc_natural_connectivity",
    "calc_reducibility": "honet.metrics.globalmetrics:calc_reducibility",
    "clustering_coefficient": "honet.metrics.nodemetrics:clustering_coefficient",
    "eigenvector_centrality": "honet.metrics.nodemetrics:eigenvector_centrality",
    "excess_average_degree": "honet.metrics.nodemetrics:excess_average_degree",
    "get_avg_degree_by_edge_size": "honet.metrics.edgemetrics:get_avg_degree_by_edge_size",
    "hnxs_global": "honet.metrics.globalmetrics:hnxs_global",
    "sub_reducibility": "honet.metrics.globalmetrics:sub_reducibility",
    "temporal_closeness_centrality": "honet.metrics.nodemetrics:temporal_closeness_centrality",
    "weighted_degree": "honet.metrics.nodemetrics:weighted_degree",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """按需加载实现模块（PEP 562）。"""
    return lazy_getattr(globals(), __name__, _EXPORTS, name, "analysis")
