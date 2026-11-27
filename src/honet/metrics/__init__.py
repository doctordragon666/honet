"""honet.metrics —— 超图指标（结点 / 超边 / 全局）。

本子包的全部符号都是惰性导出的：只有真正访问某个符号时才会 import
对应的实现模块，因此 ``import honet`` 不会因此拉入可选依赖。

指标分三类：

* 结点级：:func:`weighted_degree`、:func:`clustering_coefficient`、
  :func:`excess_average_degree`、:func:`eigenvector_centrality`、
  :func:`betweenness_centrality`；
* 超边级：:func:`get_avg_degree_by_edge_size`；
* 全局级：:func:`calc_entropy`、:func:`calc_clustering_coefficient_g`、
  :func:`calc_natural_connectivity`、:func:`calc_assortativity`、
  :func:`assortativity_test`、:func:`calc_reducibility`、
  :func:`sub_reducibility`，以及批量汇总的 :func:`hnxs_global`。

依赖分布：可约性两个函数（``calc_reducibility`` / ``sub_reducibility``）
与同配系数的零模型需要 ``hypergraphx``；其余函数只需要
``honet[analysis]``（numpy / scipy / hypernetx / networkx / pandas）。
"""

from __future__ import annotations

import sys
from types import ModuleType

from honet._lazy import lazy_getattr

#: ``{导出名: "实现模块:属性"}``
_EXPORTS = {
    "assortativity_test": "honet.metrics.assortativity_test:assortativity_test",
    "betweenness_centrality": "honet.metrics.betweenness_centrality:betweenness_centrality",
    "calc_assortativity": "honet.metrics.calc_assortativity:calc_assortativity",
    "calc_clustering_coefficient_g": (
        "honet.metrics.calc_clustering_coefficient_g:calc_clustering_coefficient_g"
    ),
    "calc_entropy": "honet.metrics.calc_entropy:calc_entropy",
    "calc_natural_connectivity": (
        "honet.metrics.calc_natural_connectivity:calc_natural_connectivity"
    ),
    "calc_reducibility": "honet.metrics.calc_reducibility:calc_reducibility",
    "clustering_coefficient": "honet.metrics.clustering_coefficient:clustering_coefficient",
    "eigenvector_centrality": "honet.metrics.eigenvector_centrality:eigenvector_centrality",
    "excess_average_degree": "honet.metrics.excess_average_degree:excess_average_degree",
    "get_avg_degree_by_edge_size": (
        "honet.metrics.avg_degree_by_edge_size:get_avg_degree_by_edge_size"
    ),
    "hnxs_global": "honet.metrics.hnxs_global:hnxs_global",
    "sub_reducibility": "honet.metrics.sub_reducibility:sub_reducibility",
    "weighted_degree": "honet.metrics.weighted_degree:weighted_degree",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """按需加载实现模块（PEP 562）。"""
    return lazy_getattr(globals(), __name__, _EXPORTS, name, "analysis")


class _MetricsPackage(ModuleType):
    """把被同名子模块遮蔽的公开名字还原成函数对象的包类型。

    本子包受「一个文件一个函数、文件名 = 函数名」的约定约束，14 个实现模块
    的文件名都与公开函数同名。Python 在 import 子模块时会把包属性写成模块
    对象，从而遮蔽惰性导出的函数（例如 ``honet.attack`` 的实现模块在模块级
    import 了 ``honet.metrics.clustering_coefficient``，之后
    ``honet.metrics.clustering_coefficient(H)`` 就会变成
    ``TypeError: 'module' object is not callable``）。

    PEP 562 的 ``__getattr__`` 只在属性缺失时触发，无法纠正「属性已存在但
    类型不对」的情况，因此在属性访问层面拦截一次：命中被遮蔽的公开名字就
    返回真正的实现对象并缓存。
    """

    def __getattribute__(self, name: str):
        exports = super().__getattribute__("_EXPORTS")
        if name in exports:
            value = super().__getattribute__(name)
            module_path, _, attribute = exports[name].partition(":")
            if isinstance(value, ModuleType) and value.__name__ == module_path:
                resolved = getattr(value, attribute)
                super().__setattr__(name, resolved)
                return resolved
        return super().__getattribute__(name)


sys.modules[__name__].__class__ = _MetricsPackage
