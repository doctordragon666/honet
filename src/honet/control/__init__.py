"""honet.control —— 高阶网络的驱动节点控制（dilation 检测、可达性与最小驱动集）。

本子包迁移自旧库的 ``honcontrol`` 模块，依赖 HypergraphAnalysisToolbox
（导入名 ``HAT``，属于 ``honet[hat]`` 可选组）。全部符号都是惰性导出的：只有
真正访问某个符号时才会 import 对应的实现模块，因此 ``import honet`` 不会因此
拉入 HAT / networkx / pandas。

实现分两层，原因是 Python 的导入语义：

* ``_`` 前缀的私有模块放真正的算法（``_dilation`` / ``_reachability`` /
  ``_greedy_accessibility`` / ``_compute_drivers_on_hat`` 等），它们之间可以
  互相 import；
* 公开模块（``star_graph`` / ``detect_dilation_and_drivers`` 等）只做带类型
  签名的转发。若让公开模块互相 import，``import honet.control.star_graph``
  会把子模块对象写回 ``honet.control`` 的包属性，遮蔽本文件的惰性导出，使
  ``honet.control.star_graph`` 从函数变成模块。

约定：驱动节点一律是 ``0..nnodes-1`` 的整数下标（HAT 内部把结点编号成整数），
原始结点名用 ``HG.nodes["Names"]`` 反查。
"""

from __future__ import annotations

from honet._lazy import lazy_getattr

_EXPORTS = {
    "add_node_fields": "honet.control.add_node_fields:add_node_fields",
    "brute_force_min_driver_sets": (
        "honet.control.brute_force_min_driver_sets:brute_force_min_driver_sets"
    ),
    "calc_drivers": "honet.control.calc_drivers:calc_drivers",
    "compute_and_export_drivers": (
        "honet.control.compute_and_export_drivers:compute_and_export_drivers"
    ),
    "compute_drivers_from_hypernetx": (
        "honet.control.compute_drivers_from_hypernetx:compute_drivers_from_hypernetx"
    ),
    "detect_dilation_and_drivers": (
        "honet.control.detect_dilation_and_drivers:detect_dilation_and_drivers"
    ),
    "detect_dilation_with_given_drivers": (
        "honet.control.detect_dilation_with_given_drivers:detect_dilation_with_given_drivers"
    ),
    "edge_list_to_incidence": "honet.control.edge_list_to_incidence:edge_list_to_incidence",
    "greedy_accessibility": "honet.control.greedy_accessibility:greedy_accessibility",
    "hypergraph_reachable": "honet.control.hypergraph_reachable:hypergraph_reachable",
    "random_uniform_hypergraph": (
        "honet.control.random_uniform_hypergraph:random_uniform_hypergraph"
    ),
    "star_graph": "honet.control.star_graph:star_graph",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """按需加载实现模块（PEP 562）。"""
    return lazy_getattr(globals(), __name__, _EXPORTS, name, "hat")
