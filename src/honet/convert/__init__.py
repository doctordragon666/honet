"""honet.convert —— 超图格式互转（hypernetx / XGI / HyperGraphX / NetworkX / HAT）。

本子包的全部符号都是惰性导出的：只有真正访问某个符号时才会 import
对应的实现模块，因此 ``import honet`` 不会因此拉入可选依赖。

依赖分布：

* 其余转换函数只需要 ``honet[analysis]``（numpy / pandas / hypernetx /
  networkx），XGI 与 HyperGraphX 互转额外需要 ``xgi``、``hypergraphx``；
* :func:`hypernetx_to_hat_directed` 依赖 HypergraphAnalysisToolbox（导入名
  ``HAT``），单独挂在 ``hat`` 组下，缺依赖时的提示会指向 ``honet[hat]``。
"""

from __future__ import annotations

from honet._lazy import lazy_getattr

#: ``{导出名: "实现模块:属性"}``；extra 由各模块自身在 require() 时指定，
#: 这里的 "analysis" 只用于实现模块本身 import 失败时的提示。
_EXPORTS = {
    "convert_hypergraphx_to_hypernetx": (
        "honet.convert.convert_hypergraphx_to_hypernetx:convert_hypergraphx_to_hypernetx"
    ),
    "convert_hypernetx_to_hypergraphx": (
        "honet.convert.convert_hypernetx_to_hypergraphx:convert_hypernetx_to_hypergraphx"
    ),
    "convert_hypernetx_to_nxgraph": (
        "honet.convert.convert_hypernetx_to_nxgraph:convert_hypernetx_to_nxgraph"
    ),
    "convert_hypernetx_to_xgi": "honet.convert.convert_hypernetx_to_xgi:convert_hypernetx_to_xgi",
    "convert_xgi_to_hypernetx": "honet.convert.convert_xgi_to_hypernetx:convert_xgi_to_hypernetx",
    "get_hyperedge_weights": "honet.convert.get_hyperedge_weights:get_hyperedge_weights",
    "hypernetx_to_hat_directed": (
        "honet.convert.hypernetx_to_hat_directed:hypernetx_to_hat_directed"
    ),
}

#: 需要 HAT 的符号，缺失依赖时提示 ``honet[hat]`` 而非 ``honet[analysis]``。
_HAT_EXPORTS = frozenset({"hypernetx_to_hat_directed"})

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """按需加载实现模块（PEP 562）。"""
    extra = "hat" if name in _HAT_EXPORTS else "analysis"
    return lazy_getattr(globals(), __name__, _EXPORTS, name, extra)
