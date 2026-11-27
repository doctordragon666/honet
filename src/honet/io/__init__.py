"""honet.io —— 外部数据（超边表、权重表、地图）的读取与合并。

本子包的全部符号都是惰性导出的：只有真正访问某个符号时才会 import
对应的实现模块，因此 ``import honet`` 不会因此拉入可选依赖。

四个 ``load_*`` / ``merge_ew`` 函数只需要 pandas（``honet[analysis]``）；
:func:`load_map_data` 依赖 geopandas，因此单独挂在 ``viz`` 组下，缺依赖时
的提示会指向 ``honet[viz]``。
"""

from __future__ import annotations

from honet._lazy import lazy_getattr

#: ``{导出名: "实现模块:属性"}``；extra 由各模块自身在 require() 时指定，
#: 这里的 "analysis" 只用于 ``load_*`` 模块本身 import 失败时的提示。
_EXPORTS = {
    "load_edge_data": "honet.io.load_edge_data:load_edge_data",
    "load_hypergraph_data": "honet.io.load_hypergraph_data:load_hypergraph_data",
    "load_map_data": "honet.io.load_map_data:load_map_data",
    "load_weight_data": "honet.io.load_weight_data:load_weight_data",
    "merge_ew": "honet.io.merge_ew:merge_ew",
}

#: 需要 geopandas 的符号，缺失依赖时提示 ``honet[viz]`` 而非 ``honet[analysis]``。
_VIZ_EXPORTS = frozenset({"load_map_data"})

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """按需加载实现模块（PEP 562）。"""
    extra = "viz" if name in _VIZ_EXPORTS else "analysis"
    return lazy_getattr(globals(), __name__, _EXPORTS, name, extra)
