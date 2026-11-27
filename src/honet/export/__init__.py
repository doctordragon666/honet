"""honet.export —— 分析结果的导出与读回（CSV / Excel / LaTeX）。

本子包对应旧库 hypertrade 的 ``dump`` 模块，提供「DataFrame ↔ 磁盘文件」的往返：
``to_latex`` 把表格渲染成论文可直接引用的 LaTeX 源码，
``beautify_global_metrics`` / ``beautify_honcommunity`` 负责按格式类别落盘，
``load_global_metrics`` / ``load_honcommunity`` 负责读回。

本子包的全部符号都是惰性导出的：只有真正访问某个符号时才会 import
对应的实现模块，因此 ``import honet`` 不会因此拉入可选依赖。

``_format_latex_value`` 是实现细节（旧库里同样以下划线开头），仅供内部与测试
使用，因此可以访问但不进入 ``__all__``。
"""

from __future__ import annotations

from honet._lazy import lazy_getattr

#: ``{导出名: "实现模块:属性"}``。私有名 ``_format_latex_value`` 的实现模块内
#: 名字是 ``format_latex_value``（本仓库的私有模块内不重复写下划线前缀）。
_EXPORTS = {
    "_format_latex_value": "honet.export._latex:format_latex_value",
    "beautify_global_metrics": "honet.export.beautify_global_metrics:beautify_global_metrics",
    "beautify_honcommunity": "honet.export.beautify_honcommunity:beautify_honcommunity",
    "load_global_metrics": "honet.export.load_global_metrics:load_global_metrics",
    "load_honcommunity": "honet.export.load_honcommunity:load_honcommunity",
    "to_latex": "honet.export.to_latex:to_latex",
}

#: 公开名字（下划线开头的实现细节不参与 ``from honet.export import *``）。
__all__ = sorted(name for name in _EXPORTS if not name.startswith("_"))


def __getattr__(name: str):
    """按需加载实现模块（PEP 562）。"""
    return lazy_getattr(globals(), __name__, _EXPORTS, name, "analysis")
