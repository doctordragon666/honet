"""在同一张超图上比较多种社区划分方法。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from honet.community._methods import compare_methods as _impl
from honet.community.method import Method


def compare_methods(
    hnx_g: Any,
    methods: Iterable[Method] | None = None,
    compare_fun: Callable[[Any, Method], Any] | None = None,
) -> dict[Method, Any]:
    """对同一张超图依次施加多种方法，收集每个方法的评价指标。

    参数:
        hnx_g: hypernetx 超图对象。
        methods: 要比较的方法序列，默认比较 :class:`Method` 的全部成员。
        compare_fun: 自定义评价函数，签名为 ``compare_fun(hnx_g, method)``，
            返回值直接作为该方法的得分。为 ``None`` 时用超图模块度。

    返回:
        ``{Method 成员: 得分}``。默认得分是模块度 ``float``。某个方法抛异常
        （例如依赖缺失、输入退化）时，该方法的得分记为 ``None``，不影响其余
        方法的比较。

    注意:
        旧实现的 ``methods`` 默认值写成 ``list[Method]``（一个 ``GenericAlias``
        对象），迭代它会直接抛 ``TypeError``，必须显式传入方法列表才能用。这里
        把默认值改为 ``None`` 再展开成全部成员，因此不传也能正常工作；同时把
        单个方法失败隔离为 ``None``，而不是让整次比较中断。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["c", "d"]})
        >>> scores = compare_methods(H, [Method.KUMAR])
        >>> list(scores) == [Method.KUMAR]
        True
    """
    return _impl(hnx_g, methods=methods, compare_fun=compare_fun)
