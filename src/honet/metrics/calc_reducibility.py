"""超图的可约性。

可约性的含义是「能否只用少数代表性层（超边规模）描述整个多层超图而不损失
信息」。``weighted=False`` 时直接调用 HyperGraphX 的官方实现；
``weighted=True`` 时改用 :func:`honet.metrics.sub_reducibility`（在官方熵上
叠加层内权重异质性成本）。
"""

from __future__ import annotations

from typing import Any

from honet import _cache
from honet._deps import require
from honet.metrics._reuse import package_function


def _to_hgx(H: Any) -> Any:
    """把 hypernetx 超图转成 HyperGraphX 超图，转换结果按结构指纹缓存。

    可约性与 ``sub_reducibility`` 都需要 HyperGraphX 对象，重复转换是纯浪费；
    转换只依赖拓扑，因此可以安全地按指纹复用。
    """
    return _cache.cached(
        H,
        "to_hypergraphx:v1",
        lambda: package_function("honet.convert", "convert_hypernetx_to_hypergraphx")(H),
    )


def calc_reducibility(H: Any, no_layer: bool = True, weighted: bool = False) -> Any:
    """计算超图的可约性。

    参数:
        H: ``hypernetx.Hypergraph``（会被转换成 HyperGraphX 超图）或
            ``hypergraphx.Hypergraph``。
        no_layer: 为 ``True`` 时只返回可约性数值；为 ``False`` 时返回
            ``(可约性, 代表层集合)`` 二元组。
        weighted: 为 ``True`` 时用加权修正版（:func:`sub_reducibility`），
            为 ``False`` 时用 HyperGraphX 官方无权实现。

    返回:
        ``no_layer=True`` 返回 ``float``；否则返回 ``(float, tuple[int, ...])``。

    抛出:
        ImportError: 未安装 hypergraphx。
        TypeError: 传入既不是 hypernetx 也不是 hypergraphx 的超图对象。

    说明:
        **与旧实现的差异**：旧代码用 ``isinstance(H_hnx, hnx.Hypergraph)`` 判断
        是否转换，其它类型会被原样传给 HyperGraphX 并在更深处抛出难懂的异常；
        这里对非 HGX 输入统一先做 hypernetx 转换，类型不符时在入口处抛
        ``TypeError``。转换沿用 ``honet.convert`` 的口径（丢掉成员数 < 2 的
        退化超边）。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["a", "b", "c"]})
        >>> value = calc_reducibility(H)
        >>> 0.0 <= value <= 1.0
        True
    """
    hgx = require("hypergraphx")
    hnx = require("hypernetx")
    reducibility = require("hypergraphx.measures.reducibility", attr="reducibility")

    def _compute() -> Any:
        # 1) 统一成 HyperGraphX 超图（HGX 的边是成员元组，不保留 hypernetx 边名）
        if isinstance(H, hgx.Hypergraph):
            graph = H
        elif isinstance(H, hnx.Hypergraph):
            graph = _to_hgx(H)
        else:
            msg = f"calc_reducibility 需要 hypernetx 或 hypergraphx 超图，收到 {type(H).__name__}"
            raise TypeError(msg)

        # 2) 按是否加权选择实现
        if weighted:
            value: Any = package_function("honet.metrics", "sub_reducibility")(
                graph, optimization="greedy"
            )
            if no_layer:
                value = value[0]
        else:
            value = reducibility(graph, optimization="greedy")
            if no_layer:
                value = value[0]
        return float(value) if no_layer else value

    # 缓存键区分 no_layer / weighted：两者返回不同口径，不能共用条目
    cache_key = f"calc_reducibility:v2:{int(no_layer)}:{int(weighted)}"
    return _cache.cached(H, cache_key, _compute)
