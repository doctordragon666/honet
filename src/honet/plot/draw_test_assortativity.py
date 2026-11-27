"""绘制度与余平均度的散点图。

对应旧库 ``hypertrade.draw.draw_test_assortativity``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from honet._deps import require
from honet.plot._axes import resolve_axes
from honet.plot._style import apply_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def draw_test_assortativity(
    result: dict[Any, float],
    ax: Axes | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
) -> Axes:
    """绘制「度 → 余平均度」散点图，用于判断同配 / 异配。

    参数:
        result: ``{度: 余平均度}``，即旧库 ``test_assortativity`` 的返回值
            （同一度只保留一个余平均度，因此用字典）。
        ax: 目标坐标轴；为 ``None`` 时新建画布。
        scatter_kwargs: 追加 / 覆盖传给 ``scatter`` 的样式参数。

    返回:
        绘制所用的 ``matplotlib.axes.Axes``（旧函数返回 ``None``）。

    说明:
        旧 docstring 提到 ``show`` 参数，但函数签名里并没有它，属于文档与实现
        不一致；本实现不含该参数，也不在此处调用 ``plt.show()``——由调用方
        决定何时显示。
    """
    apply_style()
    require("matplotlib.pyplot", extra="viz")

    _, axes = resolve_axes(ax, figsize=(7, 5))

    degrees = list(result.keys())
    excesses = list(result.values())

    scatter_options: dict[str, Any] = {"s": 45, "alpha": 0.8, "color": "#1f77b4"}
    if scatter_kwargs:
        scatter_options.update(scatter_kwargs)

    axes.scatter(degrees, excesses, **scatter_options)
    axes.set_xlabel("度")
    axes.set_ylabel("余平均度")
    axes.set_title("度与余平均度散点图")
    axes.grid(True, linestyle="--", alpha=0.3)

    return axes
