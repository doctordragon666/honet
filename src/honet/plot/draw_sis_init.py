"""绘制 SIS 模型「初始感染规模 → 最终感染规模」的多曲线对比。

对应旧库 ``hypertrade.draw.draw_sis_init``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from honet.plot._axes import maybe_show, resolve_axes
from honet.plot._style import apply_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def draw_sis_init(
    result: dict[Any, tuple[Any, Any, Any]],
    ax: Axes | None = None,
    title: str = "",
) -> Axes:
    """绘制不同初始感染规模下感染者数量随时间的曲线。

    参数:
        result: ``{曲线标签: (S, I, t)}``。标签直接作为图例文字（旧实现的
            ``draw_sis_init`` 就按这个形状遍历），因此既可以是初始感染节点数，
            也可以是任意可哈希的说明文字。
        ax: 目标坐标轴；为 ``None`` 时新建画布并调用 ``plt.show()``。
        title: 图表标题；为空字符串时不加标题。

    返回:
        绘制所用的 ``matplotlib.axes.Axes``（旧函数返回 ``None``）。

    示例:
        >>> import numpy as np
        >>> t = np.arange(3)
        >>> ax = draw_sis_init({1: (5 - t, t, t)})  # doctest: +SKIP
    """
    apply_style()

    created = ax is None
    _, axes = resolve_axes(ax, figsize=(8, 6))

    for label, series in result.items():
        # S/I/t 是旧库沿用的命名，S 与 t 此处未用到
        _S, infected, _t = series
        axes.plot(_t, infected, label=label)

    axes.set_xlabel("时间步")
    axes.set_ylabel("感染节点数量")
    axes.legend()
    axes.grid(True)

    if title:
        axes.set_title(title, fontsize=14, y=-0.2)

    if created:
        maybe_show()
    return axes
