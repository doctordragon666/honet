"""绘制 SIR 模型的易感者 / 感染者 / 康复者数量曲线。

对应旧库 ``hypertrade.draw.draw_sir``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from honet._deps import require
from honet.plot._axes import maybe_show, resolve_axes
from honet.plot._style import apply_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def draw_sir(
    S: Any,
    I: Any,  # noqa: E741 - 旧 API 的参数名，S/I/R 指三类人群，保持兼容
    R: Any,
    t: Any,
    title: str = "",
    ax: Axes | None = None,
) -> Axes:
    """绘制 SIR 模型的三条数量曲线。

    参数:
        S: 易感者数量历史。
        I: 感染者数量历史。
        R: 康复者数量历史。
        t: 时间点列表，长度与其余序列一致。
        title: 图表标题；为空字符串时不加标题。
        ax: 目标坐标轴。为 ``None`` 时新建画布并调用 ``plt.show()``；
            传入 ``ax`` 时不显示，由调用方统一处理。

    返回:
        绘制所用的 ``matplotlib.axes.Axes``（旧函数返回 ``None``）。

    示例:
        >>> import numpy as np
        >>> t = np.arange(4)
        >>> ax = draw_sir(4 - t, t, t * 0, t)  # doctest: +SKIP
    """
    apply_style()
    np = require("numpy", extra="viz")

    created = ax is None
    _, axes = resolve_axes(ax, figsize=(8, 5))

    axes.plot(t, S, "g-", label="易感")
    axes.plot(t, I, "r-", label="感染")
    axes.plot(t, R, "b-", label="康复")

    axes.set_xlabel("时间", fontsize=14)
    axes.set_ylabel("国家(地区)数量", fontsize=14)

    i_mean = float(np.mean(np.asarray(I, dtype=float)))
    axes.axhline(y=i_mean, color="b", linestyle="--", label=f"感染均值: {i_mean:.2f}")
    axes.legend(fontsize=14)

    if title:
        axes.set_title(title, fontsize=14, y=-0.2)

    if created:
        maybe_show()
    return axes
