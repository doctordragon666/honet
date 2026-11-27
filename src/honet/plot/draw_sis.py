"""绘制 SIS 模型的易感者 / 感染者数量曲线。

对应旧库 ``hypertrade.draw.draw_sis``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from honet._deps import require
from honet.plot._axes import maybe_show, resolve_axes
from honet.plot._style import apply_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def draw_sis(
    S: Any,
    I: Any,  # noqa: E741 - 旧 API 的参数名，S/I 分别指易感者与感染者，保持兼容
    t: Any,
    title: str = "",
    ax: Axes | None = None,
    y_end: float = -0.2,
) -> Axes:
    """绘制 SIS 模型的易感者与感染者数量随时间变化曲线。

    参数:
        S: 易感者数量历史（序列或 numpy 数组）。
        I: 感染者数量历史，长度与 ``S`` 一致。
        t: 时间点列表，长度与 ``S`` 一致。
        title: 图表标题；为空字符串时不加标题。
        ax: 目标坐标轴。为 ``None`` 时新建画布并调用 ``plt.show()``
            （与旧实现一致）；传入 ``ax`` 时不显示，由调用方统一处理。
        y_end: 标题的纵向位置（坐标轴坐标），默认 ``-0.2``，用于把标题放到
            图的下方，与旧实现相同。

    返回:
        绘制所用的 ``matplotlib.axes.Axes``。**与旧实现的差异**：旧函数返回
        ``None``，这里返回坐标轴对象，方便多图排版与测试断言。

    示例:
        >>> import numpy as np
        >>> t = np.arange(5)
        >>> ax = draw_sis(10 - t, t, t)  # doctest: +SKIP
    """
    apply_style()
    np = require("numpy", extra="viz")

    created = ax is None
    _, axes = resolve_axes(ax, figsize=(6, 5))

    axes.plot(t, S, "g-", label=r"$S$")
    axes.plot(t, I, "r-", label=r"$I$")

    axes.set_xlabel(r"$t$", fontsize=14)
    axes.set_ylabel(r"$V$", fontsize=14)

    # 感染者均值参考线：直观标出平台期水平
    i_mean = float(np.mean(np.asarray(I, dtype=float)))
    axes.axhline(y=i_mean, color="b", linestyle="--", label=rf"$\overline{{I}}:\ {i_mean:.2f}$")
    axes.legend(fontsize=14)

    if title:
        axes.set_title(title, fontsize=14, y=y_end)

    if created:
        maybe_show()
    return axes
