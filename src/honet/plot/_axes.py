"""坐标轴获取与多子图网格（私有）。

旧实现用 ``plt.sca(ax)`` 把「当前坐标轴」切到传入的 ax 上，再调用
``plt.plot`` 之类的全局接口。这种写法在 ``ax=None`` 时依赖 pyplot 的
隐式当前画布，多图并列时容易画错面板。这里统一改成显式拿 ``Axes`` 对象，
所有绘制都走 ``ax.plot`` / ``ax.set_*``。
"""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING, Any

from honet._deps import require

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def maybe_show() -> None:
    """在交互式后端上显示当前画布，非交互后端（Agg/PDF/SVG 等）静默跳过。

    旧实现无条件调用 ``plt.show()``，在无界面环境（CI、Agg 后端、导出图片）
    下只会刷一条 ``FigureCanvasAgg is non-interactive`` 警告。这里先看后端名，
    把「非交互后端不需要显示」这件事显式表达出来，其余行为不变。
    """
    plt = require("matplotlib.pyplot", extra="viz")
    backend = str(plt.get_backend()).lower()
    if any(name in backend for name in ("agg", "pdf", "svg", "ps", "cairo", "template")):
        return
    plt.show()


def resolve_axes(
    ax: Axes | None = None,
    *,
    figsize: tuple[float, float] = (6, 5),
) -> tuple[Figure, Axes]:
    """返回 ``(figure, axes)``。

    参数:
        ax: 目标坐标轴；为 ``None`` 时按 ``figsize`` 新建单图。
        figsize: 新建画布尺寸（英寸），仅在 ``ax is None`` 时生效。

    返回:
        ``(画布, 坐标轴)``；传入 ``ax`` 时返回 ``(ax.figure, ax)`` 本身，
        调用方据此可以安全地继续在传入的坐标轴上绘制。
    """
    plt = require("matplotlib.pyplot", extra="viz")
    if ax is None:
        _, axes = plt.subplots(figsize=figsize)
        return axes.figure, axes
    return ax.figure, ax


def axes_grid(
    count: int,
    ax: Axes | Any | None = None,
    *,
    figsize: tuple[float, float] = (10, 8),
) -> np.ndarray:
    """取出一组长度恰为 ``count`` 的坐标轴。

    参数:
        count: 需要的子图数量，必须为正整数。
        ax: 二选一：

            * ``None``：按 ``count`` 自动排布（单行两列，超过 2 个则多行两列），
              多余的空白面板会被隐藏；
            * 单个 ``Axes``：仅在 ``count == 1`` 时接受；
            * ``Axes`` 序列 / 二维 ``ndarray``：展平后取前 ``count`` 个，
              数量不足时抛 ``ValueError``。
        figsize: 自动排布时的画布尺寸。

    返回:
        一维 ``numpy.ndarray``（dtype=object），元素为 ``Axes``。

    抛出:
        ValueError: ``count`` 非正，或传入的坐标轴数量不足。
    """
    np = require("numpy", extra="viz")
    plt = require("matplotlib.pyplot", extra="viz")

    if count <= 0:
        msg = f"count 必须为正整数，收到 {count!r}"
        raise ValueError(msg)

    if ax is None:
        ncols = 2 if count > 1 else 1
        nrows = ceil(count / ncols)
        _, grid = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        flat = np.asarray(grid, dtype=object).ravel()
        for unused in flat[count:]:
            unused.set_visible(False)
        return flat[:count]

    # 单个 Axes 走这里：它没有 ravel()，但有 plot()
    if hasattr(ax, "plot"):
        if count != 1:
            msg = f"需要 {count} 个子图，但只传入了一个 Axes"
            raise ValueError(msg)
        return np.asarray([ax], dtype=object)

    flat = np.asarray(ax, dtype=object).ravel()
    if flat.size < count:
        msg = f"需要 {count} 个子图，但只传入了 {flat.size} 个 Axes"
        raise ValueError(msg)
    return flat[:count]
