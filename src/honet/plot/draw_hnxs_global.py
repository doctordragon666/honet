"""绘制高阶网络的整体性质随时间的归一化趋势。

对应旧库 ``hypertrade.draw.draw_hnxs_global``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from honet._deps import require
from honet.plot._axes import resolve_axes
from honet.plot._style import apply_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def draw_hnxs_global(
    global_metrics: Any,
    title: str = "",
    ax: Axes | None = None,
) -> Axes:
    """把 ``hnxs_global`` 的多项整体指标归一化后画成折线。

    参数:
        global_metrics: ``pandas.DataFrame``，索引为年份、每列一项整体指标
            （如 :func:`honet.metrics.hnxs_global` 的输出）。
        title: 图表标题；为空时使用「超网络特征变化趋势」。
        ax: 目标坐标轴；为 ``None`` 时新建画布。

    返回:
        绘制所用的 ``matplotlib.axes.Axes``（旧函数返回 ``None``）。

    说明:
        **与旧实现的差异**：旧代码调用 ``global_metrics_scaled.plot(...)``，
        这是 DataFrame 的便捷接口，会**另建一张图**，传进来的 ``ax`` 实际被
        忽略；本实现改为逐列 ``ax.plot``，``ax`` 参数真正生效。
        归一化口径不变：按列做 min-max 缩放，最大最小相等时取 ``1``。
    """
    apply_style()
    require("matplotlib.pyplot", extra="viz")
    np = require("numpy", extra="viz")

    _, axes = resolve_axes(ax, figsize=(10, 6))

    columns = list(global_metrics.columns)
    index = list(global_metrics.index)

    for column in columns:
        values = np.asarray(global_metrics[column], dtype=float)
        low, high = float(np.min(values)), float(np.max(values))
        if high == low:
            # 最大最小相等（含单点或常量序列）：按旧实现口径填 1
            scaled = np.ones_like(values)
        else:
            scaled = (values - low) / (high - low)
        axes.plot(index, scaled, label=str(column))

    axes.set_title(title if title else "超网络特征变化趋势", fontsize=14, y=-0.3)
    axes.set_xlabel("年份")
    axes.set_ylabel("值")
    axes.tick_params(axis="x", labelrotation=0)
    axes.legend(loc="upper right")

    return axes
