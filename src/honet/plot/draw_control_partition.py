"""绘制控制节点在各社团中的分布饼图。

对应旧库 ``hypertrade.draw.draw_control_partition``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from honet._deps import require
from honet.plot._axes import resolve_axes
from honet.plot._style import apply_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def draw_control_partition(
    drivers: list[Any],
    nodemap: list[list[Any]],
    title: str = "",
    ax: Axes | None = None,
) -> Axes:
    """绘制驱动（控制）节点落在各社团的数量占比饼图。

    参数:
        drivers: 控制结点列表；不属于任何社团的结点不计入。
        nodemap: 社团划分，每个元素是一个社团的结点列表。
        title: 图表标题；为空字符串时标题为空。
        ax: 目标坐标轴；为 ``None`` 时新建画布。

    返回:
        绘制所用的 ``matplotlib.axes.Axes``（旧函数返回 ``None``）。

    说明:
        旧实现在 ``ax is None`` 时会 ``AttributeError``，且不调用
        ``plt.show()``；本实现先解析坐标轴，新建画布时自动显示。
    """
    apply_style()
    require("matplotlib.pyplot", extra="viz")

    if not nodemap:
        msg = "nodemap 不能为空，至少需要一个社团"
        raise ValueError(msg)

    _, axes = resolve_axes(ax, figsize=(6, 6))

    driver_set = set(drivers)
    community_counts = {
        index: sum(1 for driver in driver_set if driver in set(members))
        for index, members in enumerate(nodemap)
    }
    labels = [f"社团{index}" for index in community_counts]
    sizes = list(community_counts.values())

    # 全部为 0 时饼图无法成比例，改为提示文字，避免 matplotlib 抛 ValueError
    if not any(sizes):
        axes.text(0.5, 0.5, "无控制节点落入任何社团", ha="center", va="center", fontsize=14)
        axes.set_title(title, fontsize=20, y=-0.1)
        axes.axis("equal")
        return axes

    axes.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=0,
        textprops={"fontsize": 15},
    )
    axes.set_title(title, fontsize=20, y=-0.1)
    axes.axis("equal")  # 保持饼图为圆形
    axes.tick_params(axis="both", labelfontfamily="Times New Roman")
    return axes
