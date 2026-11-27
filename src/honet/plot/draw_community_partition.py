"""绘制社团划分的规模分布饼图。

对应旧库 ``hypertrade.draw.draw_community_partition``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from honet.plot._axes import maybe_show, resolve_axes
from honet.plot._style import apply_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def draw_community_partition(
    nodemap: list[list[Any]],
    title: str = "",
    ax: Axes | None = None,
) -> Axes:
    """绘制各社团的结点数量占比饼图。

    参数:
        nodemap: 社团划分，每个元素是一个社团的结点列表。
        title: 图表标题；为空字符串时标题为空。
        ax: 目标坐标轴；为 ``None`` 时新建画布并调用 ``plt.show()``。

    返回:
        绘制所用的 ``matplotlib.axes.Axes``（旧函数返回 ``None``）。

    说明:
        **与旧实现的差异**：旧代码在 ``ax is None`` 时先执行 ``ax.pie(...)``
        再判断 ``ax is None``，因此不传 ``ax`` 必然 ``AttributeError``；
        本实现先解析坐标轴再绘制。
    """
    apply_style()

    if not nodemap:
        msg = "nodemap 不能为空，至少需要一个社团"
        raise ValueError(msg)

    created = ax is None
    _, axes = resolve_axes(ax, figsize=(6, 6))

    community_sizes = {index: len(members) for index, members in enumerate(nodemap)}
    labels = [f"社团{index}" for index in community_sizes]
    sizes = list(community_sizes.values())

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

    if created:
        maybe_show()
    return axes
