"""在单个坐标轴上绘制「各社团感染人数」曲线（SIS，私有）。

对应旧库 ``hypertrade.draw._sub_draw_sis_community``。虽然函数名以 ``_`` 开头，
但它是 :func:`honet.plot.draw_sis_community` 的组成部分，旧库也把它当作模块级
可调用对象，因此这里仍然单独成文件并对外导出。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from honet.plot._axes import maybe_show, resolve_axes
from honet.plot._community_counts import (
    as_event_dict,
    community_infected_counts,
    plot_community_counts,
)
from honet.plot._style import apply_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _sub_draw_sis_community(
    result: Any,
    community: list[list[Any]],
    ax: Axes | None = None,
    title: str = "",
) -> Axes:
    """把 SIS 事件流折算成各社团的感染人数曲线并画到一个坐标轴上。

    参数:
        result: ``{时间: [事件, ...]}`` 事件字典（``hon_sis(...,
            return_all_data=True)`` 的返回值），事件形如 ``('I', 结点, 超边)``。
        community: 社团划分，每个元素是一个社团的结点列表。
        ax: 目标坐标轴；为 ``None`` 时新建画布并调用 ``plt.show()``。
        title: 图表标题；为空字符串时不加标题。

    返回:
        绘制所用的 ``matplotlib.axes.Axes``（旧函数返回 ``None``）。

    抛出:
        RuntimeError: 传入的是 ``(t, S, I)`` 数值统计而非事件字典。
    """
    apply_style()

    event_dict = as_event_dict(result)
    created = ax is None
    _, axes = resolve_axes(ax, figsize=(8, 5))

    t_arr, cls_counts = community_infected_counts(event_dict, community)
    plot_community_counts(axes, t_arr, cls_counts)

    axes.set_xlabel("时间", fontsize=14)
    axes.set_ylabel("感染节点数量", fontsize=14)
    axes.legend(fontsize=12)

    if title:
        axes.set_title(title, fontsize=14, y=-0.3)

    if created:
        maybe_show()
    return axes
