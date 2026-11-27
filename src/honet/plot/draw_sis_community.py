"""绘制多组社团划分下的 SIS 感染曲线（2×2 面板）。

对应旧库 ``hypertrade.draw.draw_sis_conmmunity``（旧名拼写有误）。新名
:func:`honet.plot.draw_sis_community`，旧名保留为别名。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from honet._deps import require
from honet.plot._axes import axes_grid, maybe_show
from honet.plot._community_counts import (
    as_event_dict,
    community_infected_counts,
    normalize_partition,
    plot_community_counts,
)
from honet.plot._style import apply_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: 面板字母前缀，与旧实现相同（最多 4 个面板）
_PANEL_LETTERS = "abcd"


def draw_sis_community(
    result: dict[int, Any] | list[Any],
    communities: list[list[Any]],
    title: str = "",
    ax: Axes | Any | None = None,
) -> list[Axes]:
    """绘制多套社团划分下的 SIS 感染人数曲线，每套划分占一个面板。

    参数:
        result: ``{面板序号: 事件字典}``（也可传列表）。事件字典形如
            ``{时间: [('I', 结点, 超边), ...]}``，即
            ``hon_sis(..., return_all_data=True)`` 的返回值。
        communities: 每套社团划分，``communities[i]`` 与 ``result[i]`` 配对。
            每个元素是「该套划分的社团列表」（列表的列表）；若只传单个社团的
            结点列表，会按单社团划分处理（见
            :func:`honet.plot._community_counts.normalize_partition`）。
        title: 总标题；为空字符串时不加标题。每个面板的标题形如
            ``"(a) <title> (社团0)"``。
        ax: 目标坐标轴集合；为 ``None`` 时新建 2×2 画布。传入的坐标轴数量
            不足 ``len(communities)`` 时抛 ``ValueError``。

    返回:
        实际使用的坐标轴列表（长度等于 ``len(communities)``）。旧函数返回
        ``None``。

    抛出:
        ValueError: 超过 4 个面板，或传入的坐标轴数量不足。

    说明:
        **与旧实现的差异**：旧代码把整个 ``communities`` 传给每个面板的子
        函数，于是每个面板都会画出「所有划分里所有社团」的曲线，与面板标题
        「(社团 i)」不符（而且因为把列表套列表当成结点集合，实际会直接抛
        ``TypeError: unhashable type: 'list'``，该函数在旧库中根本跑不通）。
        本实现按「第 i 个面板画第 i 套划分」的直觉语义补齐：每个面板画出该套
        划分中每个社团一条曲线。事件回放、配色轮转、坐标轴标签与旧实现一致。
    """
    apply_style()
    require("matplotlib.pyplot", extra="viz")

    if len(communities) > 4:
        msg = f"最多支持 4 个面板（2×2），收到 {len(communities)} 个社团划分"
        raise ValueError(msg)

    axes_list = axes_grid(len(communities), ax, figsize=(10, 8))

    for index, partition_entry in enumerate(communities):
        event_dict = as_event_dict(result[index])
        axes = axes_list[index]

        # 第 i 个面板画第 i 套社团划分：每个社团一条曲线
        partition = normalize_partition(partition_entry)
        t_arr, cls_counts = community_infected_counts(event_dict, partition)
        plot_community_counts(axes, t_arr, cls_counts)

        panel_title = f"({_PANEL_LETTERS[index]}) {title} (社团{index})" if title else ""
        axes.set_xlabel("时间", fontsize=14)
        axes.set_ylabel("感染节点数量", fontsize=14)
        axes.legend(fontsize=12)
        if panel_title:
            axes.set_title(panel_title, fontsize=14, y=-0.3)

    if ax is None:
        maybe_show()

    return list(axes_list)


#: 旧名别名（旧库把 community 拼成 conmmunity）
draw_sis_conmmunity = draw_sis_community
