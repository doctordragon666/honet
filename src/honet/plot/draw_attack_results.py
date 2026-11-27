"""绘制多策略攻击下的自然连通度衰减曲线。

对应旧库 ``hypertrade.draw.draw_attack_results``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from honet.plot._axes import maybe_show, resolve_axes
from honet.plot._style import apply_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: 曲线标记轮转表，与旧实现相同
_MARKERS = ("o", "s", "^", "D", "v", "*", "P", "X")


def draw_attack_results(
    results: dict[str, dict[str, Any]],
    title: str = "",
    ax: Axes | None = None,
) -> Axes:
    """绘制不同攻击策略下「攻击比例 → 自然连通度」的折线。

    参数:
        results: ``{策略名: {"steps": [...], "nc_vals": [...]}}``，即
            :func:`honet.attack.execute_attack` 的返回值。
        title: 图表标题；为空字符串时标题为空。
        ax: 目标坐标轴；为 ``None`` 时新建画布并调用 ``plt.show()``。

    返回:
        绘制所用的 ``matplotlib.axes.Axes``（旧函数返回 ``None``）。

    抛出:
        ValueError: 某个策略的结果缺少 ``steps`` 或 ``nc_vals`` 键。
    """
    apply_style()

    created = ax is None
    _, axes = resolve_axes(ax, figsize=(8, 5))

    for index, (strategy_name, attack_result) in enumerate(results.items()):
        missing = {"steps", "nc_vals"} - set(attack_result)
        if missing:
            msg = f"策略 {strategy_name!r} 的结果缺少键：{sorted(missing)}"
            raise ValueError(msg)
        axes.plot(
            attack_result["steps"],
            attack_result["nc_vals"],
            label=strategy_name,
            marker=_MARKERS[index % len(_MARKERS)],
            markersize=5,
        )

    axes.set_xlabel("攻击比例")
    axes.set_ylabel("自然连通度")
    axes.set_title(title)
    axes.legend()
    axes.grid(alpha=0.3)

    if created:
        maybe_show()
    return axes
