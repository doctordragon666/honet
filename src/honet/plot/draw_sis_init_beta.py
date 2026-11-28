"""绘制 SIS 模型「初始感染比例 → 峰值 / 到达峰值时间」的双折线图。

对应旧库 ``hypertrade.draw.draw_sis_init_beta``。旧实现只有一个空壳
（函数体为空、直接返回 ``None``），docstring 里描述的目标形态是：

* 输入是一张表格，列含「指标 / 峰值 / 时间 / P」；
* ``P`` 作横坐标（初始感染比例）；
* 峰值与时间各占一个子图，不同指标用不同标记与颜色，并把每条曲线
  最高点的标记单独强调。

本实现按该描述补齐了功能，语义细节（标记形状、强调方式）在旧库中并无代码
可对照，属于**无法与旧实现逐字节对齐**的部分。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from honet._deps import require
from honet.plot._axes import axes_grid
from honet.plot._style import apply_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: 曲线标记轮转表
_MARKERS = ("o", "s", "^", "D", "v", "*", "P", "X")


def draw_sis_init_beta(
    result: Any,
    ax: Axes | Any | None = None,
    title: str = "",
) -> tuple[Axes, Axes]:
    """绘制初始感染比例对「感染峰值」与「到达峰值时间」的影响。

    参数:
        result: 结果表格，``pandas.DataFrame`` 或等价的 ``{列名: 序列}`` 字典。
            需要含 ``"P"`` （初始感染比例，作横坐标）以及至少一个指标列；
            列名含「峰值」的列画进左图，含「时间」的列画进右图。
            也可以传 ``{"指标": [...], "峰值": [...], "时间": [...], "P": [...]}``
            这种「长表」形态——此时按 ``"指标"`` 列分组，每组一条曲线。
        ax: 二选一：``None`` 自动新建 1×2 画布；或长度 ≥ 2 的 ``Axes`` 序列。
        title: 总标题；为空字符串时不加标题。

    返回:
        ``(峰值子图, 时间子图)`` 二元组。

    抛出:
        ValueError: 缺少 ``"P"`` 列，或既无「峰值」列也无「时间」列。
    """
    apply_style()
    np = require("numpy", extra="viz")

    frame = _as_frame(result)
    if "P" not in frame:
        msg = f"result 缺少 'P' 列（初始感染比例），实际列：{sorted(frame)}"
        raise ValueError(msg)

    peak_cols = [col for col in frame if "峰值" in str(col)]
    time_cols = [col for col in frame if "时间" in str(col)]
    if not peak_cols and not time_cols:
        msg = f"result 中既没有含「峰值」的列，也没有含「时间」的列，实际列：{sorted(frame)}"
        raise ValueError(msg)

    peak_ax, time_ax = axes_grid(2, ax, figsize=(12, 5))

    for axes, columns, ylabel in (
        (peak_ax, peak_cols, "峰值"),
        (time_ax, time_cols, "到达峰值时间"),
    ):
        _plot_metric_curves(axes, np, frame, columns, ylabel)

    if title:
        peak_ax.figure.suptitle(title, fontsize=14)

    return peak_ax, time_ax


def _as_frame(result: Any) -> Any:
    """把 ``DataFrame`` / 字典 / 记录列表统一成 ``{列名: list}`` 的字典视图。"""
    # 有 columns 属性即视为 DataFrame 协议，避免在这里 import pandas
    if hasattr(result, "columns") and hasattr(result, "__getitem__"):
        return {str(col): list(result[col]) for col in result.columns}
    if isinstance(result, dict):
        return {str(key): list(value) for key, value in result.items()}
    if isinstance(result, (list, tuple)):
        # 记录列表：[{"P": .., "峰值": ..}, ...]
        keys = {key for record in result for key in record}
        return {str(key): [record[key] for record in result] for key in keys}
    msg = f"result 必须是 DataFrame、字典或记录列表，收到 {type(result).__name__}"
    raise TypeError(msg)


def _plot_metric_curves(
    axes: Any,
    np: Any,
    frame: dict[str, list[Any]],
    columns: list[str],
    ylabel: str,
) -> None:
    """在 ``axes`` 上按指标画折线，并强调每条曲线最高点的标记。"""
    axes.set_xlabel("初始感染比例 P")
    axes.set_ylabel(ylabel)

    if not columns:
        axes.set_title(f"（无{ylabel}数据）")
        return

    p_values = np.asarray(frame["P"], dtype=float)
    # 长表形态：用 "指标" 列区分不同曲线，横坐标重复出现，需要按指标分组
    groups = frame.get("指标")

    for index, column in enumerate(columns):
        marker = _MARKERS[index % len(_MARKERS)]
        values = np.asarray(frame[column], dtype=float)

        if groups is None:
            axes.plot(p_values, values, marker=marker, label=column)
            _highlight_peak(axes, np, p_values, values, marker)
            continue

        for offset, name in enumerate(dict.fromkeys(groups)):
            mask = np.asarray([str(item) == str(name) for item in groups])
            if not mask.any():
                continue
            group_p = p_values[mask]
            group_v = values[mask]
            marker_for_group = _MARKERS[(index + offset) % len(_MARKERS)]
            axes.plot(group_p, group_v, marker=marker_for_group, label=f"{column}-{name}")
            _highlight_peak(axes, np, group_p, group_v, marker_for_group)

    axes.legend()


def _highlight_peak(axes: Any, np: Any, xs: Any, ys: Any, marker: str) -> None:
    """把曲线最高点画成更大的同色标记（旧 docstring 要求的强调方式）。"""
    if len(ys) == 0:
        return
    peak = int(np.argmax(ys))
    axes.plot(
        xs[peak],
        ys[peak],
        marker=marker,
        markersize=11,
        markerfacecolor="none",
        markeredgewidth=2,
        color=axes.lines[-1].get_color() if axes.lines else None,
    )
