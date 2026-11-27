"""绘制 SIS 模型的相图。

对应旧库 ``hypertrade.draw.draw_sis_phase``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from honet._deps import require
from honet.plot._axes import resolve_axes
from honet.plot._style import apply_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: 相图两态配色：0=爆发感染（红），1=无感染（浅粉）
_PHASE_COLORS = ("#fb6a4a", "#fde0dd")

#: 与 :data:`_PHASE_COLORS` 一一对应的面板文字
_PHASE_LABELS = {0: "爆发感染", 1: "无感染"}


def draw_sis_phase(
    phase_map: Any,
    beta2_border: tuple[float, float],
    beta_higher_border: tuple[float, float],
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """绘制 SIS 模型的 ``(beta2, beta_high)`` 相图。

    参数:
        phase_map: 二维相图矩阵，行对应 ``beta_higher_border`` 方向、
            列对应 ``beta2_border`` 方向；``0`` 表示爆发感染、``1`` 表示无感染
            （与 :func:`honet.contagion.sis_phase` 的输出口径一致）。
        beta2_border: ``beta2`` 轴的 ``(下界, 上界)``，用于设置图像 extent。
        beta_higher_border: ``beta_{m>=3}`` 轴的 ``(下界, 上界)``。
        ax: 目标坐标轴；为 ``None`` 时新建画布（不显示）。

    返回:
        ``(figure, axes)`` 二元组，与旧实现一致。

    说明:
        旧实现在每个取值区域的重心附近标注「爆发感染 / 无感染」。这里保留该
        行为，但重心用掩码坐标的均值直接计算，并回退到离重心最近的格点，
        因此单点区域也能标出文字。
    """
    apply_style()
    np = require("numpy", extra="viz")

    figure, axes = resolve_axes(ax, figsize=(6, 5))

    ListedColormap = require("matplotlib.colors", extra="viz", attr="ListedColormap")
    BoundaryNorm = require("matplotlib.colors", extra="viz", attr="BoundaryNorm")

    data = np.asarray(phase_map)
    if data.ndim != 2:
        msg = f"phase_map 必须是二维矩阵，收到 {data.ndim} 维"
        raise ValueError(msg)

    cmap = ListedColormap(list(_PHASE_COLORS))
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    extent = [
        float(beta2_border[0]),
        float(beta2_border[1]),
        float(beta_higher_border[0]),
        float(beta_higher_border[1]),
    ]

    axes.imshow(
        data,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="none",
    )

    axes.set_xlabel(r"$\beta_2$", fontsize=14)
    axes.set_ylabel(r"$\beta_{m\geq 3}$", fontsize=14)

    # 每个取值区域的标注位置：该区域所有格点的重心，再吸附到最近的格点
    ny, nx = data.shape
    xs = np.linspace(extent[0], extent[1], nx)
    ys = np.linspace(extent[2], extent[3], ny)
    grid_x, grid_y = np.meshgrid(xs, ys)

    for value, text in _PHASE_LABELS.items():
        mask = data == value
        if not mask.any():
            continue
        center_x, center_y = grid_x[mask].mean(), grid_y[mask].mean()
        distance = (grid_x[mask] - center_x) ** 2 + (grid_y[mask] - center_y) ** 2
        nearest = int(np.argmin(distance))
        axes.text(
            grid_x[mask][nearest],
            grid_y[mask][nearest],
            text,
            ha="center",
            va="center",
            fontsize=14,
        )

    return figure, axes
