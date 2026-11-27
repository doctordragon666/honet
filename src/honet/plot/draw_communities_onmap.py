"""在世界地图上绘制社团的空间分布。

对应旧库 ``hypertrade.draw.draw_communities_onmap``。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require
from honet.plot._axes import maybe_show

#: 社团配色轮转表（旧实现的 9 种自定义颜色，取社团数对其取模）
_COMMUNITY_COLORS = (
    "#66B2FF",
    "#99FF99",
    "#FFCC99",
    "#FF99CC",
    "#FFFF99",
    "#99FFCC",
    "#CC99FF",
    "#FF99FF",
    "#FFCCFF",
)


def draw_communities_onmap(
    nodemap: dict[str, int],
    world: Any,
    title: str = "",
) -> Any:
    """在世界地图上按社团给国家着色。

    参数:
        nodemap: ``{国家 ISO_A3_EH 代码: 社团编号}``，例如 ``{"CHN": 0}``。
            未出现在字典里的国家以浅灰显示。
        world: ``geopandas.GeoDataFrame``，至少含 ``ISO_A3_EH`` 与 ``NAME``
            两列以及几何列。可用 :func:`honet.io.load_map_data` 读取 shp。
        title: 地图标题；为空字符串时不加标题。

    返回:
        合并了社团列的地图数据框（已剔除南极洲），与旧实现一致。

    抛出:
        ImportError: 未安装 geopandas（需要 ``honet[viz]``）。
        ValueError: ``world`` 缺少必需列。

    说明:
        **与旧实现的差异**：旧代码用 cartopy 的 ``PlateCarree`` 投影并叠加
        ``cfeature.LAND`` / ``cfeature.OCEAN``，cartopy 既不在 ``honet[viz]``
        依赖里也不在本仓库的环境中；本实现改用 geopandas 自带的
        ``GeoDataFrame.plot`` 直接出图（等距圆柱近似），地图骨架用坐标轴
        边框代替，社团着色、图例、标题、南极洲过滤等语义保持一致。
        另外旧实现强制 ``plt.show()``，这里改为按需显示（总是显示，因为没有
        ``ax`` 参数可交由调用方排版）——**这一行为与旧实现相同**。
    """
    gpd = require("geopandas", extra="viz")
    plt = require("matplotlib.pyplot", extra="viz")
    pd = require("pandas", extra="viz")

    if not isinstance(world, gpd.GeoDataFrame):
        msg = f"world 必须是 geopandas.GeoDataFrame，收到 {type(world).__name__}"
        raise TypeError(msg)

    missing = {"ISO_A3_EH", "NAME"} - set(world.columns)
    if missing:
        msg = f"world 缺少必需列：{sorted(missing)}"
        raise ValueError(msg)

    communities = pd.DataFrame(list(nodemap.items()), columns=["country", "community"])
    communities_len = len(set(communities["community"]))

    merged = world.merge(communities, left_on="ISO_A3_EH", right_on="country", how="left")
    merged = merged[merged["NAME"] != "Antarctica"]

    colors = [_COMMUNITY_COLORS[index % len(_COMMUNITY_COLORS)] for index in range(communities_len)]

    figure, axes = plt.subplots(figsize=(20, 10))
    if communities_len:
        ListedColormap = require("matplotlib.colors", extra="viz", attr="ListedColormap")
        cmap = ListedColormap(colors)
        merged.plot(
            column="community",
            ax=axes,
            cmap=cmap,
            legend=False,
            missing_kwds={"color": "lightgray"},  # 无数据国家灰显
            edgecolor="black",
            linewidth=0.2,
        )
    else:
        merged.plot(ax=axes, color="lightgray", edgecolor="black", linewidth=0.2)

    axes.set_axis_off()
    axes.set_title(title if title else "", fontsize=18, fontweight="bold")

    Patch = require("matplotlib.patches", extra="viz", attr="Patch")
    legend_elements = [
        Patch(facecolor=colors[index], edgecolor="black", label=f"社团 {index}")
        for index in range(communities_len)
    ]
    if legend_elements:
        axes.legend(handles=legend_elements, loc="lower left", fontsize=12)

    figure.tight_layout()
    maybe_show()

    return merged
