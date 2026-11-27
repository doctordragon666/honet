"""读取 shp 格式的地图数据。

对应旧库 hypertrade ``core.load_map_data``。geopandas 不在 ``analysis`` extras
里，因此本模块用 ``extra="viz"`` 提示安装，并惰性调用
:func:`honet._deps.require` 取依赖——只有真正调用本函数时才会 import geopandas。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from honet._deps import require

if TYPE_CHECKING:
    import geopandas as gpd


def load_map_data(path: str | Path) -> gpd.GeoDataFrame:
    """加载地图数据。

    参数:
        path: 地图文件路径，通常是 ``.shp``（任何 geopandas 支持的矢量格式均可）。

    返回:
        ``geopandas.GeoDataFrame``，含几何列，可直接用于绘图或按区域匹配。

    抛出:
        FileNotFoundError: 文件不存在或无法识别为数据源。**与旧实现的差异**：
            旧代码捕获 ``pyogrio.errors.DataSourceError`` 后打印提示并返回空
            ``GeoDataFrame``，本实现改为抛 ``FileNotFoundError``，让“地图没读到”
            这件事无法被静默忽略。
        ImportError: 未安装 geopandas（需要 ``honet[viz]``）。

    示例:
        >>> world = load_map_data("world.shp")  # doctest: +SKIP
        >>> world.geometry.name  # doctest: +SKIP
        'geometry'
    """
    gpd = require("geopandas", extra="viz")
    try:
        return gpd.read_file(path)
    except Exception as exc:  # noqa: BLE001 - 底层异常类型随驱动版本变化
        # geopandas 的读取失败可能来自 pyogrio.DataSourceError、fiona 的
        # DriverError，或路径不存在时的各种 OSError，统一归一成 FileNotFoundError
        msg = f"地图数据文件不存在或无法读取：{path}（{exc}）"
        raise FileNotFoundError(msg) from exc
