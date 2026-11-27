"""读取超边（成员）数据表。

对应旧库 hypertrade ``core.load_edge_data``：按扩展名分派到 pandas 的
pickle 读取器或 CSV 读取器。与旧实现的唯一差异是出错方式——旧代码
``print`` 一行提示后返回空 ``DataFrame``，本实现改为抛 ``FileNotFoundError``，
避免“读取失败却拿到空表”这种静默错误继续流向下游分析。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from honet.io._read import read_table

if TYPE_CHECKING:
    import pandas as pd


def load_edge_data(path: str | Path) -> pd.DataFrame:
    """加载超边数据。

    以 ``.pkl`` 结尾的路径按 pickle 读取，其余路径一律按 CSV 读取
    （与旧实现的分派规则一致）。

    参数:
        path: 超边数据文件路径。

    返回:
        pandas DataFrame，一行表示“某条超边包含某个成员”。

    抛出:
        FileNotFoundError: 文件不存在。**与旧实现的差异**：旧代码打印
            ``File not found`` 后返回空 DataFrame，本实现改为抛异常，
            让调用方显式处理缺失文件。
        ImportError: 未安装 pandas（需要 ``honet[analysis]``）。

    示例:
        >>> load_edge_data("edges.csv").columns.tolist()  # doctest: +SKIP
        ['agreement', 'country', 'depth']
    """
    return read_table(path, label="超边数据文件")
