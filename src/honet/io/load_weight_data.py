"""读取超边（或结点）权重数据表。

对应旧库 hypertrade ``core.load_weight_data``。读取规则与
:func:`honet.io.load_edge_data` 完全一致，只是语义上承载的是权重字段
（例如贸易数据里的 ``reporterISO`` / ``primaryValue``）。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from honet.io._read import read_table

if TYPE_CHECKING:
    import pandas as pd


def load_weight_data(path: str | Path) -> pd.DataFrame:
    """加载超边权重数据。

    以 ``.pkl`` 结尾的路径按 pickle 读取，其余路径一律按 CSV 读取
    （与旧实现的分派规则一致）。

    参数:
        path: 权重数据文件路径。

    返回:
        pandas DataFrame，例如含 ``reporterISO`` 与 ``primaryValue`` 两列的
        贸易额表。

    抛出:
        FileNotFoundError: 文件不存在。**与旧实现的差异**：旧代码打印
            ``File not found`` 后返回空 DataFrame，本实现改为抛异常。
        ImportError: 未安装 pandas（需要 ``honet[analysis]``）。

    示例:
        >>> load_weight_data("weights.pkl").columns.tolist()  # doctest: +SKIP
        ['reporterISO', 'primaryValue']
    """
    return read_table(path, label="超边权重数据文件")
