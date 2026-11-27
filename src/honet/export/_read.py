"""按扩展名类别读取数据表的内部工具。

``load_global_metrics`` 与 ``load_honcommunity`` 共用同一套「基名 + 后缀 → 读表」
的逻辑，集中放在这里，避免两处重复（与 ``honet.io._read`` 的写法对称）。

与旧库 hypertrade 的差异：

* 旧实现在读取侧遇到未知 ``ext`` 会因局部变量未赋值抛 ``UnboundLocalError``，
  这里统一成 ``ValueError`` 并列出可选值；
* 旧实现 ``load_global_metrics`` 用的是 ``DataFrame.from_csv`` / ``from_excel``，
  这两个类方法在 pandas 2.x 中已被移除，这里改用 ``read_csv`` / ``read_excel``。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from honet._deps import require
from honet.export._suffix import path_with_suffix

if TYPE_CHECKING:
    import pandas as pd


def read_frame(path: str | Path, ext: str) -> pd.DataFrame:
    """按 ``ext`` 读取 ``path`` 对应的数据表。

    参数:
        path: 不含扩展名的文件基名，``str`` 或 ``Path`` 均可。
        ext: 扩展名类别，``"csv"`` / ``"excel"`` / ``"latex"``。

    返回:
        读到的 DataFrame，索引不作为数据列保留（``index_col=None``）。

    抛出:
        ValueError: ``ext`` 不在支持范围内。
        NotImplementedError: ``ext="latex"``。**与旧实现一致**：LaTeX 表格是
            排版产物，列类型与缺失值信息已丢失，无法可靠还原成 DataFrame。
        FileNotFoundError: 文件不存在。
        ImportError: 未安装 pandas，或读取 Excel 时缺少 ``openpyxl`` 引擎。
    """
    pd = require("pandas")
    target = path_with_suffix(path, ext)

    if ext == "latex":
        msg = "暂不支持从 LaTeX 表格读回 DataFrame（列类型与缺失值信息已丢失）"
        raise NotImplementedError(msg)

    if ext == "csv":
        reader = pd.read_csv
    else:  # ext == "excel"，取值已由 path_with_suffix 校验
        reader = pd.read_excel

    try:
        return reader(target, index_col=None)
    except FileNotFoundError as exc:
        msg = f"数据表文件不存在：{target}"
        raise FileNotFoundError(msg) from exc
    except ImportError as exc:  # pandas 读 .xlsx 需要 openpyxl 引擎
        msg = "读取 Excel 需要 openpyxl 引擎：pip install openpyxl"
        raise ImportError(msg) from exc
