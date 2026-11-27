"""按扩展名类别把 DataFrame 落盘的内部工具。

``beautify_global_metrics`` 与 ``beautify_honcommunity`` 共用同一套「基名 + 后缀 →
写文件」的逻辑，集中放在这里，避免两处重复（与 ``honet.io._read`` 的写法对称）。

公开模块之间刻意不互相 import：包内子模块一旦被 import，import 机制会把
**模块对象**写回 ``honet.export`` 的同名属性，从而遮蔽惰性导出的同名函数。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from honet._deps import require
from honet.export._latex import render_latex_table
from honet.export._suffix import path_with_suffix

if TYPE_CHECKING:
    import pandas as pd


def write_frame(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    ext: str = "csv",
    index: bool = False,
) -> Path:
    """把 DataFrame 按 ``ext`` 写到 ``path`` 对应的文件，返回实际写入路径。

    参数:
        frame: 要导出的表格。
        path: 不含扩展名的文件基名，``str`` 或 ``Path`` 均可。
        ext: 扩展名类别，``"csv"`` / ``"excel"`` / ``"latex"``。
        index: 是否把行索引一并写出（仅对 csv / excel 有效，LaTeX 表格不含索引）。

    返回:
        实际写入的文件路径。

    抛出:
        ValueError: ``ext`` 不在支持范围内。**与旧实现的差异**：旧代码遇到
            未知 ``ext`` 时静默返回 ``None``（既不报错也不写文件），本实现改为
            抛异常，避免“以为导出成功、实际没有文件”。
        ImportError: 未安装 pandas，或导出 Excel 时缺少 ``openpyxl`` 引擎。
    """
    require("pandas")
    target = path_with_suffix(path, ext)

    if ext == "csv":
        frame.to_csv(target, index=index)
    elif ext == "excel":
        try:
            frame.to_excel(target, index=index)
        except ImportError as exc:  # pandas 写 .xlsx 需要 openpyxl 引擎
            msg = "导出 Excel 需要 openpyxl 引擎：pip install openpyxl"
            raise ImportError(msg) from exc
    else:  # ext == "latex"，取值已由 path_with_suffix 校验
        target.write_text(render_latex_table(frame), encoding="utf-8")

    return target
