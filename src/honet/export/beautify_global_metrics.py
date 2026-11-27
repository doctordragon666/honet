"""把全局指标表按指定格式落盘。

对应旧库 hypertrade ``dump.beautify_global_metrics``。「美化」指的是统一出口：
调用方只给文件基名与格式类别，由本函数决定后缀（``.csv`` / ``.xlsx`` / ``.tex``）
与写法，避免每个脚本各写一遍 ``to_csv`` / ``to_excel`` / ``to_latex``。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from honet.export._write import write_frame

if TYPE_CHECKING:
    import pandas as pd


def beautify_global_metrics(
    global_metrics: pd.DataFrame,
    path: str | Path,
    ext: str = "csv",
    index: bool = False,
) -> Path:
    """把全局指标表导出成 CSV / Excel / LaTeX。

    参数:
        global_metrics: 全局指标表，一行通常对应一个超图、一列对应一项指标。
        path: 不含扩展名的文件基名，``str`` 或 ``Path`` 均可。
        ext: 扩展名类别，``"csv"``（默认）/ ``"excel"`` / ``"latex"``。
        index: 是否把行索引一并写出；对 ``"latex"`` 无效（LaTeX 表格不含索引）。

    返回:
        实际写入的文件路径。

    抛出:
        ValueError: ``ext`` 不在 ``{"csv", "excel", "latex"}`` 中。
            **与旧实现的差异**：旧代码对未知 ``ext`` 静默什么都不做（返回
            ``None``），本实现改为抛异常，避免“以为导出成功、实际没有文件”。
        ImportError: 未安装 pandas；导出 Excel 时还需要 ``openpyxl``。

    示例:
        >>> import pandas as pd
        >>> from pathlib import Path
        >>> out = beautify_global_metrics(pd.DataFrame({"n": [5]}), Path("metrics"))
        >>> out.name
        'metrics.csv'
    """
    return write_frame(global_metrics, path, ext=ext, index=index)
