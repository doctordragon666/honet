"""读回 :func:`honet.export.beautify_global_metrics` 导出的全局指标表。

对应旧库 hypertrade ``dump.load_global_metrics``。旧实现用的是
``DataFrame.from_csv`` / ``DataFrame.from_excel``，这两个类方法在 pandas 2.x
中已被移除（调用直接 ``AttributeError``），这里改用 ``read_csv`` / ``read_excel``，
``index_col=None`` 的口径保持一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from honet.export._read import read_frame

if TYPE_CHECKING:
    import pandas as pd


def load_global_metrics(path: str | Path, ext: str) -> pd.DataFrame:
    """加载高阶网络的整体性质数据。

    参数:
        path: 不含扩展名的文件基名，``str`` 或 ``Path`` 均可。
        ext: 扩展名类别，``"csv"`` / ``"excel"`` / ``"latex"``。

    返回:
        全局指标表，一行对应一个超图、一列对应一项指标；行索引不保留为数据列。

    抛出:
        NotImplementedError: ``ext="latex"``。**与旧实现一致**：LaTeX 表格是排版
            产物，无法可靠还原成 DataFrame。
        ValueError: ``ext`` 不在支持范围内。
        FileNotFoundError: 文件不存在。
        ImportError: 未安装 pandas；读取 Excel 时还需要 ``openpyxl``。

    与旧实现的差异:
        旧代码的 ``DataFrame.from_csv`` / ``DataFrame.from_excel`` 在 pandas 2.x
        已不存在，本实现改用模块级 ``read_csv`` / ``read_excel``（``index_col=None``），
        返回内容一致。

    示例:
        >>> from pathlib import Path
        >>> import tempfile
        >>> import pandas as pd
        >>> tmp = Path(tempfile.mkdtemp())
        >>> _ = beautify_global_metrics(pd.DataFrame({"n": [5]}), tmp / "m")
        >>> load_global_metrics(tmp / "m", "csv")["n"].tolist()
        [5]
    """
    return read_frame(path, ext)
