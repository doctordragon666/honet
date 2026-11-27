"""读取 CSV / pickle 数据表的内部工具。

``load_edge_data``、``load_weight_data`` 与 ``load_hypergraph_data`` 共用同一套
「按扩展名分派读取器」与「缺文件归一成 ``FileNotFoundError``」的逻辑，集中放在
这里避免三处重复。

注意本模块是实现细节，不导出到 ``honet.io`` 的公开命名空间（文件名下划线开头）。
公开函数模块之间刻意不互相 import：包内子模块一旦被 import，import 机制会把
**模块对象**写回 ``honet.io`` 的同名属性，从而遮蔽惰性导出的同名函数。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from honet._deps import require

if TYPE_CHECKING:
    import pandas as pd


def read_table(path: str | Path, *, label: str) -> pd.DataFrame:
    """按扩展名读取数据表，文件缺失时抛 ``FileNotFoundError``。

    以 ``.pkl`` 结尾的路径按 pickle 读取，其余路径一律按 CSV 读取
    （与旧库 hypertrade 的分派规则一致）。

    参数:
        path: 数据文件路径。
        label: 面向用户的表名，用于拼装错误信息，例如 ``"超边数据文件"``。

    返回:
        pandas DataFrame。

    抛出:
        FileNotFoundError: 文件不存在。旧实现是打印提示后返回空表，
            这里改为抛异常，避免空表被当成真实数据继续参与下游分析。
        ImportError: 未安装 pandas（需要 ``honet[analysis]``）。
    """
    pd = require("pandas")
    reader = pd.read_pickle if str(path).endswith(".pkl") else pd.read_csv
    try:
        return reader(path)
    except FileNotFoundError as exc:
        msg = f"{label}不存在：{path}"
        raise FileNotFoundError(msg) from exc
