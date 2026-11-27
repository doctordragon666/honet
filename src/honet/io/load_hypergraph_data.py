"""一次性加载超图的边数据与（可选）权重数据。

对应旧库 hypertrade ``core.load_hypergraph_data``：是
:func:`honet.io.load_edge_data` 与 :func:`honet.io.load_weight_data` 的组合入口。

实现上直接复用内部模块 :mod:`honet.io._read`，不 import 同包的两个公开函数模块：
包内子模块一旦被 import，import 机制会把模块对象写回 ``honet.io`` 的同名属性，
从而遮蔽惰性导出的同名函数（``TypeError: 'module' object is not callable``）。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, overload

from honet.io._read import read_table

if TYPE_CHECKING:
    import pandas as pd


@overload
def load_hypergraph_data(e: str | Path, w: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]: ...


@overload
def load_hypergraph_data(e: str | Path, w: None = None) -> pd.DataFrame: ...


def load_hypergraph_data(
    e: str | Path, w: str | Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    """加载超图数据。

    参数:
        e: 超边数据文件路径。
        w: 权重数据文件路径；为 ``None`` 时只加载边数据。

    返回:
        给了 ``w`` 时返回 ``(边数据, 权重数据)`` 二元组；否则只返回边数据。
        返回类型随 ``w`` 是否为 ``None`` 变化，因此用 ``@overload`` 标注。

    抛出:
        FileNotFoundError: ``e`` 或 ``w`` 指向的文件不存在。**与旧实现的差异**：
            旧代码打印 ``File not found`` 后返回两个空 DataFrame，本实现改为
            抛异常，避免空表被当作真实数据继续参与合并。
        ImportError: 未安装 pandas（需要 ``honet[analysis]``）。

    示例:
        >>> edges, weights = load_hypergraph_data("edges.csv", "weights.csv")  # doctest: +SKIP
    """
    if w is not None:
        return (
            read_table(e, label="超边数据文件"),
            read_table(w, label="超边权重数据文件"),
        )
    return read_table(e, label="超边数据文件")
