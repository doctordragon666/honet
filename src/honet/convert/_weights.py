"""超边权重的内部实现。

超边权重定义为「该超边所有结点-超边关联权重之和」，即关联矩阵 ``B``
（行=结点，列=超边）的列和。没有显式关联权重时每个结点贡献 1，此时超边
权重等于超边大小。

本模块是实现细节（文件名下划线开头），不导出到 ``honet.convert`` 的公开
命名空间。多个公开转换函数共用这份逻辑，但**不互相 import 公开模块**：
包内子模块一旦被 import，import 机制会把模块对象写回 ``honet.convert`` 的
同名属性，从而遮蔽惰性导出的同名函数。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require
from honet._hnx import incidence


def edge_weight_sums(H: Any) -> dict[Any, float]:
    """返回 ``{超边标识: 权重}``，权重为关联矩阵的列和。

    参数:
        H: hypernetx 超图对象。

    返回:
        ``{超边: 权重}``，权重统一为 Python ``float``；键顺序与
        ``H.incidence_matrix(index=True)`` 的列顺序一致。

    抛出:
        ImportError: 未安装 numpy（需要 ``honet[analysis]``）。
    """
    np = require("numpy")

    # 1. 关联矩阵：行=结点、列=超边，值为关联权重（未指定时 hypernetx 默认 1）
    matrix, _, edge_names = incidence(H, weights=True)

    # 2. 按列求和；scipy 稀疏矩阵与 numpy 稠密矩阵都支持 .sum(axis=0)
    sums = np.asarray(matrix.sum(axis=0)).flatten()

    # 3. 组合成 {超边: 权重}，矩阵元素是 numpy 标量，统一转 Python float
    return {edge: float(weight) for edge, weight in zip(edge_names, sums, strict=True)}
