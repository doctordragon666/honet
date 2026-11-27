"""感染率 ``tau`` 表的构造（私有，不对外导出）。

公开的 :func:`honet.contagion.generate_tau` 与相图相关的
:func:`honet.contagion.classify_point` 共用本模块。之所以把实现下沉到这里，
是因为公开模块之间不能互相 import：``honet.contagion`` 用 PEP 562 惰性导出，
一旦某个公开模块 import 另一个同名的公开子模块，Python 会把子模块对象写到
父包的命名空间里，遮蔽掉惰性缓存的函数对象（``module object is not callable``）。
"""

from __future__ import annotations

from typing import Any

from honet._hnx import edge_sizes

#: 支持的生成方式
TAU_TYPES = (
    "fixed",
    "fixed_compare",
    "size_based",
    "size_based_compare",
    "sum_based",
    "sum_based_compare",
)


def build_tau(H: Any, beta: list[float], type: str = "fixed") -> dict[int, float]:
    """按超边大小构造 ``{大小: 感染率}``。

    参数:
        H: hypernetx 超图对象。
        beta: 感染率参数列表。
        type: 生成方式，取值见 :data:`TAU_TYPES`。

    返回:
        ``{超边大小: 感染率}``。

    抛出:
        ValueError: ``beta`` 为空或 ``type`` 未知。
        IndexError: 带 ``_compare`` 的 ``type`` 只给了 1 个 ``beta`` 值。
    """
    if not beta:
        msg = "beta 参数不能为空或长度为 0"
        raise ValueError(msg)
    if type not in TAU_TYPES:
        msg = f"未知的 type：{type!r}，可用：{list(TAU_TYPES)}"
        raise ValueError(msg)

    sizes = sorted({int(size) for size in edge_sizes(H).values()})
    if type == "fixed":
        return {size: float(beta[0]) for size in sizes}
    if type == "fixed_compare":
        tau = {size: float(beta[1]) for size in sizes}
        tau[2] = float(beta[0])
        return tau
    if type == "size_based":
        return {size: float(beta[0]) * size for size in sizes}
    if type == "size_based_compare":
        tau = {size: float(beta[1]) * size for size in sizes}
        tau[2] = float(beta[0])
        return tau

    total = sum(sizes)
    if type == "sum_based":
        return {size: float(beta[0]) * size / total for size in sizes}
    tau = {size: float(beta[1]) * size / total for size in sizes}
    tau[2] = float(beta[0])
    return tau
