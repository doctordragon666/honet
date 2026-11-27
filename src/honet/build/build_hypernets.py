"""按年份批量构造多张 hypernetx 超图。

对应旧库 hypertrade ``honbuild.build_hypernets``：对给定的年份列表逐个调用
:func:`honet.build.build_hypernet`，把结果收成 ``{年份: 超图}`` 字典。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from honet.build._common import lazy_export

if TYPE_CHECKING:
    import pandas as pd


def build_hypernets(
    e: pd.DataFrame, index: list[int] | list[str], w: pd.DataFrame | None = None
) -> dict[Any, Any]:
    """按年份列表批量构造超图。

    参数:
        e: 超边表，必须含 ``agreement`` / ``country`` / ``depth`` / ``year`` 列。
        index: 年份列表，逐个作为 :func:`honet.build.build_hypernet` 的 ``i``。
        w: 权重表，可选；给定时按 ``period`` 列匹配年份。

    返回:
        ``{年份: hypernetx.Hypergraph}``，键与 ``index`` 一一对应且保持其顺序。
        某个年份在表中没有任何数据时，该年份对应的是一张空超图（不是缺键）。

    抛出:
        与 :func:`honet.build.build_hypernet` 相同。

    示例:
        >>> import pandas as pd
        >>> e = pd.DataFrame(
        ...     {
        ...         "agreement": ["A1", "A1", "A2"],
        ...         "country": ["CN", "JP", "CN"],
        ...         "depth": [2, 2, 1],
        ...         "year": [2000, 2000, 2001],
        ...     }
        ... )
        >>> nets = build_hypernets(e, [2000, 2001])
        >>> sorted(nets)
        [2000, 2001]
    """
    build_hypernet = lazy_export("honet.build", "build_hypernet")
    return {i: build_hypernet(e, w, i) for i in index}
