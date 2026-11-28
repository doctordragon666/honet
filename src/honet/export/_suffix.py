"""把「文件基名 + 扩展名类别」解析成磁盘路径的内部工具。

旧库 hypertrade 用 ``path + '.csv'`` 这种字符串拼接，传入 ``pathlib.Path``
会直接抛 ``TypeError``。这里统一用 f-string 拼接后交给 :class:`pathlib.Path`，
因此 ``str`` 与 ``Path`` 两种入参都能用；同时把「未知扩展名」统一成
``ValueError`` （旧实现在导出侧是静默什么都不做，在读取侧会抛
``UnboundLocalError``，两种都不便于调用方排查）。

本模块是实现细节，不导出到 ``honet.export`` 的命名空间。
"""

from __future__ import annotations

from pathlib import Path

#: 扩展名类别 → 实际文件后缀，与旧实现的三个分支一一对应
_SUFFIXES = {"csv": ".csv", "excel": ".xlsx", "latex": ".tex"}


def path_with_suffix(path: str | Path, ext: str) -> Path:
    """返回 ``path`` 拼上 ``ext`` 对应后缀后的路径。

    参数:
        path: 不含扩展名的文件基名，``str`` 或 ``Path`` 均可。
        ext: 扩展名类别，只能是 ``"csv"`` / ``"excel"`` / ``"latex"``。

    返回:
        带后缀的 :class:`pathlib.Path`。``path`` 已带同名后缀时不做去重
        （``"out.csv"`` → ``"out.csv.csv"``），与旧实现的拼接口径一致。

    抛出:
        ValueError: ``ext`` 不在支持范围内，错误信息里列出可选值。

    示例:
        >>> path_with_suffix("results/metrics", "excel").name
        'metrics.xlsx'
    """
    try:
        suffix = _SUFFIXES[ext]
    except KeyError:
        msg = f"ext 只能是 {sorted(_SUFFIXES)} 之一，收到 {ext!r}"
        raise ValueError(msg) from None
    return Path(f"{path}{suffix}")
