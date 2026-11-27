"""读回 :func:`honet.export.beautify_honcommunity` 导出的社区划分表。

对应旧库 hypertrade ``dump.load_honcommunity``。读回的是两列表
（``社团编号`` / ``国家列表``），本函数把它还原成 ``{结点: 社团编号}`` 映射，
与社区发现算法的输出形式一致，可直接喂回 ``honet.community`` 的后续分析。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from honet.export._read import read_frame

#: 社区表必须携带的两列
_COLUMNS = ("社团编号", "国家列表")


def load_honcommunity(path: str | Path, ext: str) -> dict[str, Any]:
    """加载高阶网络的社区划分数据。

    参数:
        path: 不含扩展名的文件基名，``str`` 或 ``Path`` 均可。
        ext: 扩展名类别，``"csv"`` / ``"excel"`` / ``"latex"``。

    返回:
        ``{结点: 社团编号}`` 映射。结点一律是 ``str``（写入时已 ``str()`` 化）；
        社团编号保持文件里的原始值（CSV 读回时由 pandas 推断类型）。

    抛出:
        NotImplementedError: ``ext="latex"``。**与旧实现一致**：LaTeX 表格是排版
            产物，无法可靠还原成 DataFrame。
        ValueError: ``ext`` 不在支持范围内；或表里缺少 ``社团编号`` /
            ``国家列表`` 列。**与旧实现的差异**：旧代码缺列时抛 pandas 的
            ``KeyError``，这里改为前置校验并抛 ``ValueError``，错误信息列出实际列名。
        FileNotFoundError: 文件不存在。
        ImportError: 未安装 pandas；读取 Excel 时还需要 ``openpyxl``。

    示例:
        >>> from pathlib import Path
        >>> import tempfile
        >>> tmp = Path(tempfile.mkdtemp())
        >>> _ = beautify_honcommunity({"CN": 0, "JP": 0}, tmp / "c")
        >>> load_honcommunity(tmp / "c", "csv") == {"CN": 0, "JP": 0}
        True
    """
    # pandas 由 read_frame 内部统一 require，这里不必再取一次
    frame = read_frame(path, ext)

    absent = [column for column in _COLUMNS if column not in frame.columns]
    if absent:
        msg = f"社区表必须包含列 {list(_COLUMNS)}，缺少 {absent}；实际列为 {list(frame.columns)}"
        raise ValueError(msg)

    # 与旧实现一致：先丢掉关键列缺失的行，再逐行展开成员列表
    frame = frame.dropna(subset=list(_COLUMNS))
    nodemap: dict[str, Any] = {}
    for _, row in frame.iterrows():
        community = row["社团编号"]
        for country in str(row["国家列表"]).split(","):
            nodemap[country] = community
    return nodemap
