"""把社区划分结果整理成「社团编号 → 国家列表」表并落盘。

对应旧库 hypertrade ``dump.beautify_honcommunity``。输入是社区发现算法给出的
``{结点: 社团编号}`` 映射，输出是便于人读、也便于 :func:`load_honcommunity`
读回的两列表：一行一个社团，成员用逗号拼成一个字符串。

与旧实现的行为差异：旧代码在函数内部调用同模块的 ``beautify_global_metrics``，
本仓库不允许公开实现模块之间互相 import（会把模块对象写回子包同名属性、
遮蔽惰性导出的同名函数），因此这里改为直接调用私有模块
:func:`honet.export._write.write_frame`，落盘口径完全相同。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from honet._deps import require
from honet.export._write import write_frame


def beautify_honcommunity(nodemap: dict[Any, Any], path: str | Path, ext: str = "csv") -> Any:
    """把 ``{结点: 社团}`` 映射整理成社团表并导出。

    参数:
        nodemap: 结点到社团编号的映射，例如
            ``{"CN": 0, "JP": 0, "KR": 1}``。结点按 ``str()`` 写入「国家列表」列。
        path: 不含扩展名的文件基名，``str`` 或 ``Path`` 均可。
        ext: 扩展名类别，``"csv"``（默认）/ ``"excel"`` / ``"latex"``。

    返回:
        整理后的 DataFrame，两列：``社团编号`` 与 `` 国家列表``（成员已排序、
        以逗号连接），按社团编号升序排列。

    抛出:
        ValueError: ``nodemap`` 为空。**与旧实现一致**：空映射无法生成社团表。
        ValueError: ``ext`` 不在 ``{"csv", "excel", "latex"}`` 中。
        ImportError: 未安装 pandas；导出 Excel 时还需要 ``openpyxl``。

    示例:
        >>> from pathlib import Path
        >>> df = beautify_honcommunity({"CN": 0, "JP": 0, "KR": 1}, Path("comm"))
        >>> df["国家列表"].tolist()
        ['CN,JP', 'KR']
    """
    pd = require("pandas")

    if not nodemap:
        msg = "nodemap 不能为空"
        raise ValueError(msg)

    community_to_countries: dict[Any, list[str]] = {}
    for country, community in nodemap.items():
        community_to_countries.setdefault(community, []).append(str(country))

    records = [
        {"社团编号": community, "国家列表": ",".join(sorted(countries))}
        for community, countries in sorted(community_to_countries.items(), key=lambda item: item[0])
    ]

    result_df = pd.DataFrame(records)
    write_frame(result_df, path, ext=ext)
    return result_df
