"""LaTeX 表格渲染的内部实现。

本模块集中两件事：单个单元格的格式化（对应旧库 hypertrade
``dump._format_latex_value``）与整张表的渲染（对应 ``dump.to_latex`` 的
渲染部分）。``to_latex`` 与 ``_write`` 都从这里取用，避免渲染逻辑写两份。

单元格格式化口径与旧实现完全一致：

1. 缺失值（``NaN`` / ``pd.NA`` / ``None``）输出空串，表格里留白；
2. 数值型（排除 ``bool``）先转 ``float``：整数值去掉小数部分（``3.0`` → ``3``），
   其余保留 3 位小数（``0.12345`` → ``0.123``）；
3. 其他类型一律 ``str()``，不做 LaTeX 转义（与旧实现一致，下划线等特殊字符
   需调用方自行处理）。

本模块是实现细节，不导出到 ``honet.export`` 的公开命名空间；公开名字
``_format_latex_value`` 由子包惰性映射到 :func:`format_latex_value`
（见 ``honet/export/__init__.py`` 的 ``_EXPORTS``）。
"""

from __future__ import annotations

from numbers import Number
from typing import TYPE_CHECKING, Any

from honet._deps import require

if TYPE_CHECKING:
    import pandas as pd


def format_latex_value(value: Any) -> str:
    """把单个单元格的值转成 LaTeX 表格单元格文本。

    参数:
        value: 任意单元格值，通常来自 ``DataFrame.itertuples()``。

    返回:
        可直接用 ``" & ".join(...)`` 拼进表格行的字符串。

    抛出:
        ImportError: 未安装 pandas（需要 ``honet[analysis]``）。

    示例:
        >>> format_latex_value(2.0)
        '2'
        >>> format_latex_value(0.5)
        '0.500'
    """
    pd = require("pandas")

    if pd.isna(value):
        return ""
    if isinstance(value, Number) and not isinstance(value, bool):
        numeric_value = float(value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
        return f"{numeric_value:.3f}"
    return str(value)


def render_latex_table(
    frame: pd.DataFrame,
    caption_cn: str | None = None,
    caption_en: str | None = None,
    label: str | None = None,
) -> str:
    """把 DataFrame 渲染成完整的 LaTeX 表格代码。

    版式与旧库 hypertrade ``dump.to_latex`` 逐字符一致：``center`` 环境内嵌
    ``tabular``，列格式为「首列左对齐、其余列居中」（``l*{n-1}{c}``），
    行距 1.2、列距 4pt，表头与表尾各一条 ``\\hline``。

    参数:
        frame: 要渲染的表格；行索引不参与渲染。
        caption_cn: 中文标题，写入 ``\\bicaption`` 的第一个参数。
        caption_en: 英文标题，写入 ``\\bicaption`` 的第二个参数。
            两者都为空时不输出 ``\\bicaption`` 行。
        label: 交叉引用标签，写入 ``\\label{...}``；为空时不输出该行。

    返回:
        以换行结尾的完整 LaTeX 源码字符串。

    抛出:
        ImportError: 未安装 pandas（需要 ``honet[analysis]``）。

    示例:
        >>> import pandas as pd
        >>> render_latex_table(pd.DataFrame({"指标": ["密度"], "值": [0.5]}))
        ... # doctest: +ELLIPSIS
        '\\begin{center}\\n...'
    """
    require("pandas")
    columns = [str(column) for column in frame.columns]
    column_format = "l" if len(columns) == 1 else f"l*{{{len(columns) - 1}}}{{c}}"

    lines = ["\\begin{center}"]
    if caption_cn or caption_en:
        lines.append(f"\t\\bicaption[table]{{{caption_cn or ''}}}{{{caption_en or ''}}}")
    lines.append("\t{\\small")
    lines.append("\t\t\\setlength{\\tabcolsep}{4pt}")
    lines.append("\t\t\\renewcommand{\\arraystretch}{1.2}")
    lines.append(f"\t\t\\begin{{tabular}}{{{column_format}}}")
    lines.append("\t\t\t\\hline")
    lines.append("\t\t\t" + " & ".join(columns) + " \\\\")
    lines.append("\t\t\t\\hline")

    for row in frame.itertuples(index=False, name=None):
        row_text = " & ".join(format_latex_value(value) for value in row)
        lines.append(f"\t\t\t{row_text} \\\\")

    lines.append("\t\t\t\\hline")
    lines.append("\t\t\\end{tabular}")
    lines.append("\t}")
    if label:
        lines.append(f"\t\\label{{{label}}}")
    lines.append("\\end{center}")

    return "\n".join(lines) + "\n"
