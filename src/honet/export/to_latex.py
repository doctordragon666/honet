"""把 DataFrame 渲染成完整的 LaTeX 表格源码。

对应旧库 hypertrade ``dump.to_latex``。旧库用 pandas 的 ``DataFrame.to_latex``
之外另写一套版式（``center`` + ``tabular`` + ``bicaption``），以便中英文双标题
在中文论文模板（如 ``ctex``）里正常显示，本模块逐字符保留该版式。

与旧实现无行为差异；只是把「渲染」下沉到 :mod:`honet.export._latex`，
本模块只保留公开入口与文档。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from honet.export._latex import render_latex_table

if TYPE_CHECKING:
    import pandas as pd


def to_latex(
    df: pd.DataFrame,
    caption_cn: str | None = None,
    caption_en: str | None = None,
    label: str | None = None,
) -> str:
    """把 DataFrame 导出为完整的 LaTeX 表格代码。

    参数:
        df: 要导出的表格；行索引不参与渲染。
        caption_cn: 中文标题，写入 ``\\bicaption`` 的第一个参数。
        caption_en: 英文标题，写入 ``\\bicaption`` 的第二个参数。
            两者都为空时不输出 ``\\bicaption`` 行。
        label: 交叉引用标签，写入 ``\\label{...}``；为空时不输出该行。

    返回:
        完整的 LaTeX 源码字符串（以换行结尾），可直接 ``write_text`` 到 ``.tex``。

    抛出:
        ImportError: 未安装 pandas（需要 ``honet[analysis]``）。

    示例:
        >>> import pandas as pd
        >>> text = to_latex(pd.DataFrame({"密度": [0.5]}), caption_cn="指标")
        >>> text.startswith("\\\\begin{center}")
        True
    """
    return render_latex_table(df, caption_cn, caption_en, label)
