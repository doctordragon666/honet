"""把贸易边数据与权重数据合并成带权超边表。

对应旧库 hypertrade ``core.merge_ew``。业务背景：``e`` 描述“某协议/条约下
有哪些国家参与”（超边 = 协议，成员 = 国家），``w`` 描述“各国出口额”；
本函数按出口额把参与国归一化成权重，并按协议深度（``depth``）缩放。

口径与旧实现完全一致：

1. 只用出口方数据，按 ``reporterISO`` 汇总 ``primaryValue`` 得到国家贸易额；
2. 逐条超边（``agreement``）取参与国，权重 = 该国贸易额 / 参与国贸易额之和；
3. 再乘以该超边的 ``depth``；
4. 参与国贸易额之和为 0 的超边整体跳过（无法归一化）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from honet._deps import require

if TYPE_CHECKING:
    import pandas as pd

#: ``e`` 必须携带的列（``depth`` 单独处理：缺失时按旧实现抛 KeyError）。
_E_COLUMNS = ("agreement", "country")
#: ``w`` 必须携带的列。
_W_COLUMNS = ("reporterISO", "primaryValue")


def merge_ew(e: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """合并超图边数据和权重数据。

    参数:
        e: 超边数据，必须含 ``agreement``（超边标识）、``country``（成员）
            与 ``depth``（超边深度，取每组第一行）三列。
        w: 权重数据，必须含 ``reporterISO``（国家）与 ``primaryValue``（数值）。

    返回:
        长表 DataFrame，列为 ``agreement`` / ``country`` / ``weight`` / ``depth``，
        一行表示“某条超边里某个成员的归一化权重”。所有超边贸易额之和为 0 时
        返回空表（无列，与旧实现一致）。

    抛出:
        ValueError: ``e`` 缺 ``agreement`` / ``country`` 列，或 ``w`` 缺
            ``reporterISO`` / ``primaryValue`` 列。**与旧实现的差异**：旧代码
            只校验 ``e`` 的两列、对 ``w`` 缺列会直接抛 pandas 的 ``KeyError``，
            且校验发生在 ``w`` 汇总之后；本实现对两张表都做前置校验并统一抛
            ``ValueError``，错误信息里列出实际列名。
        KeyError: ``e`` 缺 ``depth`` 列（旧实现同样会在取深度时抛 ``KeyError``）。
        ImportError: 未安装 pandas（需要 ``honet[analysis]``）。

    示例:
        >>> import pandas as pd
        >>> e = pd.DataFrame(
        ...     {"agreement": ["A", "A"], "country": ["CN", "JP"], "depth": [2, 2]}
        ... )
        >>> w = pd.DataFrame({"reporterISO": ["CN", "JP"], "primaryValue": [3.0, 1.0]})
        >>> merge_ew(e, w)["weight"].tolist()
        [1.5, 0.5]
    """
    pd = require("pandas")

    for frame, columns, name in ((e, _E_COLUMNS, "e"), (w, _W_COLUMNS, "w")):
        absent = [column for column in columns if column not in frame.columns]
        if absent:
            msg = (
                f"{name} 必须包含列 {list(columns)}，缺少 {absent}；实际列为 {list(frame.columns)}"
            )
            raise ValueError(msg)

    # 仅使用出口数据（reporterISO 为出口国）的贸易额作为国家贸易额
    exporter_sum = w.groupby("reporterISO")["primaryValue"].sum()

    weighted_hyperedges: list[dict[str, object]] = []
    for agreement, group in e.groupby("agreement"):
        depth = group["depth"].iloc[0]
        countries = group["country"].unique().tolist()
        trades = {country: float(exporter_sum.get(country, 0.0)) for country in countries}
        total = sum(trades.values())
        if total <= 0:
            # 该超边的参与国都没有贸易额，权重无法归一化，整体跳过
            continue
        for country, trade in trades.items():
            weighted_hyperedges.append(
                {
                    "agreement": agreement,
                    "country": country,
                    "weight": (trade / total) * depth,
                    "depth": depth,
                }
            )

    return pd.DataFrame(weighted_hyperedges)
