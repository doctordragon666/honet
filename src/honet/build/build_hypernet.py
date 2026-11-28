"""由「协议—国家」长表构造一张 hypernetx 超图。

对应旧库 hypertrade ``honbuild.build_hypernet``。数据口径：一行表示“某协议
（超边）包含某个国家（成员）”，``depth`` 列是该协议的深度；可选的 ``w``
表提供各国出口额，用于给每条超边内部的成员加权。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from honet.build._common import depth_edge_properties, hypernetx, lazy_export

if TYPE_CHECKING:
    import pandas as pd


def build_hypernet(
    e: pd.DataFrame, w: pd.DataFrame | None = None, i: int | str | None = None
) -> Any:
    """构建一张超图，可选按年份筛选并可选按出口额加权。

    参数:
        e: 超边表，必须含 ``agreement`` （超边标识）、``country`` （成员）、
            ``depth``（超边深度）三列；``i`` 不为 ``None`` 时还必须含 ``year``。
        w: 权重表，含 ``reporterISO`` / ``primaryValue``，以及 ``i`` 不为
            ``None`` 时用于筛选的 ``period`` 列。为 ``None`` 时不加权。
        i: 年份。给定时会先用 ``e['year'] == int(i)`` 与 ``w['period'] == int(i)``
            过滤，再构造成单年超图；为 ``None`` 时直接用整张表构造。

    返回:
        ``hypernetx.Hypergraph``。超边 = ``agreement``，结点 = ``country``；
        给了 ``w`` 时关联矩阵的单元格权重来自 ``merge_ew`` 归一化后的
        ``weight`` 列。

    抛出:
        ValueError: ``e`` 缺 ``agreement`` / ``country`` 列，或 ``w`` 缺
            ``reporterISO`` / ``primaryValue`` 列（由 :func:`honet.io.merge_ew`
            校验）。
        KeyError: ``e`` 缺 ``depth`` 列，或 ``i`` 不为 ``None`` 而 ``e`` 缺
            ``year`` / ``w`` 缺 ``period`` 列。
        ImportError: 未安装 pandas 或 hypernetx（需要 ``honet[analysis]``）。

    备注:
        **与旧实现的差异**：旧代码用 ``cell_weight_col=None`` 表示“不加权”，
        而 hypernetx 2.4.3 在 DataFrame 构造路径下收到 ``cell_weight_col=None``
        会直接抛 ``KeyError: ['weight'] not in index``；本实现在不加权时改为
        **不传** 该参数。另外旧代码用 ``edge_weight_prop_col='depth'`` 传深度，
        2.4.3 会把该列改名成 ``weight``、使 ``H.edges[e].depth`` 变为 ``None``；
        本实现改为把深度同时写进 ``weight`` 与 ``depth`` 两个属性，因此边权重
        仍等于深度，且深度可按原名查询。

        旧实现在**给定年份**时把深度挂成超边属性（边权重 = 深度），**不给定
        年份** 时则没有挂，边权重一律取默认值 1。这一分支不对称属于旧实现原样
        保留的行为，本实现照搬，未作“修正”。

    示例:
        >>> import pandas as pd
        >>> e = pd.DataFrame(
        ...     {"agreement": ["A1", "A1", "A2"], "country": ["CN", "JP", "CN"], "depth": [2, 2, 1]}
        ... )
        >>> H = build_hypernet(e)
        >>> sorted(H.nodes)
        ['CN', 'JP']
    """
    hnx = hypernetx()
    merge_ew = lazy_export("honet.io", "merge_ew")

    if i is not None:
        e = e[e["year"] == int(i)]
        edge_properties = depth_edge_properties(e)
        if w is not None:
            w = w[w["period"] == int(i)]
            e = merge_ew(e, w)
    else:
        edge_properties = None
        if w is not None:
            e = merge_ew(e, w)

    options: dict[str, Any] = {
        "edge_col": "agreement",
        "node_col": "country",
    }
    if edge_properties:
        # 2.4.3 的 DataFrame 路径会把 edge_weight_prop_col 指定的列改名成 weight，
        # 因此深度的两个别名都在 depth_edge_properties 里写好了
        options["edge_properties"] = edge_properties
        options["edge_weight_prop_col"] = "weight"
    if w is not None:
        # 只有合并过权重表时 e 才有 weight 列；2.4.3 不接受 cell_weight_col=None
        options["cell_weight_col"] = "weight"

    return hnx.Hypergraph(e, **options)
