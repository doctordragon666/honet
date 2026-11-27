"""批量计算多个超图的驱动结点集合。"""

from __future__ import annotations

from typing import Any

from honet._deps import require
from honet.control._drivers_from_hnx import compute_drivers_from_hypernetx


def calc_drivers(hgs: dict[Any, Any], to_df: bool = True) -> Any:
    """批量计算多个超图的驱动结点集合。

    参数:
        hgs: ``{超图名称: hypernetx 超图}`` 的字典。
        to_df: 为 ``True`` 时返回 DataFrame，否则返回原始字典。

    返回:
        ``to_df=True`` 时返回含三列的 ``pandas.DataFrame``：

        * ``year``：``hgs`` 的键；
        * ``drivers``：驱动结点的**原始结点名**，用逗号连接；
        * ``partition``：驱动结点数 / 结点总数。

        ``to_df=False`` 时返回 ``{键: [驱动结点原始名字, ...]}``。

    注意:
        旧实现用 ``list(hg.nodes)[driver]`` 反查名字，但
        :func:`honet.convert.hypernetx_to_hat_directed` 转换时把结点名**按字典序
        排序**后重新编号，而 hypernetx 的 ``list(H.nodes)`` 是插入序，两者顺序
        通常不同，旧实现因此会取到错误的结点名。这里改用转换后超图的
        ``nodes["Names"]`` 反查，语义与「驱动结点下标 → 原始结点名」一致。

    示例:
        >>> import hypernetx as hnx  # doctest: +SKIP
        >>> calc_drivers({"2001": hnx.Hypergraph({"e1": ["a", "b"]})})  # doctest: +SKIP
    """
    pd = require("pandas")
    from honet.convert.hypernetx_to_hat_directed import hypernetx_to_hat_directed

    result: dict[Any, list[Any]] = {}
    totals: dict[Any, int] = {}
    for key, hg in hgs.items():
        HG = hypernetx_to_hat_directed(hg)
        res = compute_drivers_from_hypernetx(HG, out_csv=None, HAT_type=True)
        names = HG.nodes["Names"].tolist()
        result[key] = [names[driver] for driver in res["drivers"]]
        totals[key] = HG.nnodes

    if to_df:
        return pd.DataFrame(
            {
                "year": list(result.keys()),
                "drivers": [
                    ",".join(str(driver) for driver in drivers) for drivers in result.values()
                ],
                "partition": [
                    len(drivers) / totals[key]
                    for key, drivers in zip(result, result.values(), strict=True)
                ],
            }
        )
    return result
