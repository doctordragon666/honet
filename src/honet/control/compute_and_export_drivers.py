"""生成（或接收）有向超图并计算驱动结点集合，结果写入 CSV。"""

from __future__ import annotations

from typing import Any

from honet._deps import require
from honet.control._compute_drivers_on_hat import _compute_drivers_on_hat
from honet.control._incidence import edge_list_to_incidence
from honet.control._node_fields import add_node_fields
from honet.control._random_hypergraph import random_uniform_hypergraph


def compute_and_export_drivers(
    edge_list: list[dict[str, list[int]]] | None = None,
    n: int = 10,
    k: int = 4,
    m: int = 12,
    out_csv: str | None = "drivers.csv",
) -> dict[str, Any]:
    """计算驱动结点集合并导出 CSV。

    按结点数自动选择算法（暴力 / matching+greedy / matching），CSV 每行一个驱动
    结点，附带算法名与 dilation 标记。

    参数:
        edge_list: head/tail 形式的边列表；``None`` 时用 ``n`` / ``k`` / ``m``
            随机生成一条均匀有向超图。
        n: 结点总数（仅 ``edge_list`` 为 ``None`` 时使用）。
        k: 每条超边 tail 的大小（仅随机生成时使用）。
        m: 超边条数（仅随机生成时使用）。
        out_csv: 输出 CSV 路径；``None`` 或空串表示不写文件。

    返回:
        ``{"algorithm", "dilation_exists", "drivers"}``，``drivers`` 是结点下标
        升序列表。

    注意:
        结点表按「边列表中出现过的结点去重排序」编号，因此结点编号不连续时返回的
        下标是压缩后的行号而非原始编号（旧实现即如此）。

    示例:
        >>> res = compute_and_export_drivers(  # doctest: +SKIP
        ...     [{"head": [1], "tail": [0]}], out_csv=None
        ... )
        >>> res["algorithm"]  # doctest: +SKIP
        'brute_force'
    """
    np = require("numpy")
    HAT = require("HAT", extra="hat")
    pd = require("pandas")

    if edge_list is None:
        edge_list = random_uniform_hypergraph(n, k, m)

    incidence = edge_list_to_incidence(edge_list)
    used_nodes: set[int] = set()
    for e in edge_list:
        used_nodes.update(e["head"])
        used_nodes.update(e["tail"])
    nodes = pd.DataFrame({"Nodes": np.arange(len(used_nodes))})
    HG = HAT.Hypergraph(incidence_matrix=incidence, directed=True, nodes=nodes)
    return _compute_drivers_on_hat(add_node_fields(HG), out_csv)
