"""高阶同配系数的核心计算（私有，不对外导出）。

同配系数衡量「超边里超度大的结点是否倾向于和超度大的结点同处一条超边」。
按文献定义（Kirkley, Felippe & Battiston, PRL 135, 247401, 2025 的式 (5)），
对每条超边取超度最大与第二大的两个结点，用它们「超度平均秩」的皮尔逊
相关系数作为该超图的同配系数。

本模块只接受 hypernetx 超图，度与成员关系全部走 ``honet._hnx`` 的兼容函数；
``calc_assortativity`` 用它既算观测值、也算零模型样本（零模型先转成
hypernetx 超图）。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require
from honet._topology import topology


def average_ranks(values: list[float]) -> Any:
    """返回带并列平均秩的秩数组（等价于 ``scipy.stats.rankdata(method="average")``）。

    参数:
        values: 待排秩的数值列表。

    返回:
        与 ``values`` 等长的 numpy 数组，最小值的秩为 1，并列值取平均秩。

    说明:
        不引入 scipy：只用 numpy 的稳定排序，并列段整体赋同一个平均秩。
    """
    np = require("numpy")

    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="stable")
    ordered = arr[order]

    ranks = np.empty(arr.size, dtype=float)
    start = 0
    while start < arr.size:
        stop = start
        while stop + 1 < arr.size and ordered[stop + 1] == ordered[start]:
            stop += 1
        # 秩从 1 起算；并列区间 [start, stop] 的平均秩为 (start+stop)/2 + 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


def top_two(members: list[Any], degrees: dict[Any, int]) -> tuple[Any, Any]:
    """返回超边内超度最大与第二大的两个结点。

    参数:
        members: 超边成员列表（至少 2 个结点）。
        degrees: ``{结点: 超度}``。

    返回:
        ``(超度最大结点, 超度第二大结点)``；超度并列时取结点编号较小者
        （结点类型不可排序时退化为按 ``repr`` 比较）。
    """
    try:
        ordered = sorted(members, key=lambda node: (-degrees.get(node, 0), node))
    except TypeError:
        # 混合类型结点无法直接比较大小，退回字符串表示保证可排序
        ordered = sorted(members, key=lambda node: (-degrees.get(node, 0), repr(node)))
    return ordered[0], ordered[1]


def assortativity_rank_correlation(H: Any) -> float:
    """计算单个超图的同配系数（超度最大 / 第二大结点的秩相关）。

    参数:
        H: hypernetx 超图对象。

    返回:
        皮尔逊相关系数 ``float``；结点少于 2 个、成员数 ≥ 2 的超边少于 2 条，
        或任一侧秩序列方差为 0 时返回 ``0.0``（与旧实现约定一致）。
    """
    np = require("numpy")

    top = topology(H)
    nodes = list(top.nodes)
    if len(nodes) < 2:
        return 0.0

    degrees = top.degrees
    # 平均秩：超度相同的结点共享同一秩，避免人为引入顺序偏好
    rank_values = average_ranks([float(degrees.get(node, 0)) for node in nodes])
    ranks = {node: float(rank) for node, rank in zip(nodes, rank_values, strict=True)}

    xs: list[float] = []
    ys: list[float] = []
    for members in top.members.values():
        if len(members) < 2:
            # 单结点超边没有「第二大结点」，跳过
            continue
        top_node, second = top_two(list(members), degrees)
        xs.append(ranks[top_node])
        ys.append(ranks[second])

    if len(xs) < 2:
        return 0.0
    if np.std(xs, ddof=0) == 0 or np.std(ys, ddof=0) == 0:
        # 分母为 0：相关系数无定义，约定取 0
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])
