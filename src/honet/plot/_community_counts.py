"""把传播事件字典折算成「每个社团的感染数曲线」（私有）。

旧库 ``hypertrade.draw`` 的 ``_sub_draw_sis_community`` /
``_sub_draw_sir_community`` 各写了一份几乎完全相同的状态机，只是 SIR 多识别
一个 ``'R'`` 事件。这里抽成一处，两个私有绘图函数共用，避免两份代码各自演化。

事件字典的形态来自 hypernetx 的 ``discrete_SIS`` / ``discrete_SIR``
（``return_full_data=True``）：``{时间: [(动作, 结点, 传播超边), ...]}``，
动作取 ``'I'`` / ``'S'`` / ``'R'``。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require

#: 会被状态机识别的动作
ACTIONS = frozenset({"I", "S", "R"})


def normalize_partition(entry: Any) -> list[list[Any]]:
    """把「单个社团」或「社团列表」统一成社团列表。

    旧库的 ``draw_sis_conmmunity`` 把 ``communities`` 整个传给子函数，
    于是「一个社团的结点列表」和「社团划分（列表的列表）」两种形态都会
    流到同一处。这里按首元素的类型判定，让两种输入都能正确解释：

    * 首元素是列表/元组/集合 → 视为「社团划分」，逐项转成结点列表；
    * 首元素是结点本身 → 视为「单个社团」，整体包一层。

    参数:
        entry: 社团划分或单个社团的结点序列。

    返回:
        社团列表，每个元素是一个结点列表。
    """
    items = list(entry)
    if not items:
        return []
    if isinstance(items[0], (list, tuple, set, frozenset)):
        return [list(members) for members in items]
    return [items]


def as_event_dict(result: Any) -> dict[Any, list[Any]]:
    """校验并返回事件字典（``{时间: [事件, ...]}``）。

    参数:
        result: ``hon_sis`` / ``hon_sir`` 传 ``return_all_data=True`` 时的返回
            值，即事件字典本身。

    返回:
        浅拷贝后的事件字典（调用方可以安全地按时间遍历）。

    抛出:
        RuntimeError: 传入的是 ``(t, S, I)`` / ``(t, S, I, R)`` 这样的数值统计
            元组，无法按结点类别拆分。旧实现同样报错，但 SIR 分支的解包写错
            （从三元组里取四个值），本实现统一在这里判定。
        TypeError: ``result`` 既不是字典也不是序列。
    """
    # 数值统计结果：hon_sis/hon_sir 不传 return_all_data 时的形态
    if isinstance(result, (list, tuple)):
        msg = (
            "传入的是数值统计序列（(t, S, I) 或 (t, S, I, R)），无法按社团拆分；"
            "请调用 hon_sis/hon_sir(..., return_all_data=True) 并把返回的事件字典传入本函数"
        )
        raise RuntimeError(msg)

    if not isinstance(result, dict):
        msg = f"result 必须是事件字典 {{时间: [事件, ...]}}，收到 {type(result).__name__}"
        raise TypeError(msg)

    return dict(result)


def community_infected_counts(
    event_dict: dict[Any, list[Any]],
    community: list[list[Any]],
) -> tuple[Any, dict[int, list[int]]]:
    """按社团统计每个时间步处于感染态（``'I'``）的结点数。

    参数:
        event_dict: ``{时间: [事件, ...]}``，事件形如 ``('I', 结点, 超边)``；
            只有前两项必读，第三项（传播超边）会被忽略。
        community: 社团划分，每个元素是该社团的结点列表；也接受单个社团的
            结点列表（见 :func:`normalize_partition`）。

    返回:
        ``(时间数组, {社团序号: 感染数列表})``。时间数组为升序排序后的键，
        与每个感染数列表等长。

    说明:
        结点的初始状态一律记为 ``'S'``，然后按时间顺序回放事件更新状态，
        因此某个时刻的计数等于「该时刻所有事件处理完之后」的快照——与旧实现
        一致（旧代码对每个时间点先应用事件再计数）。
    """
    np = require("numpy", extra="viz")

    community = normalize_partition(community)
    t_sorted = sorted(event_dict)
    t_arr = np.asarray(t_sorted)

    # 结点全集：优先取事件里出现过的，事件为空时退化为社团并集
    nodes: set[Any] = set()
    for events in event_dict.values():
        for event in events:
            if event and len(event) >= 2:
                nodes.add(event[1])
    if not nodes:
        for members in community:
            nodes.update(members)

    state = {node: "S" for node in nodes}
    cls_counts: dict[int, list[int]] = {index: [] for index in range(len(community))}

    for time in t_sorted:
        for event in event_dict.get(time, []):
            if not event:
                continue
            action = event[0]
            if len(event) < 2:
                continue
            node = event[1]
            if action in ACTIONS:
                state[node] = action

        for index, members in enumerate(community):
            cls_counts[index].append(sum(1 for node in members if state.get(node) == "I"))

    return t_arr, cls_counts


def plot_community_counts(
    ax: Any,
    t_arr: Any,
    cls_counts: dict[int, list[int]],
    *,
    only: int | None = None,
) -> None:
    """把社团感染曲线画到 ``ax`` 上（折线 + 图例）。

    参数:
        ax: 目标坐标轴。
        t_arr: 时间数组。
        cls_counts: ``{社团序号: 感染数列表}``。
        only: 只画某一个社团；``None`` 表示全部画（旧实现的行为）。
    """
    np = require("numpy", extra="viz")
    plt = require("matplotlib.pyplot", extra="viz")

    # 颜色取 matplotlib 默认色环，按社团序号轮转，与旧实现取色方式一致
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color") or [None]
    indices = [only] if only is not None else sorted(cls_counts)

    for index in indices:
        counts = cls_counts.get(index)
        if counts is None:
            continue
        ax.plot(
            t_arr,
            np.asarray(counts),
            "-",
            label=f"社团{index}",
            color=colors[index % len(colors)],
        )
