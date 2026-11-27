"""离散时间传播模型的内部引擎（私有，不对外导出）。

旧库的 ``discrete_SIS_control`` / ``discrete_SIS_control_pulse`` /
``discrete_SIR_control`` 是三段几乎逐行重复的循环，本模块把它们合并成一个
参数化的引擎，由三个公开函数按旧语义分别调用，避免逻辑分叉后互相漂移。

与旧实现保持一致的几处「怪癖」（不要顺手「修好」，否则结果对不上）：

* 恢复判定先消耗一个 ``random.random()``；脉冲控制在恢复失败之后才判断
  阈值，随机数的消耗顺序因此固定；
* 易感者逐条遍历所属超边，只要有一条超边触发感染就 ``break``，全部未触发时
  走 ``for/else`` 分支把状态写回 ``"S"``；
* ``*_control`` 系列里的 ``initial_recovereds`` 不是「初始 R 态结点」，而是
  「受控结点」：SIS 里它们被强制转为 ``S``，SIR 里被强制转为 ``R"``；
* 每个时间步的状态迁移都基于「上一步的 ``status``」逐结点判定，写进
  ``newStatus``，步末整体替换（同步更新，不是异步就地更新）。
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from honet._deps import require

#: SIS 的三种控制模式
SIS_MODES = ("control", "pulse")
#: SIR 只有一种控制模式
SIR_MODES = ("control",)


def simulate_discrete(
    H: Any,
    tau: dict[int, float],
    gamma: float,
    *,
    model: str = "SIS",
    mode: str = "control",
    transmission_function: Any = None,
    initial_infecteds: Any = None,
    initial_recovereds: Any = None,
    rho: float | None = None,
    tmin: float = 0,
    tmax: float = float("Inf"),
    dt: float = 1.0,
    return_full_data: bool = False,
    **args: Any,
) -> Any:
    """离散时间 SIS / SIR 模拟的统一引擎。

    参数:
        H: hypernetx 超图对象。
        tau: ``{超边大小: 感染率}``，必须覆盖图中出现的全部超边大小。
        gamma: 恢复率。
        model: ``"SIS"`` 或 ``"SIR"``。
        mode: SIS 取 ``"control"`` / ``"pulse"``；SIR 只有 ``"control"``。
        transmission_function: 传播函数 ``f(node, status, members, **args) -> bool``。
        initial_infecteds: 初始感染者；与 ``rho`` 互斥。
        initial_recovereds: 受控结点（见模块 docstring 的说明）。
        rho: 初始感染比例；与 ``initial_infecteds`` 互斥。
        tmin: 模拟起始时间。
        tmax: 模拟终止时间。
        dt: 时间步长。
        return_full_data: 为 ``True`` 时返回 ``{时刻: 迁移事件列表}``。
        **args: 透传给 ``transmission_function`` 的额外参数。

    返回:
        ``return_full_data`` 为真时返回迁移事件字典，否则返回
        ``(times, S, I)``（SIS）或 ``(times, S, I, R)``（SIR），均为 numpy 数组。

    抛出:
        ValueError: ``model`` / ``mode`` 非法，或 ``rho`` 与 ``initial_infecteds`` 同时给出。
    """
    from honet.contagion._common import choose_initial_infecteds, normalize_nodes

    np = require("numpy")

    if model not in ("SIS", "SIR"):
        msg = f"未知的 model：{model!r}，只支持 'SIS' 或 'SIR'"
        raise ValueError(msg)
    if model == "SIS" and mode not in SIS_MODES:
        msg = f"未知的 mode：{mode!r}，SIS 只支持 {SIS_MODES}"
        raise ValueError(msg)
    if model == "SIR" and mode not in SIR_MODES:
        msg = f"未知的 mode：{mode!r}，SIR 只支持 {SIR_MODES}"
        raise ValueError(msg)

    if transmission_function is None:
        transmission_function = require("hypernetx", attr="threshold")

    infecteds = choose_initial_infecteds(H, initial_infecteds, rho)
    recovereds = normalize_nodes(H, initial_recovereds)
    is_sir = model == "SIR"

    status: dict[Any, str] = defaultdict(lambda: "S")
    transition_events: dict[float, list[tuple]] = {tmin: []}
    for node in infecteds:
        status[node] = "I"
        if return_full_data:
            transition_events[tmin].append(("I", node, None))
    if is_sir:
        for node in recovereds:
            status[node] = "R"
            if return_full_data:
                transition_events[tmin].append(("R", node))

    total = len(H.nodes)
    n_infected = [len(infecteds)]
    n_recovered = [len(recovereds)] if is_sir else []
    n_susceptible = [total - n_infected[-1] - (n_recovered[-1] if is_sir else 0)]

    t = tmin
    times = [t]
    new_status = status.copy()

    def over_threshold(step: float) -> bool:
        """脉冲控制的切换条件：感染比例超过阈值，或已走过 1/4 的时间。"""
        ratio_limit = max((rho or 0.0) * 2, 0.5)
        return n_infected[-1] / total > ratio_limit or step > tmax / 4

    while t < tmax and n_infected[-1] != 0:
        n_susceptible.append(n_susceptible[-1])
        n_infected.append(n_infected[-1])
        if is_sir:
            n_recovered.append(n_recovered[-1])
        if return_full_data:
            transition_events[t + dt] = []

        for node in H.nodes:
            if status[node] == "I":
                # 只消耗一次随机数：先判自然恢复，失败后再看是否被强制恢复
                recovered = random.random() <= gamma * dt
                if not recovered and node in recovereds:
                    # 控制模式：受控结点无条件恢复；脉冲模式：仅在受控期内恢复
                    recovered = mode == "control" or over_threshold(t)
                if recovered:
                    new_status[node] = "R" if is_sir else "S"
                    n_infected[-1] -= 1
                    if is_sir:
                        n_recovered[-1] += 1
                    else:
                        n_susceptible[-1] += 1
                    if return_full_data:
                        transition_events[t + dt].append(("R" if is_sir else "S", node))
            elif status[node] == "S":
                for edge_id in H.nodes.memberships[node]:
                    members = H.edges[edge_id]
                    # 注意：传的是「上一步的 status」，不是本步正在改的 new_status
                    infected_here = transmission_function(node, status, members, **args)
                    if random.random() <= tau[len(members)] * infected_here * dt:
                        new_status[node] = "I"
                        n_susceptible[-1] -= 1
                        n_infected[-1] += 1
                        if return_full_data:
                            transition_events[t + dt].append(("I", node, edge_id))
                        break
                else:
                    new_status[node] = "S"

        status = new_status.copy()
        t += dt
        times.append(t)

    if return_full_data:
        return transition_events
    if is_sir:
        return (
            np.array(times),
            np.array(n_susceptible),
            np.array(n_infected),
            np.array(n_recovered),
        )
    return np.array(times), np.array(n_susceptible), np.array(n_infected)
