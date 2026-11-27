"""离散时间传播模型的内部引擎（私有，不对外导出）。

引擎只认**仓室规格**（见 :mod:`honet.contagion._model`），不再用
``is_sir`` 之类的布尔量分叉。因此 SIS / SIR / SIRS / SIER 等模型共用同一份
循环，新增模型只需注册规格，不需要改动本文件。

与旧实现保持一致的几处「怪癖」（不要顺手「修好」，否则结果对不上）：

* 恢复判定先消耗一个 ``random.random()``；脉冲控制在恢复失败之后才判断
  阈值，随机数的消耗顺序因此固定；
* 易感者逐条遍历所属超边，只要有一条超边触发感染就 ``break``，全部未触发时
  走 ``for/else`` 分支把状态写回易感态；
* ``*_control`` 系列里的 ``initial_recovereds`` 不是「初始 R 态结点」，而是
  「受控结点」：SIS 里它们被强制转回 ``S``，SIR 里被强制转为 ``R``；
* 每个时间步的状态迁移都基于「上一步的 ``status``」逐结点判定，写进
  ``new_status``，步末整体替换（同步更新，不是异步就地更新）。
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from honet._deps import require
from honet.contagion._model import CompartmentModel, resolve_model

#: 旧代码按模型名判断可用的控制模式，这里保留以便对外报错信息一致
SIS_MODES = ("control", "pulse")
SIR_MODES = ("control",)

#: 所有模型都支持的控制模式（规格驱动后不再按模型名硬编码）
ALL_MODES = ("control", "pulse", "none")


def simulate_discrete(
    H: Any,
    tau: dict[int, float],
    gamma: float,
    *,
    model: str | CompartmentModel = "SIS",
    mode: str = "control",
    transmission_function: Any = None,
    initial_infecteds: Any = None,
    initial_recovereds: Any = None,
    initial_exposeds: Any = None,
    rho: float | None = None,
    tmin: float = 0,
    tmax: float = float("Inf"),
    dt: float = 1.0,
    return_full_data: bool = False,
    rng: Any = None,
    **args: Any,
) -> Any:
    """离散时间仓室模型模拟的统一引擎。

    参数:
        H: hypernetx 超图对象。
        tau: ``{超边大小: 感染率}``，必须覆盖图中出现的全部超边大小。
        gamma: 恢复率。
        model: 模型名或 :class:`~honet.contagion._model.CompartmentModel` 规格。
            内置 ``"SIS"`` / ``"SIR"`` / ``"SIRS"`` / ``"SIER"`` / ``"SEIR"`` / ``"SI"``。
        mode: 控制模式，``"control"`` 无条件控制、``"pulse"`` 脉冲控制、
            ``"none"`` 不做控制。
        transmission_function: 传播函数 ``f(node, status, members, **args) -> bool``。
        initial_infecteds: 初始感染者；与 ``rho`` 互斥。
        initial_recovereds: 受控结点（见模块 docstring 的说明）。
        initial_exposeds: 初始暴露者，仅含 ``E`` 仓室的模型使用。
        rho: 初始感染比例；与 ``initial_infecteds`` 互斥。
        tmin: 模拟起始时间。
        tmax: 模拟终止时间。
        dt: 时间步长。
        return_full_data: 为 ``True`` 时返回 ``{时刻: 迁移事件列表}``。
        rng: 显式随机源（``random.Random`` 实例）。为 ``None`` 时沿用全局
            ``random`` 状态（与旧行为一致）；给出时所有随机判定都走该实例，
            因此可以并行/嵌套运行互不干扰的模拟。
        **args: 透传给 ``transmission_function`` 的额外参数。

    返回:
        ``return_full_data`` 为真时返回迁移事件字典，否则返回
        ``(times, *各仓室序列)``，序列顺序由规格的 ``output_states()`` 决定。
        ``"SIS"`` 返回 ``(times, S, I)``，``"SIR"`` 返回 ``(times, S, I, R)``。

    抛出:
        ValueError: ``model`` 未注册，或 ``mode`` 非法，或 ``rho`` 与
            ``initial_infecteds`` 同时给出。
    """
    from honet.contagion._common import choose_initial_infecteds, normalize_nodes

    np = require("numpy")
    spec = resolve_model(model)

    # 显式随机源优先；未给出时退回全局 random（保持旧行为与旧结果）
    _random = rng.random if rng is not None else random.random

    if mode not in ALL_MODES:
        msg = f"未知的 mode：{mode!r}，只支持 {ALL_MODES}"
        raise ValueError(msg)

    if transmission_function is None:
        transmission_function = require("hypernetx", attr="threshold")

    infecteds = choose_initial_infecteds(H, initial_infecteds, rho, rng=rng)
    recovereds = normalize_nodes(H, initial_recovereds)
    exposeds = normalize_nodes(H, initial_exposeds)

    status: dict[Any, str] = defaultdict(lambda: spec.susceptible)
    transition_events: dict[float, list[tuple]] = {tmin: []}

    # 初始状态：显式给定的按仓室放置，其余留在易感态
    for node in infecteds:
        status[node] = spec.infected[0]
        if return_full_data:
            transition_events[tmin].append((spec.infected[0], node, None))
    for node in exposeds:
        status[node] = "E"
        if return_full_data:
            transition_events[tmin].append(("E", node, None))
    # 受控结点只在存在吸收态（SIR 型）时才改写初始状态。旧实现在 SIS 里
    # 完全不碰 status，只把 recovereds 当作「受控名单」留到循环里判断，
    # 因此这里不能把 SIS 的受控结点预先置成 S（否则会覆盖同名的初始感染者）。
    controlled_state = _controlled_state(spec)
    if _absorbing_state(spec) is not None:
        for node in recovereds:
            status[node] = controlled_state
            if return_full_data:
                transition_events[tmin].append((controlled_state, node))

    total = len(H.nodes)
    output_states = spec.output_states()
    counts: dict[str, list[int]] = {state: [] for state in output_states}

    # 初始计数严格照抄旧实现的两个公式：
    #   n_infected    = len(infecteds)
    #   n_recovered   = len(recovereds)          （仅 SIR 型模型才有这一列）
    #   n_susceptible = total - n_infected - n_recovered
    # 关键点：受控结点只在「模型存在 R 仓室」时才从易感里扣掉。SIS 没有 R
    # 仓室，受控结点虽然 status 被置成 S，但计数上并没有单独占位——这是旧
    # 实现的既有口径，改成「按 status 重新统计」会让 pulse 轨迹对不上。
    infected_state = spec.infected[0]
    n_infected_0 = len(infecteds)
    n_exposed_0 = len(exposeds)
    n_recovered_0 = len(recovereds) if _absorbing_state(spec) is not None else 0

    counts[spec.susceptible].append(total - n_infected_0 - n_recovered_0)
    for state in output_states:
        if state == spec.susceptible:
            continue
        if state == infected_state:
            counts[state].append(n_infected_0)
        elif state == "E":
            counts[state].append(n_exposed_0)
        elif state == _absorbing_state(spec):
            counts[state].append(n_recovered_0)
        else:
            counts[state].append(0)

    t = tmin
    times = [t]
    new_status = status.copy()

    def over_threshold(step: float) -> bool:
        """脉冲控制的切换条件：感染比例超过阈值，或已走过 1/4 的时间。

        注意：这里必须读**计数快照** ``counts[infected][-1]``，不能实时扫描
        ``status``。旧实现读的是 ``n_infected[-1]``，而计数是在步内被就地增减
        的，因此同一时刻不同结点看到的阈值判定会随遍历顺序变化。改成实时扫描
        会让随机数消耗路径不同，轨迹随之偏移。
        """
        ratio_limit = max((rho or 0.0) * 2, 0.5)
        infected_now = counts[spec.infected[0]][-1]
        return infected_now / total > ratio_limit or step > tmax / 4

    def shift(node: Any, source: str, target: str) -> None:
        """把 ``node`` 从 ``source`` 仓室迁到 ``target``，并同步各仓室计数。"""
        new_status[node] = target
        if source in counts and counts[source]:
            counts[source][-1] -= 1
        if target in counts and counts[target]:
            counts[target][-1] += 1
        if return_full_data:
            transition_events[t + dt].append((target, node))

    while t < tmax and any(counts[state][-1] for state in spec.infected):
        # 先把上一步的计数抄到本步，后续在副本上做增减
        for state in output_states:
            counts[state].append(counts[state][-1])
        if return_full_data:
            transition_events[t + dt] = []

        for node in H.nodes:
            current = status[node]

            # 1) 恢复 / 离开当前仓室
            if spec.can_recover(current):
                recovered = _random() <= gamma * dt
                if not recovered and node in recovereds:
                    # 控制模式：受控结点无条件恢复；脉冲模式：仅在受控期内恢复
                    recovered = mode == "control" or over_threshold(t)
                if recovered:
                    shift(node, current, spec.on_recovery[current])
                    continue

            # 2) 被感染：仅对可被感染的状态（S，以及 SIER 的 E）成立
            if spec.can_be_infected(current):
                for edge_id in H.nodes.memberships[node]:
                    members = H.edges[edge_id]
                    # 注意：传的是「上一步的 status」，不是本步正在改的 new_status
                    infected_here = transmission_function(node, status, members, **args)
                    if _random() <= tau[len(members)] * infected_here * dt:
                        shift(node, current, spec.on_infection[current])
                        break
                else:
                    new_status[node] = current

        status = new_status.copy()
        t += dt
        times.append(t)

    if return_full_data:
        return transition_events
    return (np.array(times), *(np.array(counts[state]) for state in output_states))


def _absorbing_state(spec: CompartmentModel) -> str | None:
    """返回模型的吸收态（恢复后无法再回到易感的那个状态）。

    SIR 型模型返回 ``"R"``；SIS / SIRS 这类循环模型没有吸收态，返回 ``None``。
    """
    for target in spec.on_recovery.values():
        if not spec._can_reach_susceptible(target):
            return target
    return None


def _controlled_state(spec: CompartmentModel) -> str:
    """返回受控结点在 ``spec`` 下的状态。

    SIR 型模型（存在吸收态）里受控结点落到该吸收态（``R``）；循环型模型里
    受控结点退回易感态。这与旧实现「SIS 强制转 S、SIR 强制转 R」的语义一致。
    """
    absorbing = _absorbing_state(spec)
    return absorbing if absorbing is not None else spec.susceptible
