"""随机（带噪声）传播模型的内部引擎（私有，不对外导出）。

旧库的 ``RandomDiscrete_SIS`` / ``RandomGillespie_SIS`` / ``RandomGillespie_SIR``
共用同一套「平均场 + 噪声」骨架：用平均感染率 ``beta``、结点度数与当前
感染密度 ``theta`` 相乘得到感染率，再叠加一个正态噪声。本模块把这套骨架
参数化，四个公开函数各自调用。

与旧实现的差异（有意为之，已在对应公开函数的 docstring 中注明）：

* 旧实现用 ``H.neighbors(node)`` 取邻居，2.4.3 仍可用但属于待移除接口，
  这里统一走共享拓扑索引 ``honet._topology.topology`` 的邻居表；
* 旧 ``RandomGillespie_SIS`` 在选结点时用的是**无噪声**速率，与事件总数
  （噪声速率之和）口径不一致，概率和不为 1 时靠二次归一化掩盖；这里改用
  与总数一致的噪声速率；
* 旧 ``RandomDiscrete_SIR`` 只有函数签名、函数体为空（返回 ``None``），
  这里按 SIS 版本补齐为真正的 SIR 过程。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require
from honet._topology import topology
from honet.contagion._common import choose_initial_infecteds
from honet.contagion._model import SIS, CompartmentModel, resolve_model


def _resolve(model: Any) -> CompartmentModel:
    """把 ``model`` 参数解析成仓室规格（``None`` 按 SIS 处理）。"""
    return SIS if model is None else resolve_model(model)


def _initial_state(
    H: Any,
    rho: float | None,
    initial_infecteds: Any,
    spec: CompartmentModel | None = None,
) -> tuple[list[Any], dict[Any, str]]:
    """按 ``rho`` 或显式列表确定初始状态（保证小图上也有感染者）。"""
    spec = spec or SIS
    nodes = list(H.nodes)
    infected = set(choose_initial_infecteds(H, initial_infecteds, rho))
    infected_state = spec.infected[0]
    return nodes, {
        node: (infected_state if node in infected else spec.susceptible) for node in nodes
    }


def _degrees(H: Any) -> dict[Any, int]:
    """返回 ``{结点: 成对邻居数}``（旧实现用邻居个数，不是超边度）。

    邻居表来自共享拓扑索引，多个模型共用一次构建结果，避免每个结点都
    重新遍历所属超边。
    """
    top = topology(H)
    return {node: len(top.neighbors_of(node)) for node in top.nodes}


def _theta(nodes: list[Any], degrees: dict[Any, int], state: dict[Any, str]) -> float:
    """感染密度加权量：感染者度数之和 / 全体度数之和。"""
    denominator = sum(degrees.values())
    if denominator <= 0:
        return 0.0
    numerator = sum(degrees[node] for node in nodes if state[node] == "I")
    return numerator / denominator


def simulate_random_discrete(
    H: Any,
    tau: dict[int, float],
    gamma: float,
    *,
    model: str = "SIS",
    sigma1: float = 0.1,
    sigma2: float = 0.1,
    rho: float | None = 0.1,
    initial_infecteds: Any = None,
    tmin: float = 0,
    tmax: float = float("Inf"),
    dt: float = 1.0,
    **args: Any,
) -> tuple[list[float], list[int], list[int]] | tuple[list[float], list[int], list[int], list[int]]:
    """带噪声的离散时间 SIS / SIR 模拟。

    参数:
        H: hypernetx 超图对象。
        tau: ``{超边大小: 感染率}``，只用于取平均值 ``beta``。
        gamma: 恢复率。
        model: ``"SIS"`` 或 ``"SIR"``。
        sigma1: 感染率噪声强度。
        sigma2: 恢复率噪声强度。
        rho: 初始感染比例。
        initial_infecteds: 显式初始感染者；与 ``rho`` 互斥。
        tmin: 起始时间。
        tmax: 终止时间（含端点，旧实现用 ``<=``）。
        dt: 时间步长。
        **args: 兼容旧签名，未使用。

    返回:
        ``(t, S, I)``（SIS）或 ``(t, S, I, R)``（SIR），均为普通 Python 列表。
    """
    np = require("numpy")
    spec = _resolve(model)

    beta = float(np.mean(list(tau.values())))
    degrees = _degrees(H)
    output_states = spec.output_states()

    nodes, state = _initial_state(H, rho, initial_infecteds, spec)

    t: list[float] = []
    series: dict[str, list[int]] = {name: [] for name in output_states}

    current_time = tmin
    while current_time <= tmax:
        t.append(current_time)
        for name in output_states:
            series[name].append(sum(1 for node in nodes if state[node] == name))

        theta = _theta(nodes, degrees, state)
        new_state = state.copy()

        for node in nodes:
            current = state[node]
            # 被感染：仅对可被感染的状态成立（SIER 里 S 与 E 都可被感染）
            if spec.can_be_infected(current):
                noise = np.random.normal(0, sigma1 * degrees[node] * theta)
                infection_prob = max(0.0, min(1.0, beta * degrees[node] * theta + noise))
                if np.random.random() < infection_prob * dt:
                    new_state[node] = spec.on_infection[current]
            # 恢复 / 离开当前仓室
            elif spec.can_recover(current):
                noise = np.random.normal(0, sigma2)
                recovery_prob = max(0.0, min(1.0, gamma + noise))
                if np.random.random() < recovery_prob * dt:
                    new_state[node] = spec.on_recovery[current]

        state = new_state
        current_time += dt

    return (t, *(series[name] for name in output_states))


def simulate_random_gillespie(
    H: Any,
    tau: dict[int, float],
    gamma: float,
    *,
    model: str = "SIS",
    sigma1: float = 0.1,
    sigma2: float = 0.1,
    rho: float | None = 0.1,
    initial_infecteds: Any = None,
    tmin: float = 0,
    tmax: float = float("Inf"),
    **args: Any,
) -> tuple[list[float], list[int], list[int]] | tuple[list[float], list[int], list[int], list[int]]:
    """带噪声的 Gillespie（连续时间）SIS / SIR 模拟。

    事件等待时间服从指数分布，事件类型在「感染 / 恢复」之间按速率比例抽取。

    参数:
        H: hypernetx 超图对象。
        tau: ``{超边大小: 感染率}``，只用于取平均值 ``beta``。
        gamma: 恢复率。
        model: ``"SIS"`` 或 ``"SIR"``。
        sigma1: 感染率噪声强度。
        sigma2: 恢复率噪声强度。
        rho: 初始感染比例。
        initial_infecteds: 显式初始感染者；与 ``rho`` 互斥。
        tmin: 起始时间。
        tmax: 终止时间（不含端点）。
        **args: 兼容旧签名，未使用。

    返回:
        ``(t, S, I)``（SIS）或 ``(t, S, I, R)``（SIR），均为普通 Python 列表。
    """
    np = require("numpy")
    spec = _resolve(model)

    beta = float(np.mean(list(tau.values())))
    degrees = _degrees(H)
    output_states = spec.output_states()

    nodes, state = _initial_state(H, rho, initial_infecteds, spec)

    def counts() -> dict[str, int]:
        return {name: sum(1 for node in nodes if state[node] == name) for name in output_states}

    t: list[float] = [tmin]
    initial_counts = counts()
    series: dict[str, list[int]] = {name: [initial_counts[name]] for name in output_states}

    current_time = tmin
    while current_time < tmax:
        theta = _theta(nodes, degrees, state)

        # 收集两类事件的速率（都带噪声，且与事件总数口径一致）
        susceptible_rates: list[tuple[Any, str]] = []
        recovery_rates: list[tuple[Any, str]] = []
        for node in nodes:
            current = state[node]
            if spec.can_be_infected(current):
                noise = np.random.normal(0, sigma1 * degrees[node] * theta)
                rate = max(0.0, beta * degrees[node] * theta + noise)
                susceptible_rates.append((node, rate))
            elif spec.can_recover(current):
                noise = np.random.normal(0, sigma2)
                recovery_rates.append((node, max(0.0, gamma + noise)))

        total_infection_rate = sum(rate for _, rate in susceptible_rates)
        total_recovery_rate = sum(rate for _, rate in recovery_rates)
        total_rate = total_infection_rate + total_recovery_rate

        if total_rate <= 0:
            break

        current_time += float(np.random.exponential(1 / total_rate))
        if current_time > tmax:
            break

        probabilities = [total_infection_rate / total_rate, total_recovery_rate / total_rate]
        event_type = np.random.choice(["infection", "recovery"], p=probabilities)

        if event_type == "infection" and total_infection_rate > 0:
            weights = [rate / total_infection_rate for _, rate in susceptible_rates]
            total_weight = sum(weights)
            if total_weight > 0:
                node_probs = [weight / total_weight for weight in weights]
                selected = np.random.choice([n for n, _ in susceptible_rates], p=node_probs)
                state[selected] = spec.on_infection[state[selected]]
        elif recovery_rates:
            selected = np.random.choice([n for n, _ in recovery_rates])
            state[selected] = spec.on_recovery[state[selected]]

        step_counts = counts()
        t.append(current_time)
        for name in output_states:
            series[name].append(step_counts[name])

    return (t, *(series[name] for name in output_states))
