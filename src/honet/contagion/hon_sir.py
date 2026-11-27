"""SIR 模型统一入口（按方法枚举分派）。"""

from __future__ import annotations

from typing import Any

from honet._deps import require
from honet.contagion._common import choose_initial_infecteds, seed_everything
from honet.contagion._discrete import simulate_discrete
from honet.contagion._random_models import simulate_random_discrete, simulate_random_gillespie
from honet.contagion._strategies import resolve_strategy
from honet.contagion.contagion_method import Contagion_Method


def hon_sir(
    H: Any,
    gamma: float,
    tau: dict[int, float],
    method: Contagion_Method = Contagion_Method.Discrete,
    initial_infection: Any = None,
    initial_recovereds: Any = None,
    strategy: str = "collective",
    tmax: float = 1000,
    seed: int | None = 42,
    return_all_data: bool = False,
) -> Any:
    """SIR 模型统一入口（旧名 ``honSIR``）。

    按 ``method`` 分派到四种算法：

    * ``Discrete`` —— hypernetx 的 ``discrete_SIR``；给了 ``initial_recovereds``
      时改走 ``discrete_sir_control`` 的语义；
    * ``Gillespie`` —— hypernetx 的 ``Gillespie_SIR``；
    * ``RandomDiscrete`` —— ``random_discrete_sir`` 的语义；
    * ``RandomGillespie`` —— ``random_gillespie_sir`` 的语义。

    参数:
        H: hypernetx 超图对象。
        gamma: 恢复率。
        tau: ``{超边大小: 感染率}``。
        method: 算法选择，见 :class:`Contagion_Method`。
        initial_infection: 初始感染者列表；``None`` 时按 ``rho=0.1`` 随机抽。
        initial_recovereds: 受控（强制康复）结点。
        strategy: 传播策略名。
        tmax: 模拟终止时间。
        seed: 随机种子；``None`` 表示不播种。
        return_all_data: 是否返回完整迁移数据。

    返回:
        ``(times, S, I, R)`` 四个 numpy 数组，或完整迁移数据。

    抛出:
        ValueError: ``method`` 不是 :class:`Contagion_Method` 的成员。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> t, S, I, R = hon_sir(H, 0.3, {2: 0.4}, tmax=5, seed=1)
        >>> len(t) == len(R)
        True
    """
    seed_everything(seed)
    dt = 1  # 离散时间步长

    if method == Contagion_Method.Discrete:
        if initial_recovereds is not None:
            # 与 discrete_sir_control 一致：受控结点被强制转为 R
            return simulate_discrete(
                H,
                tau,
                gamma,
                model="SIR",
                mode="control",
                transmission_function=resolve_strategy(strategy),
                initial_infecteds=initial_infection,
                initial_recovereds=initial_recovereds,
                tmin=0,
                tmax=tmax,
                dt=dt,
                return_full_data=return_all_data,
            )

        function = require("hypernetx", attr="discrete_SIR")
        kwargs: dict[str, Any] = {
            "transmission_function": resolve_strategy(strategy),
            "tmin": 0,
            "tmax": tmax,
            "dt": dt,
            "return_full_data": return_all_data,
        }
        # 初始感染者统一走 choose_initial_infecteds，语义仍是「按 rho=0.1 抽样」，
        # 但保证小图上至少 1 个，避免上游对空初始状态除零。
        if initial_infection is None:
            initial_infection = choose_initial_infecteds(H, None, 0.1)
        return function(H, tau, gamma, initial_infecteds=initial_infection, **kwargs)

    if method == Contagion_Method.Gillespie:
        function = require("hypernetx", attr="Gillespie_SIR")
        gillespie_kwargs: dict[str, Any] = {
            "transmission_function": resolve_strategy(strategy),
            "tmax": tmax,
        }
        if initial_infection is None:
            initial_infection = choose_initial_infecteds(H, None, 0.1)
        return function(H, tau, gamma, initial_infecteds=initial_infection, **gillespie_kwargs)

    if method == Contagion_Method.RandomDiscrete:
        return simulate_random_discrete(H, tau, gamma, model="SIR", rho=0.1, tmax=tmax, dt=dt)

    if method == Contagion_Method.RandomGillespie:
        return simulate_random_gillespie(H, tau, gamma, model="SIR", rho=0.1, tmax=tmax)

    msg = f"未知的传播方法：{method!r}"
    raise ValueError(msg)
