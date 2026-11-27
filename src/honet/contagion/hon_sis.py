"""SIS 模型统一入口（按方法枚举分派）。"""

from __future__ import annotations

from typing import Any

from honet._deps import require
from honet.contagion._common import choose_initial_infecteds, seed_everything
from honet.contagion._discrete import simulate_discrete
from honet.contagion._random_models import simulate_random_discrete, simulate_random_gillespie
from honet.contagion._strategies import resolve_strategy
from honet.contagion.contagion_method import Contagion_Method


def hon_sis(
    H: Any,
    gamma: float,
    tau: dict[int, float],
    method: Contagion_Method = Contagion_Method.Discrete,
    initial_infection: Any = None,
    strategy: str = "collective",
    tmax: float = 1000,
    seed: int | None = 42,
    return_all_data: bool = False,
    control_nodes: Any = None,
) -> Any:
    """SIS 模型统一入口（旧名 ``honSIS``）。

    按 ``method`` 分派到四种算法：

    * ``Discrete`` —— hypernetx 的 ``discrete_SIS``；给了 ``control_nodes``
      时改走 ``discrete_sis_control`` 的语义；
    * ``Gillespie`` —— hypernetx 的 ``Gillespie_SIS``；
    * ``RandomDiscrete`` —— ``random_discrete_sis`` 的语义；
    * ``RandomGillespie`` —— ``random_gillespie_sis`` 的语义。

    参数:
        H: hypernetx 超图对象。
        gamma: 恢复率。
        tau: ``{超边大小: 感染率}``。
        method: 算法选择，见 :class:`Contagion_Method`。
        initial_infection: 初始感染者列表；``None`` 时按 ``rho=0.1`` 随机抽。
        strategy: 传播策略名，见 :mod:`honet.contagion._strategies`。
        tmax: 模拟终止时间。
        seed: 随机种子；``None`` 表示不播种。
        return_all_data: 是否返回完整迁移数据。
        control_nodes: 控制结点；仅 ``Discrete`` 方法使用。

    返回:
        ``(times, S, I)`` 三个 numpy 数组，或完整迁移数据。

    抛出:
        ValueError: ``method`` 不是 :class:`Contagion_Method` 的成员。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> t, S, I = hon_sis(H, 0.3, {2: 0.4}, tmax=5, seed=1)
        >>> len(t) == len(I)
        True
    """
    seed_everything(seed)
    dt = 1  # 离散时间步长

    if method == Contagion_Method.Discrete:
        if control_nodes is not None:
            # 与 discrete_sis_control 的「非 target」分支一致：collective + 脉冲控制
            return simulate_discrete(
                H,
                tau,
                gamma,
                model="SIS",
                mode="pulse",
                transmission_function=require("hypernetx", attr="collective_contagion"),
                initial_recovereds=control_nodes,
                rho=0.1,
                tmin=0,
                tmax=tmax,
                dt=dt,
                return_full_data=return_all_data,
            )

        function = require("hypernetx", attr="discrete_SIS")
        kwargs: dict[str, Any] = {
            "transmission_function": resolve_strategy(strategy),
            "tmin": 0,
            "tmax": tmax,
            "dt": dt,
            "return_full_data": return_all_data,
        }
        # 初始感染者统一走 choose_initial_infecteds，避免 rho 在小图上算出 0 个
        # 结点、进而让上游 Gillespie 分支除零；语义仍是「按 rho=0.1 抽样」。
        if initial_infection is None:
            initial_infection = choose_initial_infecteds(H, None, 0.1)
        return function(H, tau, gamma, initial_infecteds=initial_infection, **kwargs)

    if method == Contagion_Method.Gillespie:
        function = require("hypernetx", attr="Gillespie_SIS")
        gillespie_kwargs: dict[str, Any] = {
            "transmission_function": resolve_strategy(strategy),
            "tmax": tmax,
            "return_full_data": return_all_data,
        }
        if initial_infection is None:
            initial_infection = choose_initial_infecteds(H, None, 0.1)
        return function(H, tau, gamma, initial_infecteds=initial_infection, **gillespie_kwargs)

    if method == Contagion_Method.RandomDiscrete:
        return simulate_random_discrete(H, tau, gamma, model="SIS", rho=0.1, tmax=tmax, dt=dt)

    if method == Contagion_Method.RandomGillespie:
        return simulate_random_gillespie(H, tau, gamma, model="SIS", rho=0.1, tmax=tmax)

    msg = f"未知的传播方法：{method!r}"
    raise ValueError(msg)
