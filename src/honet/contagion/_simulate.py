"""仓室传播模型的统一入口（按方法枚举分派）。

本模块把旧实现里 ``hon_sis`` / ``hon_sir`` 两份 95% 相同的分派逻辑合并成
一份，由 :mod:`honet.contagion._model` 的仓室规格驱动。因此：

* ``hon_sis`` / ``hon_sir`` 只是 :func:`simulate` 的薄封装（保持旧签名与旧行为）；
* 新增模型（SIRS / SIER / SEIR / SI，或自定义规格）不需要写新入口，
  直接 ``simulate(H, ..., model="SIER")`` 即可。

分派规则与旧实现一致：

* ``Discrete`` —— hypernetx 的 ``discrete_SIS`` / ``discrete_SIR``；
* ``Gillespie`` —— hypernetx 的 ``Gillespie_SIS`` / ``Gillespie_SIR``；
* ``RandomDiscrete`` / ``RandomGillespie`` —— 本包的带噪声随机模型。

控制分支（``control_nodes`` / ``initial_recovereds``）只在 ``Discrete`` 下生效，
与旧实现一致。
"""

from __future__ import annotations

import inspect
from typing import Any

from honet._deps import require
from honet.contagion._common import choose_initial_infecteds, seed_everything
from honet.contagion._discrete import simulate_discrete
from honet.contagion._model import CompartmentModel, resolve_model
from honet.contagion._random_models import simulate_random_discrete, simulate_random_gillespie
from honet.contagion._strategies import resolve_strategy
from honet.contagion.contagion_method import Contagion_Method

#: hypernetx 提供的模型入口名，按 (模型名, 方法) 索引
_HNX_FUNCTIONS = {
    ("SIS", Contagion_Method.Discrete): "discrete_SIS",
    ("SIS", Contagion_Method.Gillespie): "Gillespie_SIS",
    ("SIR", Contagion_Method.Discrete): "discrete_SIR",
    ("SIR", Contagion_Method.Gillespie): "Gillespie_SIR",
}


def simulate(
    H: Any,
    gamma: float,
    tau: dict[int, float],
    model: str | CompartmentModel = "SIS",
    method: Contagion_Method = Contagion_Method.Discrete,
    initial_infection: Any = None,
    initial_recovereds: Any = None,
    initial_exposeds: Any = None,
    strategy: str = "collective",
    tmax: float = 1000,
    seed: int | None = 42,
    return_all_data: bool = False,
    control_nodes: Any = None,
    rng: Any = None,
) -> Any:
    """模拟任意已注册仓室模型在超图上的传播过程。

    参数:
        H: hypernetx 超图对象。
        gamma: 恢复率（每个感染态结点每单位时间恢复的概率）。
        tau: ``{超边大小: 感染率}``。
        model: 模型名或 :class:`~honet.contagion._model.CompartmentModel` 规格。
            内置 ``"SIS"`` / ``"SIR"`` / ``"SIRS"`` / ``"SIER"`` / ``"SEIR"`` / ``"SI"``。
            注意 ``Gillespie`` 方法只支持 hypernetx 内置的 ``SIS`` / ``SIR``。
        method: 算法选择，见 :class:`~honet.contagion.Contagion_Method`。
        initial_infection: 初始感染者；``None`` 时按 ``rho=0.1`` 随机抽。
        initial_recovereds: 受控（强制离开感染态）的结点。
        initial_exposeds: 初始暴露者，仅含 ``E`` 仓室的模型使用。
        strategy: 传播策略名，见 :mod:`honet.contagion._strategies`。
        tmax: 模拟终止时间。
        seed: 随机种子；``None`` 表示不播种。
        return_all_data: 是否返回完整迁移数据。
        control_nodes: 控制结点；仅 ``Discrete`` 方法使用，等价于
            ``initial_recovereds``（保留旧参数名）。
        rng: 显式随机源。传入 ``random.Random``（离散引擎）或
            ``numpy.random.Generator``（随机模型），可避免读写全局随机状态，
            从而并行/嵌套运行互不干扰的模拟。为 ``None`` 时沿用 ``seed``
            设置的全局状态（旧行为）。

    返回:
        ``(times, *各仓室序列)``。``"SIS"`` 返回三个序列，``"SIR"`` / ``"SIER"``
        返回四个，序列顺序由模型规格的 ``output_states()`` 决定。

    抛出:
        ValueError: ``method`` 非法，或该模型不支持所选的 ``method``。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> t, S, I = simulate(H, 0.3, {2: 0.4}, model="SIS", tmax=5, seed=1)
        >>> len(t) == len(I)
        True
        >>> result = simulate(H, 0.3, {2: 0.4}, model="SIER", tmax=5, seed=1)
        >>> len(result)  # times + S/E/I/R
        5
    """
    seed_everything(seed)
    dt = 1  # 离散时间步长
    spec = resolve_model(model)
    controlled = control_nodes if control_nodes is not None else initial_recovereds

    if method == Contagion_Method.Discrete:
        if controlled is not None or initial_exposeds is not None:
            # 有控制目标或暴露者时走本包的引擎：hypernetx 没有这些分支
            return simulate_discrete(
                H,
                tau,
                gamma,
                model=spec,
                mode="pulse" if control_nodes is not None else "control",
                transmission_function=resolve_strategy(strategy),
                initial_infecteds=initial_infection,
                initial_recovereds=controlled,
                initial_exposeds=initial_exposeds,
                tmin=0,
                tmax=tmax,
                dt=dt,
                return_full_data=return_all_data,
                rng=rng,
            )

        function_name = _HNX_FUNCTIONS.get((spec.name, method))
        if function_name is None:
            # hypernetx 只内置 SIS / SIR；其它模型统一走本包引擎
            return simulate_discrete(
                H,
                tau,
                gamma,
                model=spec,
                mode="none",
                transmission_function=resolve_strategy(strategy),
                initial_infecteds=initial_infection,
                initial_exposeds=initial_exposeds,
                rho=0.1 if initial_infection is None else None,
                tmin=0,
                tmax=tmax,
                dt=dt,
                return_full_data=return_all_data,
                rng=rng,
            )

        function = require("hypernetx", attr=function_name)
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
        function_name = _HNX_FUNCTIONS.get((spec.name, method))
        if function_name is None:
            msg = (
                f"模型 {spec.name!r} 不支持 Gillespie 方法；"
                f"hypernetx 只内置 SIS / SIR 的连续时间实现，"
                f"请改用 Contagion_Method.Discrete 或 RandomGillespie"
            )
            raise ValueError(msg)
        function = require("hypernetx", attr=function_name)
        gillespie_kwargs: dict[str, Any] = {
            "transmission_function": resolve_strategy(strategy),
            "tmax": tmax,
        }
        # hypernetx 2.4.3 的 Gillespie_SIR 签名里没有 return_full_data
        # （只有 Gillespie_SIS 有），多传会抛 TypeError，因此按需附加。
        if "return_full_data" in inspect.signature(function).parameters:
            gillespie_kwargs["return_full_data"] = return_all_data
        if initial_infection is None:
            initial_infection = choose_initial_infecteds(H, None, 0.1)
        return function(H, tau, gamma, initial_infecteds=initial_infection, **gillespie_kwargs)

    if method == Contagion_Method.RandomDiscrete:
        return simulate_random_discrete(
            H, tau, gamma, model=spec, rho=0.1, tmax=tmax, dt=dt, rng=rng
        )

    if method == Contagion_Method.RandomGillespie:
        return simulate_random_gillespie(H, tau, gamma, model=spec, rho=0.1, tmax=tmax, rng=rng)

    msg = f"未知的传播方法：{method!r}"
    raise ValueError(msg)
