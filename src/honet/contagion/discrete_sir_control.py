"""离散时间 SIR 模型（支持控制结点）。"""

from __future__ import annotations

from typing import Any

from honet.contagion._common import seed_everything
from honet.contagion._discrete import simulate_discrete


def discrete_sir_control(
    H: Any,
    tau: dict[int, float],
    gamma: float,
    transmission_function: Any = None,
    initial_infecteds: Any = None,
    initial_recovereds: Any = None,
    rho: float | None = None,
    tmin: float = 0,
    tmax: float = float("Inf"),
    dt: float = 1.0,
    return_full_data: bool = False,
    *,
    seed: int | None = None,
    **args: Any,
) -> Any:
    """离散时间 SIR 模型，支持控制结点（旧名 ``discrete_SIR_control``）。

    与旧实现一致：``initial_recovereds`` 里的结点一律被视为受控结点，每个
    时间步只要处于 ``I`` 态就会被强制转为 ``R``，而不是仅仅作为初始 ``R``。

    参数:
        H: hypernetx 超图对象。
        tau: ``{超边大小: 感染率}``。
        gamma: 恢复率。
        transmission_function: 传播函数或策略名；``None`` 表示 ``threshold``。
        initial_infecteds: 初始感染者；与 ``rho`` 互斥。
        initial_recovereds: 受控结点（被强制转为 ``R``）。
        rho: 初始感染比例。
        tmin: 起始时间。
        tmax: 终止时间。
        dt: 时间步长。
        return_full_data: 为 ``True`` 时返回迁移事件字典。
        seed: 随机种子；给定后可复现。
        **args: 透传给传播函数。

    返回:
        ``(times, S, I, R)`` 四个 numpy 数组，或 ``{时刻: 事件列表}``。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> t, S, I, R = discrete_sir_control(H, {2: 0.4}, 0.3, tmax=5, seed=3)
        >>> len(t) == len(R)
        True
    """
    from honet.contagion._strategies import resolve_strategy

    seed_everything(seed)
    return simulate_discrete(
        H,
        tau,
        gamma,
        model="SIR",
        mode="control",
        transmission_function=resolve_strategy(
            "threshold" if transmission_function is None else transmission_function
        ),
        initial_infecteds=initial_infecteds,
        initial_recovereds=initial_recovereds,
        rho=rho,
        tmin=tmin,
        tmax=tmax,
        dt=dt,
        return_full_data=return_full_data,
        **args,
    )
