"""离散时间 SIS 模型（脉冲控制）。"""

from __future__ import annotations

from typing import Any

from honet.contagion._common import seed_everything
from honet.contagion._discrete import simulate_discrete


def discrete_sis_control_pulse(
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
    """离散时间 SIS 模型，使用脉冲控制（旧名 ``discrete_SIS_control_pulse``）。

    受控结点只有在「感染比例超过 ``max(2*rho, 0.5)``」或「已走过 ``tmax/4``
    时间」之后，才会被强制恢复；自然恢复概率 ``gamma * dt`` 始终先行判定。

    参数:
        H: hypernetx 超图对象。
        tau: ``{超边大小: 感染率}``。
        gamma: 恢复率。
        transmission_function: 传播函数或策略名；``None`` 表示 ``threshold``。
        initial_infecteds: 初始感染者；与 ``rho`` 互斥。
        initial_recovereds: 受控结点。
        rho: 初始感染比例（同时决定脉冲阈值）。
        tmin: 起始时间。
        tmax: 终止时间。
        dt: 时间步长。
        return_full_data: 为 ``True`` 时返回迁移事件字典。
        seed: 随机种子；给定后可复现。
        **args: 透传给传播函数。

    返回:
        ``(times, S, I)`` 三个 numpy 数组，或 ``{时刻: 事件列表}``。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> t, S, I = discrete_sis_control_pulse(H, {2: 0.4}, 0.3, tmax=5, seed=2)
        >>> len(t) == len(S) == len(I)
        True
    """
    from honet.contagion._strategies import resolve_strategy

    seed_everything(seed)
    return simulate_discrete(
        H,
        tau,
        gamma,
        model="SIS",
        mode="pulse",
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
