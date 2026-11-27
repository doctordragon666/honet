"""带噪声的随机 Gillespie SIR 模型。"""

from __future__ import annotations

from typing import Any

from honet.contagion._common import seed_everything
from honet.contagion._random_models import simulate_random_gillespie


def random_gillespie_sir(
    H: Any,
    tau: dict[int, float],
    gamma: float,
    sigma1: float = 0.1,
    sigma2: float = 0.1,
    transmission_function: Any = None,
    initial_infecteds: Any = None,
    initial_recovereds: Any = None,
    rho: float | None = None,
    tmin: float = 0,
    tmax: float = float("Inf"),
    return_full_data: bool = False,
    *,
    seed: int | None = None,
    **args: Any,
) -> tuple[list[float], list[int], list[int], list[int]]:
    """随机 Gillespie SIR 模型（旧名 ``RandomGillespie_SIR``）。

    感染者恢复后进入 ``R`` 态且不再被感染；事件等待时间服从指数分布。

    参数:
        H: hypernetx 超图对象。
        tau: ``{超边大小: 感染率}``，只用于取均值 ``beta``。
        gamma: 恢复率。
        sigma1: 感染率噪声强度。
        sigma2: 恢复率噪声强度。
        transmission_function: 兼容旧签名，本实现未使用。
        initial_infecteds: 兼容旧签名，本实现按 ``rho`` 抽样。
        initial_recovereds: 兼容旧签名，本实现未使用。
        rho: 初始感染比例。
        tmin: 起始时间。
        tmax: 终止时间（不含端点）。
        return_full_data: 兼容旧签名，本实现未使用。
        seed: 随机种子；给定后可复现。
        **args: 兼容旧签名。

    返回:
        ``(t, S, I, R)`` 四个普通 Python 列表。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> t, S, I, R = random_gillespie_sir(H, {2: 0.4}, 0.3, rho=0.5, tmax=4, seed=1)
        >>> len(t) == len(R)
        True
    """
    seed_everything(seed)
    return simulate_random_gillespie(
        H,
        tau,
        gamma,
        model="SIR",
        sigma1=sigma1,
        sigma2=sigma2,
        rho=0.1 if rho is None else rho,
        initial_infecteds=initial_infecteds,
        tmin=tmin,
        tmax=tmax,
        **args,
    )
