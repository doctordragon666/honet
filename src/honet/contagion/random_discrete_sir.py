"""带噪声的随机离散时间 SIR 模型。"""

from __future__ import annotations

from typing import Any

from honet.contagion._common import seed_everything
from honet.contagion._random_models import simulate_random_discrete


def random_discrete_sir(
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
    dt: float = 1.0,
    return_full_data: bool = False,
    *,
    seed: int | None = None,
    **args: Any,
) -> tuple[list[float], list[int], list[int], list[int]]:
    """随机离散时间 SIR 模型（旧名 ``RandomDiscrete_SIR``）。

    .. note::
        旧库里这个函数只有签名、函数体为空（调用后返回 ``None``）。本实现按
        同目录 ``random_discrete_sis`` 的口径补齐为真正的 SIR 过程：感染概率
        与 SIS 相同，恢复概率 ``gamma`` 带 ``sigma2`` 噪声，恢复后进入 ``R``
        态且不再被感染。参数语义与返回结构沿用旧签名，因此调用方代码无需修改。

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
        tmax: 终止时间（含端点）。
        dt: 时间步长。
        return_full_data: 兼容旧签名，本实现未使用。
        seed: 随机种子；给定后可复现。
        **args: 兼容旧签名。

    返回:
        ``(t, S, I, R)`` 四个普通 Python 列表。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> t, S, I, R = random_discrete_sir(H, {2: 0.4}, 0.3, rho=0.5, tmax=4, seed=1)
        >>> len(t) == len(R)
        True
    """
    seed_everything(seed)
    return simulate_random_discrete(
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
        dt=dt,
        **args,
    )
