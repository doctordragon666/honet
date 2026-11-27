"""SIR 模型统一入口（``hon_sir`` / 旧名 ``honSIR``）。

实现已收敛到 :func:`honet.contagion._simulate.simulate`；本模块只保留旧签名。
新代码建议直接用 ``simulate(..., model="SIR")``，或换成 ``"SIER"`` / ``"SIRS"``
等模型名即可，无需新的入口函数。
"""

from __future__ import annotations

from typing import Any

from honet.contagion._simulate import simulate
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
      时改走控制语义；
    * ``Gillespie`` —— hypernetx 的 ``Gillespie_SIR``；
    * ``RandomDiscrete`` / ``RandomGillespie`` —— 带噪声的平均场模型。

    参数:
        H: hypernetx 超图对象。
        gamma: 恢复率。
        tau: ``{超边大小: 感染率}``。
        method: 算法选择，见 :class:`~honet.contagion.Contagion_Method`。
        initial_infection: 初始感染者列表；``None`` 时按 ``rho=0.1`` 随机抽。
        initial_recovereds: 受控（强制康复）结点。
        strategy: 传播策略名。
        tmax: 模拟终止时间。
        seed: 随机种子；``None`` 表示不播种。
        return_all_data: 是否返回完整迁移数据。

    返回:
        ``(times, S, I, R)`` 四个序列，或完整迁移数据。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> t, S, I, R = hon_sir(H, 0.3, {2: 0.4}, tmax=5, seed=1)
        >>> len(t) == len(R)
        True
    """
    return simulate(
        H,
        gamma,
        tau,
        model="SIR",
        method=method,
        initial_infection=initial_infection,
        initial_recovereds=initial_recovereds,
        strategy=strategy,
        tmax=tmax,
        seed=seed,
        return_all_data=return_all_data,
    )
