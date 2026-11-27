"""SIS 模型统一入口（``hon_sis`` / 旧名 ``honSIS``）。

实现已收敛到 :func:`honet.contagion._simulate.simulate`；本模块只保留旧签名，
保证既有调用方与旧代码不受影响。新代码建议直接用 ``simulate(..., model="SIS")``
或更通用的模型名。
"""

from __future__ import annotations

from typing import Any

from honet.contagion._simulate import simulate
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
      时改走脉冲控制语义；
    * ``Gillespie`` —— hypernetx 的 ``Gillespie_SIS``；
    * ``RandomDiscrete`` —— 带噪声的离散时间平均场模型；
    * ``RandomGillespie`` —— 带噪声的连续时间平均场模型。

    参数:
        H: hypernetx 超图对象。
        gamma: 恢复率。
        tau: ``{超边大小: 感染率}``。
        method: 算法选择，见 :class:`~honet.contagion.Contagion_Method`。
        initial_infection: 初始感染者列表；``None`` 时按 ``rho=0.1`` 随机抽。
        strategy: 传播策略名，见 :mod:`honet.contagion._strategies`。
        tmax: 模拟终止时间。
        seed: 随机种子；``None`` 表示不播种。
        return_all_data: 是否返回完整迁移数据。
        control_nodes: 控制结点；仅 ``Discrete`` 方法使用。

    返回:
        ``(times, S, I)`` 三个序列，或完整迁移数据。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> t, S, I = hon_sis(H, 0.3, {2: 0.4}, tmax=5, seed=1)
        >>> len(t) == len(I)
        True
    """
    return simulate(
        H,
        gamma,
        tau,
        model="SIS",
        method=method,
        initial_infection=initial_infection,
        strategy=strategy,
        tmax=tmax,
        seed=seed,
        return_all_data=return_all_data,
        control_nodes=control_nodes,
    )
