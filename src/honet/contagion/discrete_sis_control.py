"""离散时间 SIS 模型（支持控制结点）。"""

from __future__ import annotations

from typing import Any

from honet._deps import require
from honet.contagion._common import seed_everything
from honet.contagion._discrete import simulate_discrete


def discrete_sis_control(
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
    """离散时间 SIS 模型，支持控制结点（旧名 ``discrete_SIS_control``）。

    当 ``transmission_function`` 取字符串 ``"target"`` 时按「目标控制」处理
    （受控结点无条件恢复）；其余情况一律走脉冲控制，与旧实现一致。

    参数:
        H: hypernetx 超图对象。
        tau: ``{超边大小: 感染率}``，必须覆盖图中出现的全部超边大小。
        gamma: 恢复率。
        transmission_function: 传播函数或策略名；``None`` 表示 ``threshold``。
        initial_infecteds: 初始感染者；与 ``rho`` 互斥。
        initial_recovereds: 受控结点（SIS 中被强制转为 ``S``）。
        rho: 初始感染比例。
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
        >>> t, S, I = discrete_sis_control(H, {2: 0.4}, 0.3, tmax=5, seed=1)
        >>> len(t) == len(S) == len(I)
        True
    """
    seed_everything(seed)
    if transmission_function == "target":
        # 目标控制：受控结点被直接恢复，不经过脉冲阈值判定
        function = require("hypernetx", attr="collective_contagion")
        mode = "control"
    else:
        # 旧实现无视传入的传播函数，一律以 collective_contagion 走脉冲控制；
        # 这里保持一致，避免同一份输入在新旧库下得到不同轨迹。
        function = require("hypernetx", attr="collective_contagion")
        mode = "pulse"

    return simulate_discrete(
        H,
        tau,
        gamma,
        model="SIS",
        mode=mode,
        transmission_function=function,
        initial_infecteds=initial_infecteds,
        initial_recovereds=initial_recovereds,
        rho=rho,
        tmin=tmin,
        tmax=tmax,
        dt=dt,
        return_full_data=return_full_data,
        **args,
    )
