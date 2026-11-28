"""时变参数解析（私有）。

传播模型的参数（感染率 ``tau``、恢复率 ``gamma``）传统上都是常数，但真实场景
里它们往往随时间变化：流感有季节性、防疫政策会在特定日期调整、宣传活动的
效果随热度衰减。

本模块提供统一的「常数或时间函数」解析规则，让所有传播模型用同一套语义：

* 传**常数**（``float`` / ``dict``）时行为与旧版完全一致，不做任何额外计算；
* 传**可调用对象**（``Callable[[float], T]``）时，每一步用当前时间 ``t`` 求值。

为什么不让调用方自己在外层循环里改参数：那需要把整个模拟过程拆成多段手工拼接，
既容易漏掉状态延续，也无法表达连续变化。这里把「随时间求值」下沉到模型内部，
调用方只需给一个函数。
"""

from __future__ import annotations

from typing import Any


def resolve_at(value: Any, time: float) -> Any:
    """取参数在时刻 ``time`` 的取值。

    参数:
        value: 常数，或接收时间返回取值的可调用对象。
        time: 当前模拟时间。

    返回:
        ``value`` 本身（常数），或 ``value(time)`` 的返回值（可调用）。

    示例:
        >>> resolve_at(0.3, 5.0)
        0.3
        >>> resolve_at(lambda t: 0.1 * t, 2.0)
        0.2
    """
    return value(time) if callable(value) else value


def resolve_tau_at(tau: Any, time: float) -> dict[int, float]:
    """把 ``tau`` 解析成 ``{超边大小: 感染率}`` 字典。

    ``tau`` 支持三种形态：

    * ``{大小: 感染率}`` 常数表——原样返回；
    * ``Callable[[float], dict[int, float]]``——用 ``time`` 求值；
    * 数值常数——包装成 ``{2: 值}``（与「只有成对交互」的直觉一致）。

    参数:
        tau: 感染率参数。
        time: 当前模拟时间。

    返回:
        ``{超边大小: 感染率}``。

    抛出:
        TypeError: 求值结果既不是字典也不是数值。
    """
    resolved = resolve_at(tau, time)
    if isinstance(resolved, dict):
        return {int(size): float(rate) for size, rate in resolved.items()}
    if isinstance(resolved, int | float):
        return {2: float(resolved)}
    msg = f"tau 必须解析为 {{超边大小: 感染率}} 或数值，收到 {type(resolved).__name__}"
    raise TypeError(msg)


def beta_from_tau(tau_table: dict[int, float]) -> float:
    """把 ``tau`` 表压成单个 ``beta``（取各阶平均值）。

    与旧实现一致：旧代码在循环外做 ``np.mean(list(tau.values()))``。
    """
    if not tau_table:
        return 0.0
    return float(sum(tau_table.values()) / len(tau_table))
