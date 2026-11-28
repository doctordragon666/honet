"""线性时变（LTV）系统的参数解析（私有）。

控制框架里 ``A``、``B`` 传统上是常数矩阵，但真实系统往往时变：脑刺激实验中
连接强度随疲劳变化、通信网络随负载改变、机械系统的刚度随温度漂移。

本模块把「常数矩阵或时间函数」的解析规则收敛到一处，供
:mod:`honet.control._simulate_system` 与 LTV 可控性分析共用：

* 传 ``ndarray`` 时行为与旧版完全一致，不做任何额外计算；
* 传 ``Callable[[float], ndarray]`` 时，在每个时刻用 ``t`` 求值。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require


def resolve_matrix(value: Any, time: float, *, rows: int, cols: int, name: str) -> Any:
    """取矩阵在时刻 ``time`` 的取值并校验形状。

    参数:
        value: 常数矩阵，或接收时间返回矩阵的可调用对象。
        time: 当前时刻。
        rows: 期望行数。
        cols: 期望列数。
        name: 参数名，仅用于报错信息。

    返回:
        形状为 ``(rows, cols)`` 的 ndarray。

    抛出:
        ValueError: 求值结果形状不符。
        TypeError: 求值结果无法转成数值矩阵。
    """
    np = require("numpy")
    raw = value(time) if callable(value) else value
    try:
        matrix = np.atleast_2d(np.asarray(raw, dtype=float))
    except (TypeError, ValueError) as exc:
        msg = f"{name} 在 t={time} 处无法转成数值矩阵：{exc}"
        raise TypeError(msg) from exc
    if matrix.shape != (rows, cols):
        msg = f"{name} 在 t={time} 处形状应为 {(rows, cols)}，实际是 {matrix.shape}"
        raise ValueError(msg)
    return matrix


def is_time_varying(value: Any) -> bool:
    """判断参数是否时变（可调用）。

    示例:
        >>> is_time_varying([[1.0]])
        False
        >>> is_time_varying(lambda t: [[1.0]])
        True
    """
    return callable(value)


def sample_grid(start: float, stop: float, steps: int) -> list[float]:
    """生成 ``steps`` 个采样时刻（含两端点）。

    参数:
        start: 起点。
        stop: 终点。
        steps: 采样点数，必须 >= 2。

    返回:
        递增的时刻列表。

    抛出:
        ValueError: ``steps < 2`` 或 ``stop < start``。
    """
    if steps < 2:
        msg = f"steps 必须 >= 2，收到 {steps}"
        raise ValueError(msg)
    if stop < start:
        msg = f"stop 必须 >= start，收到 start={start}, stop={stop}"
        raise ValueError(msg)
    if stop == start:
        return [start] * steps
    span = (stop - start) / (steps - 1)
    return [start + index * span for index in range(steps)]
