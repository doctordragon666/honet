"""传播动力学方法的枚举。"""

from __future__ import annotations

from enum import Enum, auto, unique


@unique
class Contagion_Method(Enum):
    """高阶网络传播动力学的可选方法。

    各成员对应的实际算法：

    * ``Gillespie`` —— 连续时间 Gillespie 精确模拟（无噪声）；
    * ``RandomGillespie`` —— 连续时间 Gillespie + 速率噪声；
    * ``Discrete`` —— 离散时间同步更新；
    * ``RandomDiscrete`` —— 离散时间 + 概率噪声。

    类名保留旧库的 ``Contagion_Method`` 拼写（下划线分隔且首字母大写），
    以免破坏既有调用方；它同时以 ``ContagionMethod`` 的名字导出。

    示例:
        >>> Contagion_Method.Discrete.value
        3
    """

    Gillespie = auto()
    RandomGillespie = auto()
    Discrete = auto()
    RandomDiscrete = auto()
