"""高阶网络社区划分方法的枚举。"""

from __future__ import annotations

from enum import Enum, auto, unique


@unique
class Method(Enum):
    """可选的社区划分方法。

    成员与旧库 ``hypertrade.honcommunity.Method`` 逐一对应，名字保持原样
    （``KUMAR`` / ``NMF`` / ``Random`` 等是常量名而非函数名，改成 snake_case
    会破坏既有调用方 ``Method.KUMAR``）：

    * ``KUMAR`` —— Kumar 等人的超图模块度贪心算法；
    * ``NMF`` —— 非负矩阵分解（需要 scikit-learn）；
    * ``Random`` —— 随机均分基线；
    * ``LAST_STEP`` —— 以 Kumar 结果为初值的 Last-Step 局部优化；
    * ``RECURSIVE`` —— 递归二分（对模块度提升的社区继续拆分）；
    * ``Basic`` —— 团投影成普通图后用 Louvain 检测；
    * ``Hierarchy`` —— 超边层次聚类后按多数票投影回结点。

    示例:
        >>> Method.KUMAR.name
        'KUMAR'
    """

    KUMAR = auto()
    NMF = auto()
    Random = auto()
    LAST_STEP = auto()
    RECURSIVE = auto()
    Basic = auto()
    Hierarchy = auto()
