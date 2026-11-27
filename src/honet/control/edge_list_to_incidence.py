"""head/tail 形式的边列表 → 有向关联矩阵。"""

from __future__ import annotations

from typing import Any

from honet.control._incidence import edge_list_to_incidence as _impl


def edge_list_to_incidence(edge_list_local: list[dict[str, list[int]]]) -> Any:
    """把 head/tail 形式的边列表转换为有向关联矩阵。

    有向超图的关联矩阵约定：head 结点为 ``+1``、tail 结点为 ``-1``、其余为 ``0``。
    HAT 用这个符号约定判断超图是否有向（有正有负即有向）。

    参数:
        edge_list_local: 边列表，每项含 ``head`` 与 ``tail`` 两个结点下标列表。

    返回:
        ``numpy.ndarray``，形状 ``(结点数, 超边数)``、``dtype=int``。列下标是边在
        列表中的序号。

    注意:
        行下标是「出现过的结点排序后的位置」，不是结点编号本身：若结点编号不连续
        （例如只用到 2、3），行 0 对应结点 2。旧实现即如此，调用方需自行维护映射。

    示例:
        >>> edge_list_to_incidence([{"head": [1], "tail": [0]}]).tolist()
        [[-1], [1]]
    """
    return _impl(edge_list_local)
