"""head/tail 形式的边列表 → 有向关联矩阵（实现）。

有向超图的关联矩阵约定：head 结点为 ``+1``、tail 结点为 ``-1``、其余为 ``0``。
HAT 用这个符号约定判断超图是否有向（有正有负即有向）。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require


def edge_list_to_incidence(edge_list_local: list[dict[str, list[int]]]) -> Any:
    """把 head/tail 边列表转换为关联矩阵。

    参数:
        edge_list_local: 边列表，每项含 ``head`` 与 ``tail`` 两个结点下标列表。

    返回:
        ``numpy.ndarray``，形状 ``(结点数, 超边数)``、``dtype=int``；
        行下标是结点编号（取所有出现过的结点排序后的位置），列下标是边在
        ``edge_list_local`` 中的序号。

    注意:
        旧实现按「出现过的结点排序后重新编号」建行索引，因此若结点编号不连续，
        矩阵行号与结点编号不再一一对应（例如只用到结点 2、3 时行 0 对应结点 2）。
        这里保持旧行为，调用方需要自己维护这份映射。
    """
    np = require("numpy")

    nodes: set[int] = set()
    for e in edge_list_local:
        nodes.update(e["head"])
        nodes.update(e["tail"])
    nodes_sorted = sorted(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes_sorted)}

    incidence = np.zeros((len(nodes_sorted), len(edge_list_local)), dtype=int)
    for j, e in enumerate(edge_list_local):
        for h in e["head"]:
            incidence[node_to_idx[h], j] = 1
        for t in e["tail"]:
            incidence[node_to_idx[t], j] = -1
    return incidence
