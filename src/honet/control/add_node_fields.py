"""在 HAT 超图的结点表上登记每个结点参与的超边及其角色。"""

from __future__ import annotations

from typing import Any

from honet.control._node_fields import add_node_fields as _impl


def add_node_fields(HG_local: Any) -> Any:
    """为结点表添加 ``edges`` / ``head`` / ``tail`` 三列。

    三列都记录「超边行号列表」：``edges`` 是该结点所属的全部超边，``head`` 是它
    作为头结点的超边，``tail`` 是它作为尾结点的超边。无向超图没有方向，成员
    同时计入 ``head`` 与 ``tail``。

    参数:
        HG_local: HAT 超图对象，结点表需含 ``Nodes`` 列。

    返回:
        传入的同一个对象（原地修改，与旧实现一致）。

    示例:
        >>> HG = add_node_fields(HG)  # doctest: +SKIP
        >>> HG.nodes["edges"].tolist()  # doctest: +SKIP
        [[0], [0, 1], [1]]
    """
    return _impl(HG_local)
