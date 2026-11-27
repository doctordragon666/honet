"""从 hypernetx 超图出发计算驱动结点集合（实现）。"""

from __future__ import annotations

from typing import Any

from honet.control._compute_drivers_on_hat import _compute_drivers_on_hat
from honet.control._node_fields import add_node_fields


def compute_drivers_from_hypernetx(
    hnx_hg: Any, out_csv: str | None = "drivers_from_hnx.csv", HAT_type: bool = False
) -> dict[str, Any]:
    """从 hypernetx 超图（或现成的 HAT 超图）计算驱动结点集合并导出 CSV。

    参数:
        hnx_hg: hypernetx 超图，**或**已构造好的 HAT 有向超图（见 ``HAT_type``）。
        out_csv: 输出 CSV 路径；``None`` 或空串表示不写文件。
        HAT_type: 为 ``True`` 时表示 ``hnx_hg`` 已经是 HAT 超图，跳过转换。

    返回:
        ``{"algorithm": 算法名, "dilation_exists": 是否完全不可控,
        "drivers": 驱动结点下标升序列表}``。下标对应 ``HG.nodes["Names"]``
        （转换时结点名按字典序排序后编号）。

    注意:
        旧实现在 ``HAT_type=True`` 时不会给 ``HG`` 赋值，会抛
        ``UnboundLocalError``；这里修正为直接使用传入对象。
    """
    if HAT_type:
        HG = hnx_hg
    else:
        # 延迟到函数内导入，避免 honet.control 在 import 阶段拉入转换模块
        from honet.convert.hypernetx_to_hat_directed import hypernetx_to_hat_directed

        HG = hypernetx_to_hat_directed(hnx_hg)
    return _compute_drivers_on_hat(add_node_fields(HG), out_csv)
