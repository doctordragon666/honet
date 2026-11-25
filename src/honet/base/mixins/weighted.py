"""honet.base.mixins.weighted —— 加权能力 mixin。"""

from __future__ import annotations


class WeightedMixin:
    """加权能力 mixin：``is_weighted=True``，权重默认 ``1.0`` 且必须为正的有限实数。

    只提供 :meth:`_prepare_weight`，具体存储由各形态自己负责，
    因此可以和任意形态基类组合（``...W`` / ``...WD``）。
    """

    is_weighted: bool = True

    def _prepare_weight(self, weight: float = 1.0) -> float:
        """校验并归一化权重。

        Args:
            weight: 交互强度／频次，默认 1.0。

        Returns:
            归一化为 ``float`` 的权重。

        Raises:
            ValueError: 不是实数（含 ``bool``、字符串）或不是正数（含 NaN、0、负数）。
        """
        # bool 是 int 的子类，但 True/False 作为权重没有语义，直接拒绝
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"权重必须是实数：{weight!r}")
        # ``w != w`` 是 NaN 的判据；``not w > 0`` 同时挡住 0、负数与 NaN
        if not weight > 0:
            raise ValueError(f"权重必须为正数：{weight!r}")
        return float(weight)


__all__ = ["WeightedMixin"]
