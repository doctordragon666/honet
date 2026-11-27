"""相图单点分类。"""

from __future__ import annotations

from typing import Any

from honet.contagion._stability import classify


def classify_point(
    beta2: float,
    beta_higher: float,
    gamma_global: float,
    H_global: Any,
    kind: str,
    function: Any = None,
) -> bool:
    """把相图上的一个参数点分类为稳定 / 不稳定。

    先用 ``beta=[beta2, beta_higher]`` 生成 ``tau`` 表，再按基本再生数条件
    判断该点落在稳定区还是暴发区。

    参数:
        beta2: 大小为 2 的超边的感染率。
        beta_higher: 其余超边大小的感染率参数。
        gamma_global: 恢复率。
        H_global: hypernetx 超图对象。
        kind: :func:`generate_tau` 的生成方式。
        function: 自定义的 ``tau`` 生成函数
            ``f(H, beta=[...], type=kind) -> dict``；``None`` 时用
            :func:`generate_tau`。

    返回:
        ``True`` 表示该点稳定（不会暴发）。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c", "d"]})
        >>> classify_point(0.01, 0.01, 0.5, H, "fixed_compare")
        True
    """
    return classify(H_global, beta2, beta_higher, gamma_global, kind, function)
