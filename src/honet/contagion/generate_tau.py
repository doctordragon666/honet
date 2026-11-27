"""按超边大小生成感染率 tau 表。"""

from __future__ import annotations

from typing import Any

from honet.contagion._tau import build_tau


def generate_tau(H: Any, beta: list[float], type: str = "fixed") -> dict[int, float]:
    """根据超边大小生成感染率 ``tau`` 表。

    参数:
        H: hypernetx 超图对象。
        beta: 感染率参数列表，含义随 ``type`` 变化。
        type: 生成方式，取值见下表。

    ================== ===========================================================
    type               含义（``beta[0]`` 为主参数，``beta[1]`` 为「其他大小」参数）
    ================== ===========================================================
    ``fixed``          所有超边大小同一感染率 ``beta[0]``
    ``fixed_compare``  大小为 2 的超边取 ``beta[0]``，其余取 ``beta[1]``
    ``size_based``     ``beta[0] * m``（与超边大小成正比）
    ``size_based_compare`` 大小为 2 取 ``beta[0]``，其余取 ``beta[1] * m``
    ``sum_based``      ``beta[0] * m / sum(m)``（各不同大小的 tau 之和为 ``beta[0]``）
    ``sum_based_compare``  大小为 2 取 ``beta[0]``，其余 ``beta[1] * m / sum(m)``
    ================== ===========================================================

    返回:
        ``{超边大小: 感染率}``，键为 ``int``。

    抛出:
        ValueError: ``beta`` 为空，或 ``type`` 未知。
        IndexError: 带 ``_compare`` 的 ``type`` 只给了 1 个 ``beta`` 值。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["a", "b", "c"]})
        >>> generate_tau(H, [0.2], type="fixed")
        {2: 0.2, 3: 0.2}
    """
    return build_tau(H, beta, type)
