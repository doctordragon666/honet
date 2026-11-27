r"""按比例逐步执行攻击并记录网络度量曲线。

攻击流程：复制超图 → 按 ``rho`` 比例从攻击序列头部取出一批结点（或超边）
移除 → 用度量函数评估当前网络 → 重复，直到序列耗尽或度量降到 0。

默认度量是自然连通度（natural connectivity）：

.. math::

    \bar{\lambda} = \ln\left(\frac{1}{N}\sum_{i=1}^{N} e^{\lambda_i}\right)

其中 :math:`\lambda_i` 为超图邻接矩阵的特征值。值越大说明网络冗余路径越多、
对移除越鲁棒；网络被打散时降到 0，此时攻击提前结束。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

#: 支持的攻击目标
_TARGETS = ("node", "edge")


def _natural_connectivity(H: Any) -> float:
    """计算超图的自然连通度（``execute_attack`` 的默认度量）。

    直接复用 :func:`honet.metrics.calc_natural_connectivity`：它带结构指纹
    缓存，且已处理「只剩孤立结点时 ``adjacency_matrix`` 抛 ValueError」的
    退化情形。攻击过程每步都会重建超图对象，按指纹缓存可避免同一结构被
    重复求特征值。

    参数:
        H: hypernetx 超图对象。

    返回:
        自然连通度；结点为空、或无法构造邻接矩阵时返回 ``0.0``。
    """
    from honet.metrics._reuse import package_function

    if not H.nodes:
        return 0.0
    try:
        return float(package_function("honet.metrics", "calc_natural_connectivity")(H))
    except ValueError:
        # 只剩孤立结点时 2.x 的 adjacency_matrix 会因空索引抛 ValueError
        return 0.0
    except Exception:  # noqa: BLE001 - 度量失败不应中断整条攻击曲线
        return 0.0


def execute_attack(
    hg: Any,
    attack_sequence: Any,
    target: str = "node",
    func: Callable[[Any], float] | None = None,
    rho: float = 0.01,
) -> dict[str, dict[str, list[float]]]:
    """按攻击序列逐步移除结点或超边，记录每一步的网络度量。

    参数:
        hg: hypernetx 超图对象；为 ``None`` 或类型不符时返回空字典。
        attack_sequence: 攻击序列，可以是结点列表，也可以是
            ``{策略名称: 结点列表}`` 形式的字典（一次跑多套策略）。
        target: 攻击目标，``"node"`` 移除结点，``"edge"`` 移除超边。
        func: 度量函数，接收当前超图并返回浮点值；为 ``None`` 时用
            自然连通度。
        rho: 每步移除的比例，每步至少移除 1 个。

    返回:
        ``{策略名称: {"steps": [...], "nc_vals": [...]}}``。

        * ``nc_vals`` 是度量值序列，首项固定为 ``1.0``（旧实现约定的
          「未攻击」基准位，即使传了 ``func`` 也不调用它）；
        * ``steps`` 是与 ``nc_vals`` 等长的「已移除比例」序列，首项为 ``0.0``，
          之后为 ``(原始数量 - 当前数量) / 原始数量``。

    抛出:
        ValueError: ``target`` 不是 ``"node"`` 或 ``"edge"``。

    与旧实现的差异:
        1. 旧代码用 ``hg.copy()`` 与 ``restrict_to_nodes/edges``，二者在
           hypernetx 2.4.3 中分别不存在 / 语义已变（``restrict_to_*`` 返回
           新对象且不修改自身）。这里改用 ``hg.clone()`` 加
           ``remove_nodes/remove_edges`` 真正就地移除。
        2. 旧代码的 ``steps`` 表达式漏了括号（``N - n/N`` 而不是
           ``(N - n)/N``），得到的是「原始数量减比例」这种无意义的数值；
           这里按旧 docstring 声明的意图改为真正的已移除比例。
        3. 旧代码在 ``hg is None`` 时仍引用未定义的 ``attack_sample``，
           直接 ``NameError``；这里改为正常返回空字典。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> execute_attack(H, ["b"], rho=1.0)["默认策略"]["steps"]
        [0.0, 0.3333333333333333]
    """
    if target not in _TARGETS:
        msg = f"target 只能是 {_TARGETS} 之一，收到 {target!r}"
        raise ValueError(msg)

    results: dict[str, dict[str, list[float]]] = {}
    if hg is None or not hasattr(hg, "clone"):
        return results

    # 允许只传一条序列，内部统一成 {策略名称: 序列} 的形式
    samples = (
        attack_sequence if isinstance(attack_sequence, dict) else {"默认策略": attack_sequence}
    )

    for strategy_name, sequence in samples.items():
        remaining = list(sequence)
        nc_vals: list[float] = [1.0]
        steps: list[float] = [0.0]
        # clone() 是深拷贝，改动不会影响调用方传入的超图
        hg_copy = hg.clone()
        total = len(hg_copy.nodes) if target == "node" else len(hg_copy.edges)

        while remaining:
            if target == "node":
                num_to_remove = max(1, int(len(hg_copy.nodes) * rho))
                hg_copy.remove_nodes(remaining[:num_to_remove])
            else:
                num_to_remove = max(1, int(len(hg_copy.edges) * rho))
                hg_copy.remove_edges(remaining[:num_to_remove])
            remaining = remaining[num_to_remove:]

            nc_val = _natural_connectivity(hg_copy) if func is None else float(func(hg_copy))
            nc_vals.append(nc_val)

            current = len(hg_copy.nodes) if target == "node" else len(hg_copy.edges)
            steps.append((total - current) / total if total else 0.0)

            if nc_val == 0:
                break  # 度量已归零，网络被打散，提前结束

        results[strategy_name] = {"steps": steps, "nc_vals": nc_vals}

    return results
