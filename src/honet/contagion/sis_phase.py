"""SIS 相图数据。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from honet._deps import require
from honet.contagion._stability import classify


def sis_phase(
    H: Any,
    gamma: float = 0.3,
    beta2_lst: Any = None,
    beta_higher_lst: Any = None,
    kind: str = "fixed",
    function: Any = None,
    max_workers: int = 32,
) -> Any:
    """计算 SIS 相图矩阵。

    对 ``beta2_lst × beta_higher_lst`` 的每个参数组合调用
    :func:`classify_point`，把布尔结果写进矩阵：行对应
    ``beta_higher_lst``，列对应 ``beta2_lst``。

    参数:
        H: hypernetx 超图对象。
        gamma: 恢复率。
        beta2_lst: 大小为 2 的超边的感染率取值序列。
        beta_higher_lst: 其余超边大小的感染率参数取值序列。
        kind: :func:`generate_tau` 的生成方式。
        function: 自定义的 ``tau`` 生成函数，见 :func:`classify_point`。
        max_workers: 线程池上限；实际取 ``min(max_workers, 工作项数)``。

    返回:
        形状为 ``(len(beta_higher_lst), len(beta2_lst))`` 的整型 numpy 数组，
        ``1`` 表示稳定，``0`` 表示不稳定。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c", "d"]})
        >>> phase = sis_phase(H, 0.5, [0.01, 0.5], [0.01], kind="fixed_compare")
        >>> phase.shape
        (1, 2)
    """
    np = require("numpy")

    beta2_values = np.asarray([] if beta2_lst is None else beta2_lst, dtype=float)
    beta_higher_values = np.asarray([] if beta_higher_lst is None else beta_higher_lst, dtype=float)
    phase_map = np.zeros((len(beta_higher_values), len(beta2_values)), dtype=int)

    work_items = [
        (iy, ix, float(beta2), float(beta_higher))
        for iy, beta_higher in enumerate(beta_higher_values)
        for ix, beta2 in enumerate(beta2_values)
    ]
    if not work_items:
        return phase_map

    workers = min(max_workers, len(work_items))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(classify, H, beta2, beta_higher, gamma, kind, function): (iy, ix)
            for iy, ix, beta2, beta_higher in work_items
        }
        for future, (iy, ix) in futures.items():
            phase_map[iy, ix] = int(future.result())

    return phase_map
