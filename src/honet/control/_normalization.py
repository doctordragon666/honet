"""归一化与统计检验（参考 nctpy ``utils`` / ``plotting`` 的统计部分）。

脑网络控制分析里，两件事容易做错：

1. **状态向量归一化**——不同状态的规模差异巨大（有的含 5 个结点、有的含 30 个），
   不归一化就无法比较控制能量；
2. **显著性检验**——控制能量本身没有绝对标准，必须和「随机网络 / 随机权重」
   的零分布比较，才能说「这个结点显著更好」。

本模块把这两步封装好，避免调用方各写一套。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require


def normalize_state(x: Any) -> Any:
    """用欧氏范数归一化状态向量。

    参数:
        x: 状态向量，形状 ``(N,)`` 或 ``(N, k)``。

    返回:
        归一化后的向量/矩阵。零向量的范数为 0，此时原样返回（避免除零）。

    示例:
        >>> import numpy as np
        >>> v = normalize_state(np.array([3.0, 4.0]))
        >>> np.allclose(v, [0.6, 0.8])
        True
    """
    np = require("numpy")
    array = np.asarray(x, dtype=float)
    if array.ndim == 1:
        norm = np.linalg.norm(array)
        return array if norm == 0 else array / norm
    norms = np.linalg.norm(array, axis=0)
    norms = np.where(norms == 0, 1.0, norms)
    return array / norms


def normalize_weights(
    x: Any,
    *,
    rank: bool = True,
    add_constant: bool = True,
) -> Any:
    """归一化控制权重（用于填充 ``B`` 的对角线）。

    参数:
        x: 原始权重，形状 ``(N,)``。
        rank: 是否先做秩变换（把原始值替换为排名）。默认 ``True``，
            使结果对异常值不敏感。
        add_constant: 是否加一个常数使最小值不为 0。默认 ``True``。

    返回:
        归一化到 ``[0, 1]`` 区间的权重。

    示例:
        >>> import numpy as np
        >>> w = normalize_weights(np.array([10.0, 20.0, 30.0]))
        >>> w.min() > 0 and w.max() <= 1.0
        True

    说明:
        流程为：秩变换（可选）→ 缩放到单位区间 → 加常数（可选）。
        加常数是为了避免权重为 0 导致该结点实际上不受控。
    """
    np = require("numpy")
    values = np.asarray(x, dtype=float).reshape(-1)
    if values.size == 0:
        return values

    if rank:
        # 秩变换：与 scipy.stats.rankdata 一致（并列取平均秩，1 基）
        rankdata = require("scipy.stats", attr="rankdata")
        values = np.asarray(rankdata(values), dtype=float)

    span = values.max() - values.min()
    if span > 0:
        values = (values - values.min()) / span
    else:
        values = np.zeros_like(values)

    if add_constant:
        # 与 nctpy 一致：整体 +1，结果落在 [1, 2]。
        # 注意这不是「缩放到 (0,1]」——后者是另一种归一化，会改变相对量级。
        values = values + 1.0
    return values


def null_p_value(observed: float, null_distribution: Any) -> float:
    """用零分布计算单侧 p 值。

    参数:
        observed: 观测值。
        null_distribution: 零分布样本，形状 ``(n,)``。

    返回:
        p 值，即零分布中 ``>= observed`` 的比例。为防止 0，下界取 ``1/n``。

    示例:
        >>> import numpy as np
        >>> p = null_p_value(5.0, np.arange(100, dtype=float))
        >>> 0 < p <= 0.01
        True

    说明:
        与 nctpy 一致，用「加一平滑」的经验 p 值 ``(k+1)/(n+1)``，避免
        ``p = 0`` 这种无法在对数轴上显示的结果。
    """
    np = require("numpy")
    samples = np.asarray(null_distribution, dtype=float).reshape(-1)
    if samples.size == 0:
        msg = "null_distribution 不能为空"
        raise ValueError(msg)
    count = int(np.sum(samples >= observed))
    return float((count + 1) / (samples.size + 1))


def fdr_p_values(p_values: Any, alpha: float = 0.05) -> tuple[Any, Any]:
    """Benjamini-Hochberg FDR 校正。

    参数:
        p_values: 原始 p 值，形状 ``(m,)``。
        alpha: 显著性水平。

    返回:
        ``(adjusted, rejected)`` 二元组：校正后的 p 值数组，以及布尔数组
        标记哪些假设被拒绝（即显著）。

    示例:
        >>> import numpy as np
        >>> adj, rej = fdr_p_values(np.array([0.001, 0.02, 0.5]))
        >>> rej.tolist()
        [True, True, False]

    说明:
        实现的是 BH 步进法：把 p 值升序排列后取
        ``p_(i) * m / i`` 的累积最小值，保证单调。
    """
    np = require("numpy")
    values = np.asarray(p_values, dtype=float).reshape(-1)
    total = values.size
    if total == 0:
        return values, np.zeros(0, dtype=bool)

    order = np.argsort(values)
    ranked = values[order]
    # BH: p_(i) * m / i，再从后往前取累积最小值保证单调
    scaled = ranked * total / np.arange(1, total + 1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(total)
    adjusted[order] = np.minimum(monotone, 1.0)
    rejected = adjusted < alpha
    return adjusted, rejected


def label_to_state(labels: Any) -> tuple[Any, list[str]]:
    """把字符串标签转成整数编码（参考 nctpy ``convert_states_str2int``）。

    参数:
        labels: 标签序列。

    返回:
        ``(codes, unique_labels)``：整数编码数组与去重后的标签列表。

    示例:
        >>> codes, names = label_to_state(["B", "A", "B"])
        >>> codes.tolist(), names
        ([1, 0, 1], ['A', 'B'])
    """
    np = require("numpy")
    items = [str(item) for item in labels]
    if not items:
        msg = "labels 不能为空"
        raise ValueError(msg)
    unique = sorted(set(items))
    mapping = {name: index for index, name in enumerate(unique)}
    codes = np.array([mapping[item] for item in items], dtype=int)
    return codes, unique
