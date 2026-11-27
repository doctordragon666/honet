"""从攻击样本中截取最重要的前若干个结点。

攻击样本（``attack_sequence``）通常已经按重要性排好序，本函数只负责按
「个数」或「比例」截断，供后续 ``execute_attack`` 使用。
"""

from __future__ import annotations

from typing import Any

#: 既没有给 num 也没有给 partion 时默认截取的个数
_DEFAULT_NUM = 10


def main_country(
    attack_sample: Any,
    num: int | None = None,
    partion: float | None = None,
) -> list[Any]:
    """输出攻击样本中最重要的前 ``num`` 个或前 ``partion`` 比例的结点。

    参数:
        attack_sample: 攻击样本，可以是已排序的结点列表，也可以是
            ``{名称: 结点}`` 形式的字典（此时按字典的值截取）。
        num: 截取的结点数量；为 ``None`` 时按 ``partion`` 或默认 10 个。
        partion: 截取的比例，取值 ``0~1``，实际个数向下取整。

    返回:
        截取后的结点标识列表。

    说明:
        与旧实现一致，``num`` 与 ``partion`` 同时给出时以 ``num`` 为准并
        打印提示（旧库的行为是 ``print``，这里保留以便调用方无感迁移）。
        比例截断可能得到 0 个，调用方需自行保证 ``partion`` 足够大。

    示例:
        >>> main_country(["a", "b", "c", "d"], partion=0.5)
        ['a', 'b']
    """
    if num is None:
        if partion is not None:
            num = int(len(attack_sample) * partion)
        else:
            num = _DEFAULT_NUM
    elif partion is not None:
        print("不可同时设置num和partion参数，请选择其中一个进行设置")

    # 返回选择的节点的uid列表
    if isinstance(attack_sample, dict):
        return list(attack_sample.values())[:num]
    return list(attack_sample[:num])
