"""传播策略字符串 → hypernetx 传播函数的解析（私有，不对外导出）。

hypernetx 的离散 / Gillespie 模型都接受一个 ``transmission_function``，
签名是 ``f(node, status, edge, **kwargs) -> bool``。旧库用字符串挑选策略，
本模块集中维护这张映射表。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require

#: 策略名 → hypernetx 中对应的传播函数名
_STRATEGY_FUNCTIONS: dict[str, str] = {
    "individual": "individual_contagion",
    "collective": "collective_contagion",
    "default": "threshold",
    "threshold": "threshold",
    "majority_vote": "majority_vote",
    # 下面两个是写给 *_control 系列区分控制方式的，传播函数仍沿用 individual
    "pulse": "individual_contagion",
    "target": "individual_contagion",
}


def resolve_strategy(strategy: str | Any) -> Any:
    """把策略字符串解析成 hypernetx 的传播函数。

    参数:
        strategy: 策略名（``individual`` / ``collective`` / ``threshold`` /
            ``default`` / ``majority_vote`` / ``pulse`` / ``target``）；
            已经是可调用对象时原样返回，便于调用方直接传函数。

    返回:
        hypernetx 的传播函数。

    抛出:
        ValueError: 策略名不在支持列表内。
    """
    if callable(strategy):
        return strategy
    try:
        function_name = _STRATEGY_FUNCTIONS[strategy]
    except KeyError:
        msg = f"未知的传播策略：{strategy!r}，可用：{sorted(_STRATEGY_FUNCTIONS)}"
        raise ValueError(msg) from None
    return require("hypernetx", attr=function_name)
