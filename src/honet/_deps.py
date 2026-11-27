"""可选依赖的统一导入入口。

honet 的核心容器不依赖任何第三方库；分析类子包按需 import 各自需要的包。
本模块集中处理「依赖缺失时给出可执行的安装提示」这一件事，避免每个实现
模块各写一份 try/except。
"""

from __future__ import annotations

from typing import Any

_HINT = 'pip install "honet[analysis]"（或 uv add "honet[analysis]"）'


def require(module: str, *, extra: str = "analysis", attr: str | None = None) -> Any:
    """导入可选依赖，缺失时抛出带安装提示的 ImportError。

    参数:
        module: 顶层模块名，例如 ``"numpy"``、``"hypernetx"``。
        extra: 所属 extras 组名，仅用于错误提示。
        attr: 需要一并取出的属性名；为 ``None`` 时返回模块本身。

    返回:
        模块对象，或 ``getattr(模块, attr)``。
    """
    import importlib

    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        hint = _HINT if extra == "analysis" else f'pip install "honet[{extra}]"'
        msg = f"honet 的可选依赖 {module!r} 未安装（需要 honet[{extra}]）。安装方式：{hint}。"
        raise ImportError(msg) from exc
    return mod if attr is None else getattr(mod, attr)


def has(module: str) -> bool:
    """判断可选依赖是否可用，不抛异常（用于自动降级的分支）。"""
    import importlib.util

    return importlib.util.find_spec(module) is not None
