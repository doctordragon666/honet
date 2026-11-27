"""可选依赖子包的惰性导出机制。

honet 的 ``dependencies`` 为空，``import honet`` 不允许触发任何第三方依赖。
分析类子包（``io`` / ``convert`` / ``build`` / ``metrics`` / ``community`` /
``contagion`` / ``attack`` / ``control`` / ``plot`` / ``export``）依赖 numpy、
pandas、hypernetx、matplotlib、HAT 等可选包，因此它们自身以及顶层 ``honet``
都使用 PEP 562 的模块级 ``__getattr__`` 惰性加载：只有真正访问某个符号时才
import 对应实现模块，缺依赖时抛出附带安装提示的 ``ImportError``。

约定（子包 ``__init__.py`` 的标准写法）::

    from honet._lazy import lazy_dir, lazy_getattr

    _EXPORTS = {
        "hon_sis": "honet.contagion.hon_sis:hon_sis",
    }
    __all__ = sorted(_EXPORTS)

    def __getattr__(name: str):
        return lazy_getattr(globals(), __name__, _EXPORTS, name, "analysis")

    __dir__ = lambda: lazy_dir(_EXPORTS)  # 不推荐，见下方 __dir__ 说明

``_EXPORTS`` 的值是 ``"实现模块路径:模块内属性名"``，必须是**单文件**实现
模块，不得指向另一个惰性子包，否则首次访问会连带解析整包。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

#: 可选依赖组 → 面向用户的安装提示（与 pyproject.toml 的 optional-dependencies 对应）
_EXTRAS_HINT: dict[str, str] = {
    "analysis": 'uv add "honet[analysis]"（或 pip install "honet[analysis]"）',
    "viz": 'uv add "honet[viz]"（或 pip install "honet[viz]"）',
    "excel": 'uv add "honet[excel]"（或 pip install "honet[excel]"）',
    "hat": 'uv add "honet[hat]"（或 pip install "honet[hat]"）',
    "all": 'uv add "honet[all]"（或 pip install "honet[all]"）',
}


def lazy_getattr(
    module_globals: dict[str, Any],
    module_name: str,
    exports: dict[str, str],
    name: str,
    extra: str,
) -> Any:
    """按需 import ``exports[name]`` 指向的实现模块并缓存结果。

    参数:
        module_globals: 调用方模块的 ``globals()``，用于解析后写回缓存，
            使同名属性后续访问不再重复 import。
        module_name: 调用方模块名，仅用于错误信息。
        exports: ``{导出名: "实现模块:属性"}`` 映射。
        name: 被访问的属性名。
        extra: 缺失依赖时提示的 extras 组名。

    返回:
        导出对象本身。

    抛出:
        AttributeError: ``name`` 不在 ``exports`` 中。
        ImportError: 实现模块或其第三方依赖缺失。
    """
    try:
        target = exports[name]
    except KeyError:
        msg = f"module {module_name!r} has no attribute {name!r}"
        raise AttributeError(msg) from None

    module_path, _, attribute = target.partition(":")
    try:
        module = import_module(module_path)
    except ImportError as exc:
        hint = _EXTRAS_HINT.get(extra, extra)
        msg = (
            f"honet.{name} 需要可选依赖，但 import {module_path} 失败：{exc}。"
            f"安装方式：{hint}；完整分析功能用 honet[all]。"
        )
        raise ImportError(msg) from exc

    value = getattr(module, attribute)
    module_globals[name] = value
    return value


def lazy_dir(exports: dict[str, str]) -> list[str]:
    """返回 ``dir(子包)`` 可见的公开名字列表（对应 ``__all__``）。"""
    return sorted(exports)
