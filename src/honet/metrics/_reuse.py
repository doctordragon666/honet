"""跨模块取用公开函数的统一入口（私有，不对外导出）。

``honet`` 的分析子包用 PEP 562 惰性导出：包属性在被访问前并不存在，由
``__getattr__`` 在访问时 import 实现模块并写回包属性。

若实现模块直接写 ``from honet.metrics.x import x``，Python 的 import 机制会
先把**子模块对象**写进包属性 ``honet.metrics.x``，之后惰性导出拿到的就是
模块而不是函数，公开 API 被永久遮蔽（``honet.metrics.x`` 不再是可调用对象）。

本模块统一改用 ``importlib`` 取出函数本身，并把函数写回包属性，使包属性
始终是可调用的公开符号。子包内部复用兄弟函数、以及复用 ``honet.convert``
的转换函数时都走这里，不要直接 ``from ... import ...``。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


def package_function(package: str, name: str) -> Any:
    """取出 ``<package>.<name>`` 实现模块里的同名函数，并修复包属性。

    参数:
        package: 惰性导出的包名，例如 ``"honet.metrics"``、``"honet.convert"``。
        name: 公开函数名，同时也是实现模块名（如 ``"clustering_coefficient"``）。

    返回:
        实现模块中与该名字同名的函数对象。

    抛出:
        ImportError: 实现模块的第三方依赖缺失。
        AttributeError: 实现模块里没有同名函数。
    """
    pkg = import_module(package)
    module = import_module(f"{package}.{name}")
    function = getattr(module, name)
    # 子模块导入已把模块对象写到包属性上；改回函数，保持惰性导出的对外契约
    setattr(pkg, name, function)
    return function
