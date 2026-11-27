"""高阶网络上的传播动力学：SIS / SIR 模拟、感染率生成与相图。

本子包的全部符号都是惰性导出的：只有真正访问某个符号时才会 import
对应的实现模块，因此 ``import honet`` 不会因此拉入可选依赖。

旧的驼峰命名（``honSIS`` / ``honSIR`` / ``Contagion_Method`` /
``discrete_SIS_control`` / ``RandomDiscrete_SIS`` 等）保留为别名，指向同一个
对象，便于旧代码平滑迁移；新代码请使用 snake_case 名字。
"""

from __future__ import annotations

import sys
from types import ModuleType

from honet._lazy import lazy_getattr

_EXPORTS = {
    # 传播模型入口
    "hon_sis": "honet.contagion.hon_sis:hon_sis",
    "hon_sir": "honet.contagion.hon_sir:hon_sir",
    "simulate": "honet.contagion._simulate:simulate",
    # 仓室模型规格（可扩展：注册新规格即可支持新模型）
    "CompartmentModel": "honet.contagion._model:CompartmentModel",
    "register_model": "honet.contagion._model:register_model",
    "get_model": "honet.contagion._model:get_model",
    "available_models": "honet.contagion._model:available_models",
    # 参数与相图
    "generate_tau": "honet.contagion.generate_tau:generate_tau",
    "is_stable": "honet.contagion.is_stable:is_stable",
    "classify_point": "honet.contagion.classify_point:classify_point",
    "sis_phase": "honet.contagion.sis_phase:sis_phase",
    "Contagion_Method": "honet.contagion.contagion_method:Contagion_Method",
    # 离散时间模型
    "discrete_sis_control": "honet.contagion.discrete_sis_control:discrete_sis_control",
    "discrete_sis_control_pulse": (
        "honet.contagion.discrete_sis_control_pulse:discrete_sis_control_pulse"
    ),
    "discrete_sir_control": "honet.contagion.discrete_sir_control:discrete_sir_control",
    # 带噪声的随机模型
    "random_discrete_sis": "honet.contagion.random_discrete_sis:random_discrete_sis",
    "random_discrete_sir": "honet.contagion.random_discrete_sir:random_discrete_sir",
    "random_gillespie_sis": "honet.contagion.random_gillespie_sis:random_gillespie_sis",
    "random_gillespie_sir": "honet.contagion.random_gillespie_sir:random_gillespie_sir",
    # 旧名别名（值指向同一实现模块里的同一属性）
    "honSIS": "honet.contagion.hon_sis:hon_sis",
    "honSIR": "honet.contagion.hon_sir:hon_sir",
    "ContagionMethod": "honet.contagion.contagion_method:Contagion_Method",
    "discrete_SIS_control": "honet.contagion.discrete_sis_control:discrete_sis_control",
    "discrete_SIS_control_pulse": (
        "honet.contagion.discrete_sis_control_pulse:discrete_sis_control_pulse"
    ),
    "discrete_SIR_control": "honet.contagion.discrete_sir_control:discrete_sir_control",
    "RandomDiscrete_SIS": "honet.contagion.random_discrete_sis:random_discrete_sis",
    "RandomDiscrete_SIR": "honet.contagion.random_discrete_sir:random_discrete_sir",
    "RandomGillespie_SIS": "honet.contagion.random_gillespie_sis:random_gillespie_sis",
    "RandomGillespie_SIR": "honet.contagion.random_gillespie_sir:random_gillespie_sir",
    # 旧库私有辅助函数：语义等价，新名 resolve_strategy，旧名保留以便旧代码直接调用
    "_parse_strategy": "honet.contagion._strategies:resolve_strategy",
}

__all__ = sorted(_EXPORTS)

#: 内置模型规格的便捷访问名。这些是**数据对象**而非函数，因此不放进
#: ``__all__``（子包约定 ``__all__`` 里的名字必须可调用，测试会校验）。
#: 需要时用 ``contagion.MODELS["SIER"]`` 或 ``contagion.get_model("SIER")``。
MODELS = ("SIS", "SIR", "SIRS", "SIER", "SEIR", "SI")


def __getattr__(name: str):
    """按需加载实现模块（PEP 562）。

    额外处理两件事：

    1. 本子包的实现模块文件名与公开函数同名（如 ``honet/contagion/hon_sis.py``
       与函数 ``hon_sis``）。Python 在 import 子模块后会把包属性 ``hon_sis``
       指向**模块对象**，遮蔽惰性导出的函数，表现为
       ``TypeError: 'module' object is not callable``。因此在解析出值之后，
       把 ``_EXPORTS`` 中同名属性也一并绑定成同一个对象。
    2. ``MODELS`` 里的模型常量（``contagion.SIER`` 等）按需从
       ``honet.contagion._model`` 取，避免 import 期就构造规格对象。
    """
    if name in MODELS:
        from honet.contagion._model import get_model

        value = get_model(name)
        globals()[name] = value
        return value

    value = lazy_getattr(globals(), __name__, _EXPORTS, name, "analysis")
    canonical = _EXPORTS[name].partition(":")[2]
    if canonical in _EXPORTS:
        globals()[canonical] = value
    return value


class _ContagionPackage(ModuleType):
    """把被同名子模块遮蔽的公开名字还原成函数对象的包类型。

    本子包受「一个文件一个函数、文件名 = 函数名」的约定约束，14 个实现模块
    的文件名都与公开函数同名。Python 在 import 子模块时会把包属性写成模块
    对象，从而遮蔽惰性导出的函数；而 PEP 562 的 ``__getattr__`` 只在属性缺失
    时触发，无法纠正「属性已存在但类型不对」的情况。因此在属性访问层面拦截
    一次：命中被遮蔽的公开名字就返回真正的实现对象。
    """

    def __getattribute__(self, name: str):
        exports = super().__getattribute__("_EXPORTS")
        if name in exports:
            value = super().__getattribute__(name)
            # 被同名实现模块遮蔽时，还原成模块里的目标属性
            module_path, _, attribute = exports[name].partition(":")
            if isinstance(value, ModuleType) and value.__name__ == module_path:
                resolved = getattr(value, attribute)
                super().__setattr__(name, resolved)
                return resolved
        return super().__getattribute__(name)


sys.modules[__name__].__class__ = _ContagionPackage
