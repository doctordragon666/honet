"""仓室模型规格：用声明式数据描述传播模型，而不是用布尔量分叉。

## 为什么需要这一层

旧实现用 ``model: str`` 加大量 ``is_sir = model == "SIR"`` 布尔判断来区分
SIS 与 SIR。这种做法在只有两个模型时勉强可用，但每加一个模型（SIRS、SIER、
SEIR……）都要在所有分叉点再补一个分支，改一处漏一处。

这里把「模型」抽象成一份**声明式规格**：

* 有哪些仓室（compartment）、各自的初始状态名；
* 每个仓室在「被感染 / 被恢复」时迁移到哪个仓室；
* 各仓室是否参与感染力统计；
* 返回值里要不要输出该仓室的时间序列。

引擎只读规格、不再判断具体模型名，因此新增模型只需**注册一份规格**，
无需改动任何引擎代码。

## 命名约定

状态名沿用流行病学惯例：``S`` 易感、``E`` 暴露（已感染但尚无传染性）、
``I`` 感染、``R`` 恢复。``E`` 与 ``R`` 都属于「已离开 S」的状态。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: 流行病学惯例中的标准仓室名（仅用于文档与提示，不做强制限制）
CONVENTIONAL_STATES = ("S", "E", "I", "R")

#: 状态名必须是「单个大写字母」或「大写字母开头的字母数字串」，避免与
#: 超图结点名混淆，同时允许 ``D``（接种）、``Q``（隔离）、``V``（疫苗）等
#: 领域自定义仓室。
_STATE_PATTERN = re.compile(r"^[A-Z][A-Za-z0-9]*$")


@dataclass(frozen=True)
class CompartmentModel:
    """一个仓室模型的声明式规格。

    属性:
        name: 模型名，如 ``"SIS"`` / ``"SIR"`` / ``"SIER"``。
        states: 全部仓室状态，按输出列顺序排列，如 ``("S", "I", "R")``。
        susceptible: 易感状态名。
        infected: 具有传染性的状态名集合（感染力的统计来源）。
        on_infection: ``{源状态: 感染后状态}``，例如 SIR 的 ``{"S": "I"}``，
            SEIR 的 ``{"S": "E"}``。
        on_recovery: ``{源状态: 恢复后状态}``。SIS 里 ``"I" -> "S"``，
            SIR 里 ``"I" -> "R"``。
        initial: ``{状态: 初始人数表达式}``，只用于文档说明；实际初始状态由
            调用方传入的 ``initial_infecteds`` / ``initial_recovereds`` 决定。
        tracked: 需要输出时间序列的状态（通常是 ``states`` 的全部）。
        description: 一句话说明，用于生成文档。

    说明:
        ``on_infection`` / ``on_recovery`` 合起来定义了状态转移图。引擎按这张
        图迁移，因此支持任意链式模型（S→E→I→R 只是其中一条路径）。
    """

    name: str
    states: tuple[str, ...]
    susceptible: str = "S"
    infected: tuple[str, ...] = ("I",)
    on_infection: dict[str, str] = field(default_factory=lambda: {"S": "I"})
    on_recovery: dict[str, str] = field(default_factory=lambda: {"I": "R"})
    initial: dict[str, str] = field(default_factory=dict)
    tracked: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        """校验规格自洽性：状态合法、转移目标存在、至少有一个传染态。"""
        if not self.name:
            msg = "CompartmentModel.name 不能为空"
            raise ValueError(msg)

        unknown = [state for state in self.states if not _STATE_PATTERN.match(state)]
        if unknown:
            msg = (
                f"模型 {self.name} 含非法状态名 {unknown}；"
                f"状态名须以大写字母开头（惯例用 {CONVENTIONAL_STATES}，"
                f"也可自定义如 D=接种、Q=隔离、V=疫苗）"
            )
            raise ValueError(msg)

        if not self.states:
            msg = f"模型 {self.name} 必须至少声明一个仓室"
            raise ValueError(msg)

        if self.susceptible not in self.states:
            msg = f"模型 {self.name} 的 susceptible={self.susceptible!r} 不在 states 中"
            raise ValueError(msg)

        if not self.infected:
            msg = f"模型 {self.name} 必须至少声明一个传染态"
            raise ValueError(msg)
        missing = [state for state in self.infected if state not in self.states]
        if missing:
            msg = f"模型 {self.name} 的传染态 {missing} 不在 states 中"
            raise ValueError(msg)

        for source, target in {**self.on_infection, **self.on_recovery}.items():
            if source not in self.states:
                msg = f"模型 {self.name} 的转移源 {source!r} 不在 states 中"
                raise ValueError(msg)
            if target not in self.states:
                msg = f"模型 {self.name} 的转移目标 {target!r} 不在 states 中"
                raise ValueError(msg)

    @property
    def is_sir_like(self) -> bool:
        """是否存在**无法回到易感态**的吸收态（即 SIR 型而非 SIS/SIRS 型）。

        判据是可达性：从该恢复目标出发，沿 ``on_infection`` / ``on_recovery``
        的转移图能否回到 ``susceptible``。能回到（如 SIS 的 ``I -> S``、SIRS 的
        ``I -> R -> S``）就是循环型；回不去（如 SIR 的 ``R``）才是吸收态。
        """
        targets = set(self.on_recovery.values())
        if not targets:
            # 没有任何恢复转移（如 SI）：感染态是吸收态
            return True
        return any(not self._can_reach_susceptible(target) for target in targets)

    def _can_reach_susceptible(self, start: str) -> bool:
        """从 ``start`` 沿转移图能否到达易感态（广度优先，含起点判断）。"""
        if start == self.susceptible:
            return True
        seen = {start}
        queue = [start]
        while queue:
            state = queue.pop()
            for target in (self.on_infection.get(state), self.on_recovery.get(state)):
                if target is None or target in seen:
                    continue
                if target == self.susceptible:
                    return True
                seen.add(target)
                queue.append(target)
        return False

    def target_of_infection(self, state: str) -> str | None:
        """返回 ``state`` 被感染后迁移到的状态；该状态不可被感染时返回 ``None``。"""
        return self.on_infection.get(state)

    def target_of_recovery(self, state: str) -> str | None:
        """返回 ``state`` 恢复后迁移到的状态；该状态不会恢复时返回 ``None``。"""
        return self.on_recovery.get(state)

    def can_be_infected(self, state: str) -> bool:
        """该状态的结点是否可能被感染。"""
        return state in self.on_infection

    def can_recover(self, state: str) -> bool:
        """该状态的结点是否可能恢复。"""
        return state in self.on_recovery

    def output_states(self) -> tuple[str, ...]:
        """返回需要输出时间序列的状态（未显式指定 ``tracked`` 时取全部）。"""
        return self.tracked or self.states


def register_model(model: CompartmentModel, *, overwrite: bool = False) -> None:
    """把模型规格注册到全局表，供 ``model="SIER"`` 这类字符串查找。

    参数:
        model: 待注册的模型规格。
        overwrite: 已存在同名模型时是否覆盖。

    抛出:
        ValueError: 同名模型已存在且 ``overwrite=False``。
    """
    if model.name in _REGISTRY and not overwrite:
        msg = f"模型 {model.name!r} 已注册；如需覆盖请传 overwrite=True"
        raise ValueError(msg)
    _REGISTRY[model.name] = model


def get_model(model: str | CompartmentModel) -> CompartmentModel:
    """按名字或直接给定规格取得模型。

    参数:
        model: 模型名（须已注册）或 :class:`CompartmentModel` 实例。

    返回:
        对应的模型规格。

    抛出:
        ValueError: 名字未注册，并列出全部已注册模型名。
    """
    if isinstance(model, CompartmentModel):
        return model
    try:
        return _REGISTRY[model]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY))
        msg = f"未知的 model：{model!r}；已注册的有 {available}"
        raise ValueError(msg) from None


def available_models() -> tuple[str, ...]:
    """返回全部已注册的模型名（排序）。"""
    return tuple(sorted(_REGISTRY))


# ---------------------------------------------------------------------------
# 内置模型
# ---------------------------------------------------------------------------

#: SIS：感染后康复回到易感，可反复感染
SIS = CompartmentModel(
    name="SIS",
    states=("S", "I"),
    infected=("I",),
    on_infection={"S": "I"},
    on_recovery={"I": "S"},
    initial={"S": "全部结点 - 初始感染者", "I": "初始感染者"},
    description="易感—感染—易感：康复后不产生免疫，可反复感染。",
)

#: SIR：康复后获得永久免疫
SIR = CompartmentModel(
    name="SIR",
    states=("S", "I", "R"),
    infected=("I",),
    on_infection={"S": "I"},
    on_recovery={"I": "R"},
    initial={"S": "全部结点 - 初始感染者 - 受控结点", "I": "初始感染者", "R": "受控结点"},
    description="易感—感染—恢复：康复后获得永久免疫，不再被感染。",
)

#: SIRS：康复后免疫会失效，重新回到易感
SIRS = CompartmentModel(
    name="SIRS",
    states=("S", "I", "R"),
    infected=("I",),
    on_infection={"S": "I"},
    # 注意 R 也会迁移回 S，因此恢复图里同时含 I→R 与 R→S
    on_recovery={"I": "R", "R": "S"},
    initial={"S": "全部结点 - 初始感染者", "I": "初始感染者", "R": "初始恢复者"},
    description="易感—感染—恢复—易感：免疫有时效，康复者会重新变为易感。",
)

#: SIER：感染后先进入暴露期（无传染性），再转为有传染性
SIER = CompartmentModel(
    name="SIER",
    states=("S", "E", "I", "R"),
    infected=("I",),
    # S 被感染后先进入 E（暴露），E 再进展为 I
    on_infection={"S": "E", "E": "I"},
    on_recovery={"I": "R"},
    initial={
        "S": "全部结点 - 初始感染者 - 受控结点",
        "E": "初始暴露者",
        "I": "初始感染者",
        "R": "受控结点",
    },
    description="易感—暴露—感染—恢复：感染后先经历无传染性的潜伏期，再转为有传染性。",
)

#: SEIR：与 SIER 同构，只是命名顺序不同（暴露在感染之前）
SEIR = CompartmentModel(
    name="SEIR",
    states=("S", "E", "I", "R"),
    infected=("I",),
    on_infection={"S": "E", "E": "I"},
    on_recovery={"I": "R"},
    initial={"S": "易感", "E": "暴露", "I": "感染", "R": "恢复"},
    description="易感—暴露—感染—恢复：与 SIER 同构，暴露期先于感染期。",
)

#: SI：只有易感与感染，感染后永不恢复（用于研究纯扩散过程）
SI = CompartmentModel(
    name="SI",
    states=("S", "I"),
    infected=("I",),
    on_infection={"S": "I"},
    on_recovery={},
    initial={"S": "全部结点 - 初始感染者", "I": "初始感染者"},
    description="易感—感染：感染后不恢复，用于研究纯扩散过程。",
)


_REGISTRY: dict[str, CompartmentModel] = {}
for _model in (SIS, SIR, SIRS, SIER, SEIR, SI):
    register_model(_model)

#: 兼容旧代码的模型名（旧实现只认 SIS / SIR）
_ALIASES = {"SIS": SIS, "SIR": SIR}


def resolve_model(model: Any) -> CompartmentModel:
    """把旧接口里的 ``model`` 参数（字符串或规格）解析成规格。

    与 :func:`get_model` 的区别：本函数额外接受 ``None`` （按 SIS 处理），
    便于旧代码在未显式传模型时保持原有行为。
    """
    if model is None:
        return SIS
    return get_model(model)
