"""honet.base.mixins —— 「加权 / 有向」两种能力 mixin。

每个 mixin 只覆盖一个规范化钩子，因此可与任意形态基类自由组合：
``...W``（加权）、``...D``（有向）、``...WD``（两者叠加）。
"""

from honet.base.mixins.directed import DirectedMixin
from honet.base.mixins.weighted import WeightedMixin

__all__ = ["DirectedMixin", "WeightedMixin"]
