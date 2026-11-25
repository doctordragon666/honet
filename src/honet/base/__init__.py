"""honet.base —— 抽象基类与能力 mixin 的公共入口。

仅做 re-export，实现见 :mod:`honet.base.network` 与 :mod:`honet.base.mixins`。
"""

from honet.base.network import HigherOrderNetwork

__all__ = ["HigherOrderNetwork"]
