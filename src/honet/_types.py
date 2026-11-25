"""honet._types —— 包内共享的基础类型别名。

放在私有模块里，避免 ``core`` / ``base`` / 各形态子包之间为了一个别名互相导入，
从而杜绝循环导入。

* :data:`Node` —— 结点类型：任何可哈希对象都可以作为结点；
* :data:`Members` —— 一条高阶交互的成员表示：无向为 ``frozenset``，
  有向为保持顺序的 ``tuple``。
"""

from __future__ import annotations

from collections.abc import Hashable

# 结点类型别名：任何可哈希对象都可以作为结点
Node = Hashable

# 一条交互的成员表示：无向为 frozenset，有向为保持顺序的 tuple
Members = frozenset[Node] | tuple[Node, ...]

__all__ = ["Members", "Node"]
