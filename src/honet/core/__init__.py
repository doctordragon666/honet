"""honet.core —— 轻量超图容器：:class:`Hyperedge` 与 :class:`Hypergraph`。

两个类各自独立成模块（``hyperedge.py`` / ``hypergraph.py``），本文件只做
re-export，保证 ``from honet.core import Hyperedge, Hypergraph, Node`` 依旧可用。
"""

from honet._types import Node
from honet.core.hyperedge import Hyperedge
from honet.core.hypergraph import EdgeKey, Hypergraph

__all__ = ["EdgeKey", "Hyperedge", "Hypergraph", "Node"]
