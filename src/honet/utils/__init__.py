"""honet.utils —— 高阶网络的常用度量工具函数。

每个函数独立成模块，本文件只做 re-export。所有函数都只做“读”操作，
不修改传入的超图，返回值均为纯 Python 内建结构，便于直接喂给
``numpy`` / ``pandas`` / ``networkx`` 等下游库。
"""

from honet.utils.clique_projection import clique_projection
from honet.utils.degree_sequence import degree_sequence
from honet.utils.incidence_matrix import incidence_matrix

__all__ = ["clique_projection", "degree_sequence", "incidence_matrix"]
