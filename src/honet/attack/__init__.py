"""honet.attack —— 超图与普通图的攻击鲁棒性分析。

本子包提供「生成攻击序列 → 按比例逐步移除结点/超边 → 记录度量曲线」这条
完整链路，包含随机、基于度（含余平均度与聚类系数）、基于中心性三类序列，
以及面向普通 NetworkX 图的批量序列生成。

本子包的全部符号都是惰性导出的：只有真正访问某个符号时才会 import
对应的实现模块，因此 ``import honet`` 不会因此拉入可选依赖。
"""

from __future__ import annotations

from honet._lazy import lazy_getattr

_EXPORTS = {
    "basic_attack_sequence": "honet.attack.basic_attack_sequence:basic_attack_sequence",
    "centrality_attack_sequence": (
        "honet.attack.centrality_attack_sequence:centrality_attack_sequence"
    ),
    "degree_attack_sequence": "honet.attack.degree_attack_sequence:degree_attack_sequence",
    "execute_attack": "honet.attack.execute_attack:execute_attack",
    "main_country": "honet.attack.main_country:main_country",
    "random_attack_sequence": "honet.attack.random_attack_sequence:random_attack_sequence",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """按需加载实现模块（PEP 562）。"""
    return lazy_getattr(globals(), __name__, _EXPORTS, name, "analysis")
