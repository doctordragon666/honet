"""honet.build —— 超图的构建与随机生成。

本子包的全部符号都是惰性导出的：只有真正访问某个符号时才会 import
对应的实现模块，因此 ``import honet`` 不会因此拉入可选依赖。

两类功能：

* :func:`build_hypernet` / :func:`build_hypernets` —— 由“协议—国家”长表
  构造真实数据超图（可加权、可按年份筛选）；
* :func:`generate_random_hypernet` / :func:`scale_free_hypergraph` /
  :func:`clustered_hypergraph` / :func:`small_world_hypergraph` —— 生成
  随机、无标度、聚簇、小世界四类可控超图，均接受 ``seed`` 保证可复现。

全部实现依赖 hypernetx（``honet[analysis]``），因此本子包挂在 ``analysis``
组下。
"""

from __future__ import annotations

from honet._lazy import lazy_getattr

_EXPORTS = {
    "build_hypernet": "honet.build.build_hypernet:build_hypernet",
    "build_hypernets": "honet.build.build_hypernets:build_hypernets",
    "clustered_hypergraph": "honet.build.clustered_hypergraph:clustered_hypergraph",
    "generate_random_hypernet": "honet.build.generate_random_hypernet:generate_random_hypernet",
    "scale_free_hypergraph": "honet.build.scale_free_hypergraph:scale_free_hypergraph",
    "small_world_hypergraph": "honet.build.small_world_hypergraph:small_world_hypergraph",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """按需加载实现模块（PEP 562）。"""
    return lazy_getattr(globals(), __name__, _EXPORTS, name, "analysis")
