"""honet —— 面向复杂系统的高阶网络（higher-order network）建模工具包。

高阶网络用“超边”刻画任意数量的个体同时发生的交互，比成对网络更能描述
合作者群体、多元化学反应、多体碰撞、群组社交等真实结构。

公开 API 一览：

============= ==========================================================
名字           说明
============= ==========================================================
Hyperedge     超边：若干结点的无序集合 + 可选标签 + 权重
Hypergraph    超图容器：构造、增删、度/邻居/子图查询
Node          结点类型别名（任意 ``Hashable``）
incidence_matrix    关联矩阵 ``B``（结点 × 超边）
degree_sequence     高阶度序列（降序）
clique_projection   团投影（高阶 → 成对网络）
HigherOrderNetwork  四类网络形态的抽象基类（不可实例化）
WeightedMixin / DirectedMixin  加权 / 有向能力 mixin
============= ==========================================================

四类网络形态各占一个子包，每类再派生 ``W`` / ``D`` / ``WD`` 三个变体，
共 16 个具体类：

* ``SimplicialComplexNetwork`` —— 单纯复形（``honet.simplicial``，按极大单形/面存储，向下封闭）；
* ``HypergraphNetwork`` —— 超图（``honet.hypergraph``，按超边存储，不向下封闭）；
* ``MultilayerNetwork`` —— 多层网络（``honet.multilayer``，按层存储，跨层共享结点）；
* ``TemporalNetwork`` —— 时序网络（``honet.temporal``，按 (时间戳, 事件) 存储，按时间有序）。

内部模块结构（每个模块只放一个类／一个函数）::

    honet/
    ├── _types.py                  Node / Members 类型别名
    ├── base/network.py            HigherOrderNetwork
    ├── base/mixins/weighted.py    WeightedMixin
    ├── base/mixins/directed.py    DirectedMixin
    ├── core/hyperedge.py          Hyperedge
    ├── core/hypergraph.py         Hypergraph
    ├── simplicial/network.py      单纯复形基类 + W/D/WD
    ├── hypergraph/network.py      超图基类 + W/D/WD
    ├── multilayer/network.py      多层基类 + W/D/WD
    ├── temporal/network.py        时序基类 + W/D/WD
    └── utils/*.py                 每个度量函数一个模块

最小用法::

    import honet

    hg = honet.Hypergraph(nodes=["a", "b", "c"], edges=[["a", "b", "c"], ["b", "c"]])
    print(hg.num_nodes, hg.num_edges)       # 3 2
    print(hg.degree("b"))                   # 2
    print(honet.degree_sequence(hg))        # [2, 2, 1]
    print(honet.incidence_matrix(hg))

    net = honet.HypergraphNetworkWD(edges=[["a", "b"]])
    net.add_edge(["b", "c"], weight=2.5)
    print(net, net.weights)                 # 加权有向超图

本文件被 ``tests/`` 下的测试与使用方的 ``import honet`` 语句依赖，
对外契约由 ``__all__`` 维护：新增公开名字必须同时加入 ``__all__``。
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from honet.base import HigherOrderNetwork
from honet.base.mixins import DirectedMixin, WeightedMixin
from honet.core import Hyperedge, Hypergraph, Node
from honet.hypergraph import (
    HypergraphNetwork,
    HypergraphNetworkD,
    HypergraphNetworkW,
    HypergraphNetworkWD,
)
from honet.multilayer import (
    MultilayerNetwork,
    MultilayerNetworkD,
    MultilayerNetworkW,
    MultilayerNetworkWD,
)
from honet.simplicial import (
    SimplicialComplexNetwork,
    SimplicialComplexNetworkD,
    SimplicialComplexNetworkW,
    SimplicialComplexNetworkWD,
)
from honet.temporal import (
    TemporalNetwork,
    TemporalNetworkD,
    TemporalNetworkW,
    TemporalNetworkWD,
)
from honet.utils import clique_projection, degree_sequence, incidence_matrix

# 包版本号的唯一来源是 pyproject.toml 的 [project].version。
# 本模块不硬编码版本，而是在运行期从已安装分发包的元数据中读取，
# 因此改版本时只需改 pyproject.toml 一处（可用 `uv version` 命令），
# 不存在“两处版本不一致”的问题。
try:
    # 正常路径：uv sync / uv run 会把本项目以可编辑模式装进虚拟环境，元数据可用
    __version__ = _dist_version("honet")
except PackageNotFoundError:  # pragma: no cover - 仅在未安装的裸源码树中出现
    # 兜底值：表明“元数据不可用”，避免 import 直接失败；构建/安装后不会是此值
    __version__ = "0.0.0+unknown"

# 公开 API 白名单：``from honet import *`` 只会导入这里列出的名字
__all__ = [
    "DirectedMixin",
    "HigherOrderNetwork",
    "Hyperedge",
    "Hypergraph",
    "HypergraphNetwork",
    "HypergraphNetworkD",
    "HypergraphNetworkW",
    "HypergraphNetworkWD",
    "MultilayerNetwork",
    "MultilayerNetworkD",
    "MultilayerNetworkW",
    "MultilayerNetworkWD",
    "Node",
    "SimplicialComplexNetwork",
    "SimplicialComplexNetworkD",
    "SimplicialComplexNetworkW",
    "SimplicialComplexNetworkWD",
    "TemporalNetwork",
    "TemporalNetworkD",
    "TemporalNetworkW",
    "TemporalNetworkWD",
    "WeightedMixin",
    "__version__",
    "clique_projection",
    "degree_sequence",
    "incidence_matrix",
]
