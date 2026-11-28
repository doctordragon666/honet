# honet

[![PyPI](https://img.shields.io/badge/PyPI-honet-blue)](https://pypi.org/project/honet/)
[![Python](https://img.shields.io/badge/python-3.12%2B-3776AB)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)

面向复杂系统的**高阶网络**（higher-order networks）建模工具包。

成对网络（pairwise network）只能描述“两个个体之间的连接”，而真实系统里大量交互是
**多元**的：一篇论文有多位作者、一个反应需要多种反应物、一次会议有多方参与。
`honet` 用**超边**（hyperedge）刻画这类高阶交互，并提供构造、查询与常用度量。

## 特性

- `Hypergraph` / `Hyperedge`：轻量、无第三方运行时依赖，结点可为任意可哈希对象；
- 保序存储：结点与超边均按插入顺序返回，复现结果稳定；
- 自动维护一致性：添加超边时自动补建结点，重复超边显式报错而非静默覆盖；
- 常用度量：关联矩阵、高阶度序列、团投影（高阶 → 成对网络）；
- 完整类型标注（随包分发 `py.typed`），可直接被 mypy / pyright 消费。

## 安装

`honet` 使用 [uv](https://docs.astral.sh/uv/) 管理，所有操作都通过 uv 完成。

作为依赖加入你的项目：

```bash
uv add honet
```

在临时环境中直接试用，不落盘安装：

```bash
uv run --with honet python -c "import honet; print(honet.__version__)"
```

## 快速开始

```python
import honet

# 三元高阶交互 {a, b, c} 与二元交互 {c, d}
hg = honet.Hypergraph(
    nodes=["a", "b", "c", "d"],
    edges=[["a", "b", "c"], ["c", "d"]],
)

print(hg.num_nodes, hg.num_edges)        # 4 2
print(hg.degree("c"))                    # 2：c 参与了两条超边
print(hg.neighbors("a"))                 # frozenset({'b', 'c'})
print(hg.is_uniform)                     # False：边大小 3 与 2 不一致

print(honet.degree_sequence(hg))         # [2, 1, 1, 1]
print(honet.incidence_matrix(hg))        # [[1, 0], [1, 0], [1, 1], [0, 1]]
print(honet.clique_projection(hg)["c"])  # frozenset({'a', 'b', 'd'})

# 导出子超图（原图不受影响）
sub = hg.subhypergraph(["a", "b", "c"])
print(sub.num_edges)                     # 1
```

带标签的超边可区分“同一批成员、不同语义”的高阶交互：

```python
from honet import Hyperedge, Hypergraph

hg = Hypergraph()
hg.add_edge(["alice", "bob"], label="paper-2023")
hg.add_edge(["alice", "bob"], label="paper-2024")   # 合法：标签不同
print(hg.num_edges)                                 # 2
```

## 从 pandas 表格构造

`Hypergraph` 与四种高阶网络都提供 `from_dataframe` 类方法，接受**长表**
（一行 = 某个结点属于某条超边 / 单形 / 事件），列名可自定义，可选权重列：

```python
import pandas as pd
from honet import Hypergraph, HypergraphNetworkW, SimplicialComplexNetwork

df = pd.DataFrame(
    {
        "paper": ["p1", "p1", "p2"],
        "author": ["a", "b", "a"],
        "citations": [3, 3, 1],
    }
)

Hypergraph.from_dataframe(df, edge_col="paper", node_col="author").num_edges   # 2
HypergraphNetworkW.from_dataframe(
    df, edge_col="paper", node_col="author", weight_col="citations"
)
SimplicialComplexNetwork.from_dataframe(
    df, edge_col="paper", node_col="author"
).num_faces   # 面：{a}、{b}、{a,b}
```

多层与时序网络多一个身份列，分别按 `layer` / `time` 分组，因此同一超边标识
在不同层、不同时刻互不合并：

```python
from honet import MultilayerNetwork, TemporalNetwork

layers = pd.DataFrame(
    {
        "layer": ["work", "work", "family"],
        "edge": ["e1", "e1", "e1"],
        "node": ["a", "b", "a"],
    }
)
MultilayerNetwork.from_dataframe(layers).layers        # ('work', 'family')

events = pd.DataFrame(
    {
        "time": [1, 1, 2, 2],
        "edge": ["e1", "e1", "e1", "e1"],
        "node": ["a", "b", "a", "b"],
    }
)
TemporalNetwork.from_dataframe(events).times           # (1.0, 2.0)
```

同一分组内的权重必须一致，标识/结点为空、权重不是实数或超边重复都会抛出明确异常。
pandas 不是必需依赖：按需安装 `honet[pandas]`，核心包运行时不 import pandas。

## API 概览

| 名字                            | 说明                                                                        |
| ------------------------------- | --------------------------------------------------------------------------- |
| `honet.Hypergraph`            | 超图容器：`add_node` / `add_edge` / `remove_node` / `subhypergraph` |
| `honet.Hyperedge`             | 不可变超边：`members`、`label`、`weight`（默认 1.0）                  |
| `honet.incidence_matrix(hg)`  | 关联矩阵 `B`，形状 `num_nodes × num_edges`                             |
| `honet.degree_sequence(hg)`   | 高阶度序列，降序                                                            |
| `honet.clique_projection(hg)` | 团投影：`dict[node, frozenset[neighbor]]`                                 |
| `honet.Node`                  | 结点类型别名（`Hashable`）                                                |

## 类层次：四种形态 × 加权 / 有向变体

除上面的轻量 `Hypergraph` 容器外，`honet` 还提供一套**形态明确**的类层次：
抽象基类 `HigherOrderNetwork` 定义公共接口（`nodes` / `edges` / `num_nodes` /
`num_edges` / `degree` / `incidence_matrix`），不可直接实例化；四个形态基类各自实现，
再分别派生 `W`（加权）、`D`（有向）、`WD`（加权有向）三个变体，共 **16 个具体类**：

```
HigherOrderNetwork (ABC，不可实例化)          honet/base/network.py
├── SimplicialComplexNetwork          单纯复形：按极大单形/面存储，子面自动向下封闭
│   ├── SimplicialComplexNetworkW     加权          honet/simplicial/network.py
│   ├── SimplicialComplexNetworkD     有向（单形顶点有序）
│   └── SimplicialComplexNetworkWD    加权有向
├── HypergraphNetwork                 超图：按超边存储，不要求向下封闭
│   ├── HypergraphNetworkW                         honet/hypergraph/network.py
│   ├── HypergraphNetworkD
│   └── HypergraphNetworkWD
├── MultilayerNetwork                 多层：按“层名 → 超边”存储，结点跨层共享
│   ├── MultilayerNetworkW                         honet/multilayer/network.py
│   ├── MultilayerNetworkD
│   └── MultilayerNetworkWD
└── TemporalNetwork                   时序：按 (时间戳, 事件) 存储，始终按时间有序
    ├── TemporalNetworkW                           honet/temporal/network.py
    ├── TemporalNetworkD
    └── TemporalNetworkWD
```

包内实现按「一个文件只放一个类／一个函数」拆分到 `src/honet/` 下的 `base/`、`core/`、
`simplicial/`、`hypergraph/`、`multilayer/`、`temporal/`、`utils/` 子包。
目录结构与开发约定见[开发指南](https://doctordragon666.github.io/honet/development/index.html)。

加权与有向能力由 `WeightedMixin` / `DirectedMixin` 承载（`is_weighted` / `is_directed`
标记、权重默认 `1.0` 且必须为正、有向成员保留顺序并拒绝自环与重复结点），
但每个变体都是**显式声明的真实类**，没有运行时动态造类。

```python
import honet

# 加权有向超图：成员顺序即方向，权重表达强度
net = honet.HypergraphNetworkWD(edges=[["a", "b", "c"]])
net.add_edge(["c", "d"], weight=2.5)
print(net)                     # <HypergraphNetworkWD nodes=4 ... weighted=True directed=True>
print(net.weights)             # (1.0, 2.5)

# 多层网络：同一批结点在不同层有不同关系
ml = honet.MultilayerNetwork(layers=["work"], nodes=["a", "b", "c"])
ml.add_edge(["a", "b"], layer="work")
ml.add_edge(["a", "c"], layer="family")            # 新层自动注册
print(ml.degree("a"), ml.layer_degree("work", "a"))  # 2 1

# 时序网络：同一组结点在不同时间反复出现
tn = honet.TemporalNetwork(events=[(3.0, ["a", "b"]), (1.0, ["a"])])
print(tn.times, tn.active_nodes(1.0))              # (1.0, 3.0) frozenset({'a'})

# 单纯复形：一个三元面自动拥有全部子面
sc = honet.SimplicialComplexNetwork(facets=[["a", "b", "c"]])
print(sc.num_facets, sc.num_faces, sc.dimension)   # 1 7 2
```

## 示例

`examples/` 下是七个 Jupyter notebook 加一个综合实例脚本，用硬编码的小数据覆盖全部公开 API。
notebook 只是文档，**不随包分发**，也不需要装进项目依赖——按需通过 uv 拉起 jupyter：

```bash
# 打开 JupyterLab 交互浏览（honet 已在项目环境中）
uv run --with jupyter --with ipykernel jupyter lab examples/

# 或在终端里就地执行某个 notebook，并把输出写回文件
uv run --with jupyter --with ipykernel jupyter nbconvert \
  --to notebook --execute --inplace examples/01_hypergraph_basics.ipynb
```

| Notebook | 一句话说明 |
| --- | --- |
| `01_hypergraph_basics.ipynb` | 构造超图、关联矩阵、高阶度、团投影、四种网络形态 |
| `02_metrics.ipynb` | 9 个结构指标 |
| `03_generators_convert.ipynb` | 随机超图生成、格式互转 |
| `04_contagion.ipynb` | SIS/SIR 传播、阈值、相图 |
| `05_control.ipynb` | 可控性、最优控制、牵制选择 |
| `06_community_attack.ipynb` | 社区发现、攻击鲁棒性 |
| `07_io_export_plot.ipynb` | 读写、导出、绘图 |
| `08_case_study_collaboration.py` | 端到端综合实例（脚本，非 notebook） |

## 给 AI 助手

仓库提供两份机器可读的使用说明，可直接作为上下文提供给 LLM 或编码代理：

- [`llms.txt`](https://github.com/doctordragon666/honet/blob/main/llms.txt)：包定位、
  何时该用、安装方式、最小示例、选型表、API 速查与常见陷阱；
- [`AGENTS.md`](https://github.com/doctordragon666/honet/blob/main/AGENTS.md)：给编码代理
  的使用约定（选型、构造顺序、必须注意的语义差异）。

两份文件与真实 API 同步维护；改动公共接口时应一并更新，避免 AI 生成不存在的调用。

## 文档

完整文档（Sphinx 构建，含逐符号 API 参考）在线浏览：

**<https://doctordragon666.github.io/honet/>**

推送 `main` 后由 [`.github/workflows/docs.yml`](.github/workflows/docs.yml) 自动构建并部署；
构建使用 `-W`（警告即错误），文档里的断链、缺符号、坏表格都会让 CI 失败。

本页与 `examples/` 面向**使用者**；构建、测试、发布等**开发者文档**在 `docs/`，
不随包分发：

| 文档 | 面向 | 内容 |
| --- | --- | --- |
| 本页 README | 用户 | 安装、快速开始、API 概览、类层次、示例 |
| [在线文档](https://doctordragon666.github.io/honet/) | 用户 / 开发 | 使用指南、API 参考、开发文档全站 |
| [`examples/`](https://github.com/doctordragon666/honet/tree/main/examples) | 用户 | 七个可执行 notebook + 一个综合实例脚本 |
| [`CHANGELOG.md`](https://github.com/doctordragon666/honet/blob/main/CHANGELOG.md) | 用户 / 开发 | 按 dev / rc / stable 通道记录的版本变更 |

本地构建文档：

```bash
uv sync --group docs --extra all
uv run --group docs sphinx-build -b html docs docs/_build/html -W
```

## 许可证

[GPL-3.0-or-later](LICENSE)（GNU General Public License v3.0 或任何更新版本）。

Copyright (c) 2026 DoctorDragon。你可以自由使用、修改与再分发本软件；
分发衍生作品时需同样以 GPL-3.0-or-later 授权，并提供对应源代码。
