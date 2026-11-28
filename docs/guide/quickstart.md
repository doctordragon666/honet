# 快速开始

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

核心容器零第三方依赖；分析功能按 extras 加载：

| extras | 内容 |
| --- | --- |
| （无） | `Hypergraph` / `Hyperedge` 与四个形态类，零依赖 |
| `analysis` | hypernetx / networkx / numpy / pandas / scipy / scikit-learn / xgi / hypergraphx |
| `viz` | `analysis` + matplotlib + geopandas |
| `excel` | openpyxl，用于 `.xlsx` 往返 |
| `hat` | `analysis` + HypergraphAnalysisToolbox，驱动节点控制需要 |
| `all` | 以上全部 |

```bash
uv add "honet[analysis]"   # 推荐：分析功能
uv add "honet[all]"        # 全装
```

依赖缺失时不会在 `import honet` 阶段报错，只在真正访问对应功能时提示该装哪个 extra：

```python
>>> import honet
>>> honet.metrics.weighted_degree   # 未装 analysis 时
ImportError: honet.metrics 需要可选依赖，但 import honet.metrics.nodemetrics 失败：...
```

## 第一个超图

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

保序存储：结点与超边均按插入顺序返回，复现结果稳定。重复超边会**显式报错**
而非静默覆盖。

## 带标签的超边

同一批成员、不同语义的高阶交互用 `label` 区分：

```python
from honet import Hypergraph

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

## 下一步

- [网络形态](networks.md)：四种形态与 W/D/WD 变体怎么选；
- [分析功能](analysis.md)：度量、社区、传播、攻击、控制、绘图、导出；
- [API 参考](../api/index.md)：逐符号的签名与说明。
