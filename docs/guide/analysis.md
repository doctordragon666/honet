# 分析功能

`honet` 的十个分析子包都由旧库 `hypertrade` 全量迁入，功能不减。每个子包都通过
`honet._lazy` 惰性导出，因此 `import honet` 不会因为它们的第三方依赖而变重。

| 子包 | 能力 | 导出数 | 需要 |
| --- | --- | ---: | --- |
| `io` | 数据加载与合并 | 5 | `analysis` |
| `convert` | hypernetx / xgi / hypergraphx / NetworkX / HAT 互转 | 8 | `analysis` |
| `build` | 超图构建与随机生成（随机、无标度、聚类、小世界） | 6 | `analysis` |
| `metrics` | 结点 / 超边 / 全局指标 | 16 | `analysis` |
| `community` | 社区发现（Random / Basic / NMF / Recursive / Hierarchy / Kumar） | 15 | `analysis` |
| `contagion` | 传播动力学 SIS / SIR（离散、随机、Gillespie、相图） | 30 | `analysis` |
| `attack` | 攻击鲁棒性（随机 / 度 / 中心性 / 目标） | 6 | `analysis` |
| `control` | 驱动节点控制（dilation、可达性、贪心）+ 控制律设计 | 54 | `hat` / `analysis` |
| `plot` | 绘图（SIS/SIR 曲线、社区、地图、攻击结果） | 13 | `viz` |
| `export` | 导出 LaTeX / CSV / Excel | 5 | `analysis` |

## 两类超图

这是使用 honet 时最容易踩的坑：

| 类型 | 来源 | 谁接收它 |
| --- | --- | --- |
| `honet.Hypergraph` | `honet` 自己的轻量容器 | `honet.utils` 的 `incidence_matrix` / `degree_sequence` / `clique_projection` |
| `hypernetx.Hypergraph` | `honet.build.*` 的产出 | `metrics` / `community` / `contagion` / `attack` / `control` 的分析函数 |

两者同名但不同物。跨用时用 `honet.convert.convert_hypernetx_to_honet` 桥接：
它迁移边名与边级权重，并对 hypernetx 允许的重复成员集做合并。

```python
import hypernetx as hnx
from honet.convert import convert_hypernetx_to_honet

H = hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["c", "d"]})
hg = convert_hypernetx_to_honet(H)      # honet.Hypergraph，保留边名与权重
print(hg.num_nodes, hg.num_edges)       # 4 2
```

## metrics：按粒度分文件

`honet.metrics` 按「返回什么」组织，而不是一个指标一个文件：

| 模块 | 返回形态 | 函数 |
| --- | --- | --- |
| `nodemetrics` | `{结点: 分值}` | `weighted_degree`、`clustering_coefficient`、`excess_average_degree`、`eigenvector_centrality`、`betweenness_centrality`、`temporal_closeness_centrality` |
| `edgemetrics` | `{分组: 值}` | `get_avg_degree_by_edge_size` |
| `globalmetrics` | 标量或汇总表 | `calc_entropy`、`calc_information_gain`、`calc_clustering_coefficient_g`、`calc_natural_connectivity`、`assortativity_test`、`calc_assortativity`、`calc_reducibility`、`sub_reducibility`、`hnxs_global` |

```{note}
`temporal_closeness_centrality` 是唯一接收 `honet.TemporalNetwork` 的结点级指标，
其余结点级指标接收 hypernetx 超图。时序语义与静态语义不同，函数入口用
`isinstance` 显式校验，静默接受会造成误读。
```

可约性两个函数（`calc_reducibility` / `sub_reducibility`）与同配系数的零模型
需要 `hypergraphx`，其余只需 `honet[analysis]`。

## contagion：规格驱动

传播模型被抽象成**声明式规格**，引擎只读规格、不判断模型名。新增模型 = 注册一份规格，
不需要碰引擎代码。详见[传播模型](../development/contagion.md)。

```python
from honet.contagion import CompartmentModel, register_model, simulate

register_model(CompartmentModel(
    name="SIQD",
    states=("S", "I", "Q", "D"),
    infected=("I",),
    on_infection={"S": "I"},
    on_recovery={"I": "Q", "Q": "D", "D": "S"},
))

result = simulate(H, gamma=0.3, tau={2: 0.5}, model="SIQD", tmax=10)
```

内置模型：`SIS`、`SIR`、`SIRS`、`SIER`、`SEIR`、`SI`。`available_models()` 返回当前
已注册的全部模型名。

## control：两层能力

`honet.control` 有两层，依赖不同：

| 层 | 回答的问题 | 依赖 |
| --- | --- | --- |
| 拓扑层 | 控制**哪些**结点：dilation 检测、可达性、最小驱动集 | `honet[hat]`（HAT） |
| 动力学层 | **怎么**控制：可控性判据、最小能量控制律、稳态与最优控制 | numpy / scipy |

动力学层把「结点自身动力学 + 高阶耦合 + 外部输入」组装成线性系统
`Ẋ = 𝒜X + ℬU`，其中 `𝒜 = I_N ⊗ A + c/(k-1) · L ⊗ Λ`，`ℬ = Δ ⊗ B`。
完整推导、论文复现与已知边界见[控制框架](../development/control.md)。

```{warning}
拓扑层的驱动节点一律是 `0..nnodes-1` 的整数下标（HAT 内部把结点编号成整数），
原始结点名用 `HG.nodes["Names"]` 反查；动力学层的结点下标则是
`HypergraphControlSystem.L` 的行序。两套下标不要混用。
```

## plot 与 export

`plot` 需要 `honet[viz]`（matplotlib，地图相关还需 geopandas）；`export` 的 `.xlsx`
往返需要 `honet[excel]`（openpyxl）。`to_latex` 输出可直接粘进论文的表格。

## 惰性导出的含义

子包本身在 `import honet` 时**不会**被加载，第一次访问属性才 import：

```python
>>> import honet
>>> "numpy" in sys.modules          # False：核心容器零依赖
>>> honet.metrics                    # 此刻才 import honet.metrics
>>> honet.metrics.weighted_degree    # 此刻才 import 实现模块与 numpy
```

因此「装了什么」与「能不能 import」解耦：缺依赖只在真正用到时报错，错误信息里
直接给出该装的 extra。
