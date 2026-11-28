# honet 文档

面向复杂系统的**高阶网络**（higher-order networks）建模工具包。

成对网络只能描述「两个个体之间的连接」，而真实系统里大量交互是**多元**的：
一篇论文有多位作者、一个反应需要多种反应物、一次会议有多方参与。`honet` 用
**超边**（hyperedge）刻画这类高阶交互，并提供构造、查询、度量、传播、控制与
可视化能力。

## 两条使用路径

| 我想… | 从这里开始 |
| --- | --- |
| **用** honet 做分析 | [快速开始](guide/quickstart.md) → [网络形态](guide/networks.md) → [分析功能](guide/analysis.md) |
| **改** honet 源码 | [开发指南](development/index.md) → [架构](development/architecture.md) → [性能设计](development/performance.md) |

## 安装

`honet` 核心容器**零第三方依赖**，分析功能按 extras 逐步加载：

```bash
uv add honet                    # 只要核心容器
uv add "honet[analysis]"        # 分析功能（推荐）
uv add "honet[viz]"             # 加绘图与地图
uv add "honet[hat]"             # 加驱动节点控制
uv add "honet[all]"             # 全装
```

## 最小示例

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
print(honet.degree_sequence(hg))         # [2, 1, 1, 1]
```

```{toctree}
:maxdepth: 2
:caption: 使用指南

guide/index
guide/quickstart
guide/networks
guide/analysis
guide/examples
```

```{toctree}
:maxdepth: 2
:caption: API 参考

api/index
```

```{toctree}
:maxdepth: 2
:caption: 开发文档

development/index
development/architecture
development/performance
development/contagion
development/control
development/publishing
```

```{toctree}
:maxdepth: 1
:caption: 其他

changelog
```
