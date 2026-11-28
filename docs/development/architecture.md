# honet 架构

```{image} ../_static/architecture.svg
:alt: honet 架构
:width: 100%
```

上图是重构后的分层结构。三层之间只允许单向依赖：**分析层 → 适配层 → 核心层**，
反向导入会在 review 中直接拒绝。

## ① 核心层：零第三方依赖

`import honet` 只会加载这一层，`pyproject.toml` 的 `dependencies` 保持为空。

| 模块 | 职责 |
| --- | --- |
| `_types.py` | `Node` / `Members` 类型别名 |
| `_frame.py` | DataFrame 长表解析（`from_dataframe` 共用） |
| `base/` | `HigherOrderNetwork` 抽象基类 |
| `base/mixins/` | `WeightedMixin`、`DirectedMixin` |
| `core/` | `Hyperedge`、`Hypergraph` 容器 |
| `simplicial/` | 单纯复形 + `W` / `D` / `WD` 变体 |
| `hypergraph/` | 超图网络 + `W` / `D` / `WD` 变体 |
| `multilayer/` | 多层网络 + `W` / `D` / `WD` 变体 |
| `temporal/` | 时序网络 + `W` / `D` / `WD` 变体 |
| `utils/` | 关联矩阵、度序列、团投影 |

## ② 适配层：可选依赖与版本差异的唯一出口

这一层是本次重构新增的，目的是让上面两层不必关心「某个库装没装」「某个 API
在哪个版本被删了」。

| 模块 | 职责 |
| --- | --- |
| `_lazy.py` | PEP 562 惰性导出。子包只在**被访问时**才 import 实现模块 |
| `_deps.py` | 可选依赖统一导入；缺失时抛出带 `honet[extra]` 安装提示的 `ImportError` |
| `_hnx.py` | hypernetx 1.x / 2.x 兼容层。2.x 删除了 `H.nodes.degree`、`H.edges.members()`、`H.edges.size`、`hnx.bipartite()` 等接口，全部在这里适配 |

## ③ 分析层：10 个子包，158 个导出

这些子包由旧库 `hypertrade` 全量迁入，功能不减。每个子包都通过 `_lazy.py`
惰性导出，因此 `import honet` 不会因为它们的第三方依赖而变重。

| 子包 | 能力 | 导出数 | 需要 |
| --- | --- | --- | --- |
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

导出数以各子包 `__all__` 为准，由 `tools/_make_api.py` 读取；改动后重跑该脚本，
[API 参考](../api/analysis.md) 与本表会一起更新。

## 安装

```bash
uv add honet                    # 只要核心容器
uv add "honet[analysis]"        # 分析功能（推荐）
uv add "honet[viz]"             # 加绘图与地图
uv add "honet[hat]"             # 加驱动节点控制
uv add "honet[all]"             # 全装
```

依赖缺失时不会在 `import honet` 阶段报错，只在真正访问对应功能时提示该装哪个 extra：

```python
>>> import honet
>>> honet.metrics.weighted_degree  # 未装 analysis 时
ImportError: honet 的可选依赖 'numpy' 未安装（需要 honet[analysis]）...
```

## 重新生成这张图

```bash
uv run --with matplotlib python tools/_make_architecture.py
```

脚本读的是仓库真实结构，改动子包后重跑即可同步。
