# honet 示例

本目录按**用途**分成两部分：想快速查某个 API 怎么用，看第一部分；想看一个完整
分析流程怎么走，看第二部分。

```
examples/
├── 01_hypergraph_basics.ipynb       ← 第一部分：单 API 演示（notebook）
├── 02_metrics.ipynb
├── 03_generators_convert.ipynb
├── 04_contagion.ipynb
├── 05_control.ipynb
├── 06_case_study_collaboration.py   ← 第二部分：综合实例（脚本）
└── 06_case_study_result.png
```

---

## 第一部分 · 单 API 示例（notebook）

每个 notebook 覆盖一个子包，用可执行代码回答「这个 API 做什么、怎么调、有什么坑」。
**所有输出都是真实执行结果**，不是手写的示意。

| Notebook | 覆盖内容 | 关键 API |
| --- | --- | --- |
| [`01_hypergraph_basics.ipynb`](01_hypergraph_basics.ipynb) | 构造超图、关联矩阵、高阶度、团投影 | `Hypergraph` / `Hyperedge` / `incidence_matrix` / `degree_sequence` / `clique_projection` |
| [`02_metrics.ipynb`](02_metrics.ipynb) | 9 个结构指标 | `honet.metrics.*`（`calc_entropy` / `calc_reducibility` / `calc_natural_connectivity` / …） |
| [`03_generators_convert.ipynb`](03_generators_convert.ipynb) | 随机超图生成、格式互转 | `honet.build.*` / `honet.convert.*` |
| [`04_contagion.ipynb`](04_contagion.ipynb) | SIS/SIR 传播、阈值、相图 | `random_discrete_sis` / `simulate` / `sis_phase` / `generate_tau` |
| [`05_control.ipynb`](05_control.ipynb) | 可控性、最优控制、牵制选择 | `hypergraph_laplacian` / `minimum_energy_control` / `average_controllability` / `rank_simplices` |

### ⚠️ 最容易踩的两个坑

这两个坑在 notebook 里都有专门演示，但值得在这里先说明：

**1. 超图对象有两种，不能混用**

| 类型 | 由谁产生 | 能喂给谁 |
| --- | --- | --- |
| `honet.Hypergraph` | 自己 `honet.Hypergraph(...)` 构造 | `incidence_matrix` / `degree_sequence` / `clique_projection` |
| `hypernetx.Hypergraph` | 全部 `honet.build.*` 生成器 | `honet.metrics.*` / `honet.contagion.*` |

目前**没有**官方的 `hypernetx → honet.Hypergraph` 转换器。需要时手动重建，
注意去重（hypernetx 允许重复成员集，`honet.Hypergraph` 会拒绝）：

```python
honet.Hypergraph(edges=[sorted(H.edges[e]) for e in H.edges])
```

**2. `simulate` 与 `random_discrete_sis` 返回的东西不一样**

| 入口 | 返回 | 适合 |
| --- | --- | --- |
| `random_discrete_sis(...)` | `(t, S, I)`，**逐时间点**计数 | 画传播曲线、算峰值 |
| `simulate(...)` | 元组，每个元素是**事件序列** | 事件级分析 |

想画曲线却用了 `simulate`，会得到只有几个点的「曲线」——那不是 bug，
是事件驱动的本质。详见 [`04_contagion.ipynb`](04_contagion.ipynb)。

---

## 第二部分 · 综合实例（脚本）

[`06_case_study_collaboration.py`](06_case_study_collaboration.py) 是一个**端到端**
分析，回答一个具体问题：

> 一个 13 人的科研协作网络里，如果只能做一次干预，应该选谁？

它分六步串起 `honet` 的主要能力，每步都说明**为什么这么做**而不只是**怎么做**：

1. 构造：从长表建超图，带结构自检
2. 拓扑画像：枢纽（高加权度）vs 桥（结构洞）
3. 高阶结构：度分布、熵、可约性
4. 传播风险：从每个结点出发的 SIS 模拟
5. 控制入口：Laplacian 动力学 + 可控性分析
6. 综合结论：多指标交叉验证

运行：

```bash
python examples/06_case_study_collaboration.py
```

依赖 `numpy` + `hypernetx`（即 `honet[analysis]`）。绘图可选，没装 matplotlib
会自动跳过，不影响分析结果。

### 这个例子最想传达的一点

**中心性高 ≠ 适合干预。** 脚本的输出会明确显示：

- 最活跃的人（加权度最高）
- 位置最不可替代的人（桥接分数最高）
- 传播风险最高的人
- 最佳控制入口（平均可控性最高）

这四个**往往是不同的人**。而且脚本会指出平均可控性存在**并列**
（结构等价的叶子结点算出相同的值），提醒你不能只看单一指标。

图：`06_case_study_result.png`（四个视角并排对比）

---

## 重新生成 notebook

Notebook 里的输出是执行产生的真实结果，由
[`docs/_make_examples.py`](../docs/_make_examples.py) 生成：

```bash
# 全部重新生成
python docs/_make_examples.py

# 只重新生成 04
python docs/_make_examples.py 04
```

生成器会读取 [`docs/_example_sources/`](../docs/_example_sources/) 下的 `.py`
描述文件（用 `# %%` / `# %% [markdown]` 标记 cell），真实执行每个 cell，
把输出和 `execution_count` 一起写进 `.ipynb`。

**为什么不用手写 notebook？** 因为手写的输出会过时、会骗人。这里
`allow_errors=False`，任何一个 cell 报错都会中断生成——这保证了
notebook 里记录的 API 用法**真的能跑通**。

要改示例，改 `docs/_example_sources/*.py`，然后重新生成。别直接编辑 `.ipynb`，
下次生成会覆盖。

---

## 环境要求

```bash
uv sync --extra analysis    # 跑全部 notebook 和脚本
```

- `01`：零依赖（仅 `honet` 本身）
- `02` / `03` / `04`：需要 `honet[analysis]`（hypernetx、numpy 等）
- `05`：需要 `numpy`（scipy 可选）
- `06`：需要 `numpy` + `hypernetx`，matplotlib 可选
