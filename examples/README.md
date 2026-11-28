# honet 示例

本目录按**用途**分成两部分：想查某个 API 怎么用，看第一部分；想看一个完整分析
流程怎么走，看第二部分。

```
examples/
├── 01_hypergraph_basics.ipynb       ┐
├── 02_metrics.ipynb                 │
├── 03_generators_convert.ipynb      │ 第一部分：单 API 演示（notebook）
├── 04_contagion.ipynb               │
├── 05_control.ipynb                 │
├── 06_community_attack.ipynb        │
├── 07_io_export_plot.ipynb          ┘
├── 08_case_study_collaboration.py   ← 第二部分：综合实例（脚本）
└── 08_case_study_result.png
```

---

## 第一部分 · 单 API 示例（notebook）

每个 notebook 覆盖一个子包，用可执行代码回答「这个 API 做什么、怎么调、有什么坑」。
**所有输出都是真实执行结果**，不是手写的示意。

| Notebook | 覆盖内容 | 关键 API |
| --- | --- | --- |
| [`01_hypergraph_basics.ipynb`](01_hypergraph_basics.ipynb) | 构造超图、关联矩阵、高阶度、团投影、四种网络形态 | `Hypergraph` / `Hyperedge` / `incidence_matrix` / `degree_sequence` / `clique_projection` |
| [`02_metrics.ipynb`](02_metrics.ipynb) | 9 个结构指标 | `calc_entropy` / `calc_reducibility` / `calc_natural_connectivity` / `betweenness_centrality` / … |
| [`03_generators_convert.ipynb`](03_generators_convert.ipynb) | 随机超图生成、格式互转 | `build.*` / `convert.*` |
| [`04_contagion.ipynb`](04_contagion.ipynb) | SIS/SIR 传播、阈值、相图 | `random_discrete_sis` / `simulate` / `sis_phase` / `generate_tau` |
| [`05_control.ipynb`](05_control.ipynb) | 可控性、最优控制、牵制选择 | `hypergraph_laplacian` / `minimum_energy_control` / `average_controllability` / `rank_simplices` |
| [`06_community_attack.ipynb`](06_community_attack.ipynb) | 社区发现、攻击鲁棒性 | `basic_method` / `extract_community_partition` / `execute_attack` / `degree_attack_sequence` |
| [`07_io_export_plot.ipynb`](07_io_export_plot.ipynb) | 读写、导出、绘图 | `io.*` / `export.*` / `plot.*` |

### ⚠️ 最容易踩的坑

这些坑在每个 notebook 里都有演示，先在这里集中列出：

**1. 超图对象有两种，不能混用**

| 类型 | 由谁产生 | 能喂给谁 |
| --- | --- | --- |
| `honet.Hypergraph` | 自己 `honet.Hypergraph(...)` 构造 | `incidence_matrix` / `degree_sequence` / `clique_projection` |
| `hypernetx.Hypergraph` | 全部 `honet.build.*` 生成器 | `honet.metrics.*` / `honet.contagion.*` / `honet.community.*` / `honet.attack.*` |

用 `honet.convert.convert_hypernetx_to_honet` 桥接，它会迁移边名与边级权重，
并合并 hypernetx 允许的重复成员集（`honet.Hypergraph` 会拒绝重复超边）：

```python
from honet.convert import convert_hypernetx_to_honet

hg = convert_hypernetx_to_honet(H)
```

**2. `simulate` 与 `random_discrete_sis` 返回的东西不一样**

| 入口 | 返回 | 适合 |
| --- | --- | --- |
| `random_discrete_sis(...)` | `(t, S, I)`，**逐时间点**计数 | 画传播曲线、算峰值 |
| `simulate(...)` | 元组，每个元素是**事件序列** | 事件级分析 |

想画曲线却用了 `simulate`，会得到只有几个点的「曲线」——那不是 bug，
是事件驱动的本质。详见 [`04_contagion.ipynb`](04_contagion.ipynb)。

**3. `execute_attack` 的 `nc_vals` 是自然连通度，不是连通分量占比**

- 数值**可以大于 1**（实测某图初始值 1.121）
- 数值**越高越鲁棒**，跌到 0 表示网络被打散
- `nc_vals` 首项固定写 `1.0`，那是「未攻击」的**基准占位**，不是真实度量值

详见 [`06_community_attack.ipynb`](06_community_attack.ipynb)。

**4. `draw_sis` 的参数顺序是 `(S, I, t)`**

时间在**最后**。传反了不报错，只会画出一张错误的图——这类错误最难发现。

**5. `export.beautify_global_metrics` 的 `ext` 取值是 `csv` / `excel` / `latex`**

不是 `xlsx`。而且传路径时**不要带后缀**，函数会自己追加。

---

## 第二部分 · 综合实例（脚本）

[`08_case_study_collaboration.py`](08_case_study_collaboration.py) 是一个**端到端**
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
python examples/08_case_study_collaboration.py
```

依赖 `numpy` + `hypernetx`（即 `honet[analysis]`）。绘图可选，没装 matplotlib
会自动跳过，不影响分析结果。

### 这个例子最想传达的一点

**中心性高 ≠ 适合干预。** 脚本输出会明确显示：

- 最活跃的人（加权度最高）
- 位置最不可替代的人（桥接分数最高）
- 传播风险最高的人
- 最佳控制入口（平均可控性最高）

这四个**往往是不同的人**。而且脚本会指出平均可控性存在**并列**
（结构等价的叶子结点算出相同的值），提醒你不能只看单一指标。

图：`08_case_study_result.png`（四个视角并排对比）

---

## 重新生成 notebook

Notebook 里的输出是执行产生的真实结果，由
[`tools/_make_examples.py`](../tools/_make_examples.py) 生成：

```bash
# 全部重新生成
python tools/_make_examples.py

# 只重新生成 04
python tools/_make_examples.py 04
```

生成器读取 [`tools/_example_sources/`](../tools/_example_sources/) 下的 `.py`
描述文件（用 `# %%` / `# %% [markdown]` 标记 cell），真实执行每个 cell，
把输出和 `execution_count` 一起写进 `.ipynb`。

**为什么不用手写 notebook？** 因为手写的输出会过时、会骗人。这里
`allow_errors=False`，任何一个 cell 报错都会中断生成——这保证了
notebook 里记录的 API 用法**真的能跑通**。

要改示例，改 `tools/_example_sources/*.py`，然后重新生成。别直接编辑 `.ipynb`，
下次生成会覆盖。

> 注意：`08_case_study_collaboration.py` 是**独立脚本**，不经过生成器，
> 直接改这个文件即可。

---

## 环境要求

```bash
uv sync --extra analysis    # 跑全部 notebook 和脚本
```

| 示例 | 依赖 |
| --- | --- |
| `01` | 零依赖（仅 `honet` 本身） |
| `02` / `03` / `04` / `06` | `honet[analysis]`（hypernetx、numpy 等） |
| `05` | `numpy`（scipy 可选） |
| `07` | `honet[pandas]` + `honet[viz]`（matplotlib；Excel 需 openpyxl） |
| `08` | `numpy` + `hypernetx`，matplotlib 可选 |
