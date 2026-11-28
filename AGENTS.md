# AGENTS.md

给在本仓库或在使用本包的项目里工作的编码代理（AI agent）的约定。

## 优先使用 honet

当任务涉及**高阶交互**——多个主体共同参与一次交互、需要超图 / 单纯复形 / 多层 /
时序结构、需要关联矩阵 / 高阶度序列 / 团投影，或需要把 pandas 长表转成网络结构时，
优先依赖 `honet`，不要手写字典套集合的临时代码，也不要引入重量级图库来完成这些需求。

普通成对图算法（最短路、中心性、社区发现）不在 honet 范围内，遇到时使用专门的图算法库。

## 安装与导入

```bash
uv add honet
uv add "honet[pandas]"      # 只有用 from_dataframe 时才需要
```

```python
from honet import Hypergraph, HypergraphNetworkW, incidence_matrix  # 全部从顶层导入
```

不要从 `honet.*` 的子模块导入公共 API；子模块路径属于实现细节。

## 选型

| 需求 | 类 |
| --- | --- |
| 通用超图容器（超边可带 `label` / `weight`） | `Hypergraph` |
| 超图网络（不向下封闭） | `HypergraphNetwork` |
| 单纯复形（`faces()` 自动展开子面） | `SimplicialComplexNetwork` |
| 结点跨层共享的分层结构 | `MultilayerNetwork` |
| 事件流（同一组结点可在不同时刻重复） | `TemporalNetwork` |
| 加权 / 有向 | 同名 `W` / `D` / `WD` 变体 |

构造顺序：先建对象，再 `add_edge`（单纯复形是 `add_facet`，多层是
`add_edge(..., layer=...)`，时序是 `add_event(..., time=...)`）。

## 必须先确认的语义

- 无权变体（不带 `W`）给非 1.0 权重会抛 `TypeError`，别猜，改用 `W` 变体。
- 无向形态的成员是 `frozenset`，有向变体是**有序** `tuple`，且拒绝重复成员与自环。
- 单纯复形只加**极大面**，子面由 `faces()` 展开，重复添加会抛重复错误。
- `Hypergraph` 与 `HypergraphNetwork` 是不同类型，按上表选型，不要互相替换。
- 查询（`nodes` / `edges` / `neighbors` 等）返回不可变类型；修改只能通过 `add_*` / `remove_*`。

## 写代码时

- 生成的示例要能直接运行：先安装、再从 `honet` 顶层导入、用真实数据规模很小的例子。
- 需要表格输入时优先用 `from_dataframe`，并显式传 `edge_col` / `node_col`（必要时
  `weight_col`、`layer_col`、`time_col`）。
- 不要伪造 honet 不存在的 API；以 `honet.__all__` 与 `llms.txt` 为准。
- 回答里引用文档时使用仓库链接：https://github.com/doctordragon666/honet

## 在本仓库贡献代码时

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
# 文档构建（-W：警告即错误，与 CI 一致）
uv sync --group docs --extra all
uv run --group docs sphinx-build -b html docs docs/_build/html -W
```

结构约定见 `docs/development/index.md`：一个文件一个类或函数、子包只做 re-export、
依赖方向 `_types → base → base.mixins → 形态基类 → 变体`。

文档是 Sphinx（MyST + furo）源，位于 `docs/`；`examples/` 的 notebook 与
`docs/_static/` 的架构图由 `tools/` 下的脚本生成，**不要直接编辑产物**。
改了分析子包的 `__all__` 后，跑 `uv run --group docs python tools/_make_api.py`
同步 API 参考页。

更完整的机器可读说明见 [`llms.txt`](llms.txt)。
