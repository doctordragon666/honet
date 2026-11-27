# Changelog

本文件记录 `honet` 的所有重要变更，格式参考
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [PEP 440](https://peps.python.org/pep-0440/) 与
[语义化版本](https://semver.org/lang/zh-CN/)。

## 分区说明

本文件按**发布通道**分四个区，与版本号形态一一对应。发布流程见
[`docs/PUBLISHING.md`](docs/PUBLISHING.md)。

| 分区 | 版本号形态 | 含义 | 发布到 |
| --- | --- | --- | --- |
| `Unreleased` | 无版本号 | 已合入 `develop` 但尚未定版本号的改动，随时可能变 | 不发布 |
| `dev` | `X.Y.Z.devN` | 下一个版本的开发快照，功能可能不完整 | 仅 TestPyPI |
| `rc` | `X.Y.ZrcN` | 功能冻结的候选稳定版，只接受修 bug | 仅 TestPyPI |
| `stable` | `X.Y.Z` | 正式稳定版，API 向后兼容 | 正式 PyPI |

晋级规则（单向，不可跳过 `rc` 直接上正式）：

1. `Unreleased` → `dev`：开一个新版本号，把 `Unreleased` 的内容搬进 `dev` 区；
2. `dev` → `rc`：功能冻结，版本号改为 `X.Y.Zrc1`，`dev` 区清空；
3. `rc` → `stable`：`X.Y.ZrcN` 验证无问题后，版本号定为 `X.Y.Z`，内容搬进 `stable` 区；
4. 已发布的 `stable` 条目**不得修改**（只能追加新版本）。

版本号只维护在 `pyproject.toml` 一处，`honet.__version__` 在运行期从分发包元数据读取。

---

## [Unreleased]

### Added

- `Hypergraph` 与四种高阶网络（`SimplicialComplexNetwork` / `HypergraphNetwork` /
  `MultilayerNetwork` / `TemporalNetwork`）新增 `from_dataframe` 类方法：按长表
  （一行 = 结点属于某条超边 / 单形 / 事件）构造，支持自定义列名与可选权重列；
  多层与时序额外按 `layer` / `time` 分组，同一标识在不同层、不同时刻互不合并。
  缺失值、权重冲突、重复超边均给出明确异常。
- 新增可选依赖 `honet[pandas]`；核心包运行时不 import pandas，仅在类型标注中引用。
- 新增 `llms.txt` 与 `AGENTS.md`：面向 LLM 与编码代理的机器可读说明（定位、何时使用、
  安装、最小示例、选型表、API 速查、语义陷阱）；README 增加「给 AI 助手」小节并链接两者。
- **新增 10 个分析子包**（由旧库 `hypertrade` 全量迁入，功能不减，旧目录已删除）：
  `io`（数据加载与合并）、`convert`（hypernetx / xgi / hypergraphx / NetworkX / HAT 互转）、
  `build`（随机 / 无标度 / 聚类 / 小世界超图生成）、`metrics`（结点 / 超边 / 全局指标，
  含可约性、熵、同配性）、`community`（6 种社区发现方法）、`contagion`（SIS / SIR
  离散、随机、Gillespie 与相图）、`attack`（攻击鲁棒性）、`control`（驱动节点控制）、
  `plot`（曲线 / 社区 / 地图 / 攻击结果绘图）、`export`（LaTeX / CSV / Excel 导出），
  共 107 个导出符号。
- 新增 `docs/ARCHITECTURE.md` 与架构图（`docs/architecture.svg` / `.png`），
  说明「核心层 → 适配层 → 分析层」的单向依赖关系。
- 新增可选依赖组 `honet[analysis]`、`honet[viz]`、`honet[excel]`、`honet[hat]`、`honet[all]`。

### Added

- **控制框架：从「控制哪些结点」扩展到「怎么控制」**（详见 `docs/CONTROL.md`）
  - 原先 `honet.control` 只有拓扑层（dilation 检测 / 可达性 / 最小驱动集），
    缺少动力学、耦合与控制律。新增两层补齐：
  - **动力学层**（移植 Ma et al., Physica A 653 (2024) 130108）：
    `HypergraphControlSystem` 把「结点动力学 + 高阶耦合 + 外部输入」组装成
    `Ẋ = 𝒜X + ℬU`；`hypergraph_laplacian` 构造 m-阶广义 Laplacian；
    `kalman_rank` / `pbh_rank` / `gramian_rank` 三个可控性判据；
    `minimum_energy_control` 给出最小能量控制律与状态轨迹。
  - **单纯形层**（移植周进等, 中国科学: 信息科学 54(3) 2024）：
    `generalized_laplacian` 构造 `c₁L⁽¹⁾ + c₂L⁽²⁾`；`fiedler_vector` /
    `algebraic_connectivity` 取谱量；`simplex_delta` 计算选择指标 `Δ^s`；
    `rank_simplices` 给出牵制优先级；`pinning_matrix` 构造牵制矩阵。
  - 复现论文四例的 Kalman 秩（10 / 9 / 8 / 6）与最小能量驱动
    （误差 3.9e-04 / 2.9e-05）；复现脑网络表 1 的 12 个 `Δ^s` 排名
    （12/12 一致）与定理 3 的一阶近似（预测 0.003992 / 实测 0.003992）。
  - 两层都只用 numpy / scipy，不需要 HAT；`import honet` 仍不引入第三方依赖。

### Changed

- **传播模型改为规格驱动，新增模型无需改引擎**（详见 `docs/CONTAGION.md`）
  - 新增 `honet.contagion._model.CompartmentModel`：用声明式规格描述仓室、
    传染态与状态转移图，替代原先散落各处的 `is_sir` 布尔分叉
    （`_discrete.py` 9 处、`_random_models.py` 7 处）。
  - 新增 `simulate` 统一入口；`hon_sis` / `hon_sir` 收敛为薄封装，消除了两者
    之间 95% 的重复代码，行为与旧签名保持不变。
  - 内置模型扩展到 `SIS` / `SIR` / `SIRS` / `SIER` / `SEIR` / `SI`；
    通过 `register_model` 可注册任意自定义仓室模型（测试用五仓室 `SIQD` 验证）。
  - 状态名开放：除惯例的 S/E/I/R 外，允许 D（接种）、Q（隔离）、V（疫苗）等。
  - `is_sir_like` 改用可达性分析：SIRS 的 R 经 R→S 仍可回到易感，故判为循环型
    而非吸收型。
  - 新增显式随机源 `rng` 参数（借鉴 EoN 的 `fast_SIR(..., rng=...)`）：
    可并行/嵌套运行互不干扰的模拟，且不污染全局随机状态。
  - 数值兼容性：重构前后在 SIS/SIR 上做逐位对照，48 组（6 配置 × 4 seed ×
    2 tmax）全部完全一致。
- **核心层保持零第三方依赖**：分析子包通过新增的 `honet._lazy`（PEP 562 惰性导出）
  按需加载，`import honet` 不会引入 numpy / pandas / hypernetx 等包；缺依赖时只在
  真正访问对应功能时抛出带 `honet[extra]` 安装提示的 `ImportError`。
- 新增 `honet._hnx` 兼容层，隔离 hypernetx 1.x 与 2.x 的接口差异（2.x 移除了
  `H.nodes.degree`、`H.edges.members()`、`H.edges.size`、`hnx.bipartite()` 等）。
- 旧库的驼峰命名（如 `honSIS` / `RandomDiscrete_SIS` / `extract_community_partition`
  的 `Method` 枚举值）保留为别名，指向同一个 snake_case 实现，便于旧代码平滑迁移。
- **性能：减少重复计算与对象转换**（详见 `docs/PERFORMANCE.md`）
  - 新增 `honet._topology`：把 hypernetx 对象一次性抽取成拓扑快照
    （成员 / 隶属 / 度 / 规模），多个指标共享同一份结构，不再各自遍历。
    改用 `H.edges.incidence_dict`（近 0ms）替代逐边 `H.edges[e]`（500 条边约 3.2ms）。
  - 新增 `honet._cache`：按「结构特征 + 内容指纹」两级校验的缓存。
    第一级只比较 `(超边数, 结点数, 成员总数)` 三个整数，命中时不计算指纹；
    结构一变立即失效，不会返回过期结果。按 `id()` 索引并用
    `weakref.finalize` 清理，对象回收后条目自动释放。
  - `clustering_coefficient` 复杂度从 O(结点数 × 超边数) 降到 O(Σ 成员数²)。
  - `weighted_degree` 在无权超图上直接取度值，不再构造关联矩阵。
  - `get_avg_degree_by_edge_size` 改为按超边规模分桶，不再构造稠密矩阵。
  - `betweenness_centrality` 复用索引建图并缓存结果，新增 `samples` 参数支持采样近似。
  - `calc_entropy` / `calc_natural_connectivity` / `calc_reducibility` 结果缓存；
    `calc_reducibility` 的 hypernetx → hypergraphx 转换结果也被复用。
  - `execute_attack` 的默认度量改为复用 `calc_natural_connectivity`，避免每步重复求特征值。
  - 实测（200 结点 / 500 超边）：缓存命中后各指标提速 55×–21000×；
    连续计算 6 个指标时共享索引带来 1.9× 整体加速。

### Fixed

- `.gitignore` 的 `build/` 规则改为锚定根目录的 `/build/`，避免连带忽略
  `src/honet/build/` 子包导致源码无法提交。

## [dev] — `X.Y.Z.devN`（仅 TestPyPI）

（暂无。下一个开发快照将在此登记。）

## [rc] — `X.Y.ZrcN`（仅 TestPyPI）

（暂无。）

## [stable] — `X.Y.Z`（正式 PyPI）

### [0.1.1] - 2026-11-25

#### Added

- 类层次（`honet.base` + 四个形态子包）：抽象基类 `HigherOrderNetwork`，以及四种
  **数据结构不同**的形态 —— `SimplicialComplexNetwork`（极大单形/面向下封闭）、
  `HypergraphNetwork`（超边）、`MultilayerNetwork`（分层存储）、`TemporalNetwork`
  （时间戳事件流）。
- 每种形态派生 `W` / `D` / `WD` 三个显式变体，共 16 个具体类；
  `WeightedMixin` / `DirectedMixin` 承载 `is_weighted` / `is_directed`、权重默认 1.0、
  有向成员顺序与自环保留校验。
- `Hyperedge` 增加 `weight` 字段（默认 1.0，必须为正的有限实数），`Hypergraph.add_edge`
  支持 `weight=` 参数；`Hypergraph.__eq__` 将权重纳入比较。
- `examples/`：四个可直接执行的 Jupyter notebook（构造/带权超边、关联矩阵与度序列、
  团投影、实验室合作小场景），通过 `uv run --with jupyter` 打开，不进入包分发包。
- `.github/workflows/publish-pypi.yml`：每周定时检查版本号，稳定版尚未发布时自动跑
  测试、静态检查、`uv build` + `twine check`，并用 PyPI Trusted Publishing 上传；
  预发布版本（dev/rc）与已存在的版本自动跳过。
- `docs/DEVELOPMENT.md`：面向贡献者的开发指南（环境、目录结构、测试与构建、提交前清单）。

#### Changed

- 发布流程改为「`uv build` 构建 + `uvx twine` 校验/上传」：不再使用 `uv publish`，
  twine 通过 `uvx` 临时执行，不进入项目依赖；新增 `.pypirc.example` 与
  `docs/PUBLISHING.md` 的 twine 章节。
- sdist 通过 `[tool.uv.build-backend].source-include` 收录 `CHANGELOG.md`。
- 源码按「一个文件只放一个类／一个函数」拆分为子包：`base/`（抽象基类与 mixin）、
  `core/`（`Hyperedge` / `Hypergraph`）、四种形态各一个子包（子包内只有一个
  `network.py`，含形态基类与 `W` / `D` / `WD` 变体）、`utils/`（每个度量函数一个模块）；
  对外导入路径与 `__all__` 保持不变。
- 仓库托管地址统一为 `https://github.com/doctordragon666/honet`，`pyproject.toml`
  的 Homepage / Repository / Documentation / Changelog / Issues 同步更新。
- 文档按读者分层：`README.md` 只保留用户内容（安装、快速开始、API、类层次、示例），
  开发环境、目录结构、测试与构建约定移入新增的 `docs/DEVELOPMENT.md`；
  发布流程仍在 `docs/PUBLISHING.md`。README 中的文档链接改为绝对 URL，
  保证在 PyPI 页面也能正确跳转。
- 开源协议由 MIT 变更为 **GPL-3.0-or-later**：`LICENSE` 替换为 GPLv3 全文，
  `pyproject.toml` 的 `license` 字段与 README 徽章、许可证节同步更新。
  0.1.0 为 MIT，自 0.1.1 起为 GPL-3.0-or-later。

### [0.1.0] - 2026-11-24

#### Added

- 初始稳定版本，提供高阶网络（超图）的最小可用公开 API：
  - `Hyperedge`：不可变超边类型（成员集合 + 可选标签），带非空校验。
  - `Hypergraph`：按插入顺序保存结点与超边，支持自动补建结点、重复超边拒绝、
    高阶度、共边邻居、子超图导出、结构相等比较。
  - `incidence_matrix`：关联矩阵 `B`（结点 × 超边）。
  - `degree_sequence`：降序高阶度序列。
  - `clique_projection`：团投影，将高阶网络展开为成对邻接映射。
- 完整类型标注、`py.typed` 标记与中文 docstring。
- pytest 测试套件（`tests/`）与 ruff 静态检查配置。
- `docs/PUBLISHING.md`：基于 uv 的 TestPyPI / PyPI 发布指南（含 dev / rc / stable 三条通道）。
