# Changelog

本文件记录 `honet` 的所有重要变更，格式参考
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [PEP 440](https://peps.python.org/pep-0440/) 与
[语义化版本](https://semver.org/lang/zh-CN/)。

## 分区说明

本文件按**发布通道**分四个区，与版本号形态一一对应。发布流程见
[`docs/development/publishing.md`](docs/development/publishing.md)。

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

- **Sphinx 文档站 + GitHub Pages 自动部署**
  - `docs/` 改为 Sphinx（MyST + furo）源目录，分「使用指南 / API 参考 / 开发文档」三部分；
    原先散落的 `ARCHITECTURE.md` / `DEVELOPMENT.md` / `PERFORMANCE.md` / `CONTAGION.md` /
    `CONTROL.md` / `PUBLISHING.md` 迁入 `docs/development/`，内容保留。
  - 新增逐符号 API 参考（`autodoc` 从源码 docstring 生成），页面由
    `tools/_make_api.py` 读取各子包 `__all__` 生成，避免手写清单漂移。
  - 新增 `.github/workflows/docs.yml`：推送 `main` 自动构建并部署到
    <https://doctordragon666.github.io/honet/>。构建用 `-W`（警告即错误）。
  - 新增 `docs` 依赖组（sphinx / myst-parser / furo / sphinx-autodoc-typehints /
    sphinx-copybutton），不影响发布到 PyPI 的运行依赖。
- **`tools/` 目录**：`docs/` 里的一次性探测脚本已删除，仍在用的生成器迁入 `tools/`：
  `_make_examples.py`（→ `examples/*.ipynb`）、`_make_architecture.py`
  （→ `docs/_static/architecture.{svg,png}`）、`_example_sources/`、`_make_api.py`。

### Changed

- `docs/` 下的架构图移到 `docs/_static/`，架构图与 notebook 生成器路径随之更新。
- `ruff` 的 `extend-exclude` 增加 `tools`（开发期脚本按展示效果排版，不做 lint）。
- 修正若干 docstring 的行内标记：闭定界符（``` `` ``` / `**`）后紧跟全角括号或汉字时
  docutils 不识别结束标记，已在中间补空格（不影响渲染文本，只修语法）。
- `generate_tau` 的 `type` 取值表列宽不足导致 docutils 报「Malformed table」，已加宽。

- **`convert` 新增 `convert_hypernetx_to_honet`：hypernetx 超图 → `honet.Hypergraph`**
  - 此前两个「同名但不同物」的超图类型之间没有官方通道，使用者只能手写
    `honet.Hypergraph(edges=[sorted(H.edges[e]) for e in H.edges])`，而这个朴素写法
    会丢边名、丢权重，且遇到 hypernetx 允许的重复成员集（`{"e1": ["a","b"], "e2": ["b","a"]}`）
    直接抛「重复的超边」。
  - 新函数迁移结构、边名（`keep_labels`）与权重（`keep_weights`，从
    `H.edges.properties` 的 `weight` 列读取——hypernetx 2.4.3 已移除 `H.edges.weights`），
    并按 `merge_duplicates` 决定把重复成员集合并为一条（权重相加）还是保留为多条。
  - 传入 `honet.Hypergraph` 时给出「无需转换」的明确提示，而不是后续属性错误。
- **`contagion` 支持时变参数**：`random_discrete_sis` / `random_discrete_sir` /
  `random_gillespie_sis` / `random_gillespie_sir` 的 `tau` 与 `gamma` 现在可以传
  `Callable[[float], ...]`，模拟每步用当前时间求值，从而表达季节性传播、
  政策在某日调整、宣传活动效果衰减等场景。传常数时行为与旧版完全一致
  （常数走快路径，不做额外求值）。
- **`control` 支持线性时变系统（LTV）**
  - `simulate_system` / `simulate_impulse` 的 `A`、`B` 可以是 `Callable[[float], ndarray]`，
    每步用当前时刻求值（冻结系数近似，误差 `O(dt²)`，与零阶保持的输入假设同阶）。
  - 新增 `ltv_gramian`（数值估计可控性 Gramian）与 `ltv_controllability`
    （返回 `LTVControllabilityResult`：Gramian、秩、奇异值、是否可控、条件数）。
    LTV 系统的可控性由 Gramian 的秩决定，常数系统的 Kalman 秩判据对其不适用；
    测试保证两者在常数情形下结论一致。

- **`honet.metrics` 新增 `calc_information_gain`：高阶网络相对成对（投影）网络的信息增益**
  - 公式参考《基于引文大数据的高阶网络建模及信息增益比较研究》（李佳旭等，
    系统科学与数学 41(10), 2021）：把低阶模型的熵减去高阶模型的熵，差值即高阶
    表示带来的信息增益。本实现把「低阶模型」取为该超图的**团投影成对网络**、
    「高阶模型」取为超图本身，两侧共用同一套度分布熵公式：
    `G = S_pair − S_high`，高阶侧用结点超度、成对侧用团投影的邻居数。
  - `G > 0` 表示投影把信息摊得更均匀（一阶模型存在高阶信息损失）；`G = 0`
    表示两者多样性相同（如全部超边都是 2 元边时投影即原网络）；`G < 0` 表示
    投影反而更集中。
  - `detail=True` 同时返回 `pairwise_entropy` / `hypergraph_entropy` /
    `information_gain`，便于核对两侧熵值。结果按结构指纹缓存。
  - 该指标与可约性同放在全局指标模块中。补 7 个单元测试。
- **`metrics` 新增 `temporal_closeness_centrality`：时序网络结点重要性**
  - 静态中心性假设信息可沿任意边传播，但时序网络里只有当后续事件时间戳
    **严格更晚**时信息才传得过去。本函数只承认这种「时间尊重路径」，按其
    跳数计算调和接近中心性：`C(v) = Σ_{u≠v} 1/d_t(v,u) / (n−1)`。
  - 参考了 Kim & Anderson（*Phys. Rev. E* 85, 026107, 2012）的「时间有序图」
    框架。没有直接移植现有实现：GitHub 上 `juancamilog/temporal_centrality`
    是 Python 2 代码，且作者在 README 中自述其介数归一化与论文不一致；
    `networkx-temporal` 无时序介数/接近中心性，`pathpyG` 的时序接近中心性
    以 `delta` 时间窗定义。本实现按**跳数**而非绝对时间计时，因此结果与
    时间戳的物理单位（年 / 秒 / 帧）无关。
  - 强制类型：只接受 `honet.TemporalNetwork` 及其 `W` / `D` / `WD` 变体，
    传一次性超图或普通对象抛 `TypeError`——时序语义与静态语义不同，
    静默接受会让结果产生误读。
  - 支持 `max_hops` 限制可见跳数（对应时间窗思想），非法值抛 `ValueError`。
    取值恒在 `[0, 1]`，不可达结点贡献 0 而不拉垮其他结点。补 12 个单元测试。
- **重写 `examples/`，按用途分为两部分**
  - 第一部分：7 个 API 示例 notebook（`01_hypergraph_basics` / `02_metrics` /
    `03_generators_convert` / `04_contagion` / `05_control` / `06_community_attack` /
    `07_io_export_plot`），覆盖全部 9 个子包（`community` 与 `attack` 合为 06，
    `io` / `export` / `plot` 合为 07）。
    全部输出为**真实执行结果**，由 `docs/_make_examples.py` 从
    `docs/_example_sources/*.py` 生成（`allow_errors=False`，任一 cell 报错即中断）。
  - 第二部分：`08_case_study_collaboration.py` 端到端综合实例，用 13 人协作网络
    演示「构造 → 拓扑画像 → 高阶指标 → 传播风险 → 控制入口 → 综合结论」六步，
    带详细中间注释与结构自检。
  - `examples/README.md` 说明两部分结构与重新生成方式，并集中记录两个最易踩的坑
    （两类超图对象不可混用、`simulate` 与 `random_discrete_sis` 返回格式不同）。
  - 删除被替代的旧 notebook（`01_build_hypergraph` / `02_incidence_and_degrees` /
    `03_clique_projection` / `04_collaboration_scenario`）。
- 新增 `docs/_make_examples.py`：把带 `# %%` 标记的 `.py` 描述文件执行成 notebook，
  输出与 `execution_count` 一并写入，避免手写 notebook 输出过时。

### Changed

- **`honet.metrics` 实现按粒度收敛为三个模块**（公开 API 与函数签名全部不变）
  - 原先 14 个指标各占一个文件（`calc_entropy.py` / `weighted_degree.py` / …），
    文件名与公开函数名相同，导致「子模块 import 会遮蔽同名包属性」这一隐患，
    必须靠包类型的 `__getattribute__` 拦截补救。
  - 现按返回粒度归并为：`nodemetrics.py`（结点级 5 个）、`edgemetrics.py`
    （超边级 1 个）、`globalmetrics.py`（全局级 9 个，含新增的信息增益）。
    辅助模块 `_assortativity.py` / `_reuse.py` 保留不动。
  - 实现模块名与公开函数名不再相同，遮蔽隐患随之消失，`__init__.py` 里的
    `_MetricsPackage.__getattribute__` 补丁已删除。子包内部可直接
    `from honet.metrics.nodemetrics import clustering_coefficient`。
  - 兼容性：`honet.metrics.<name>` 的 15 个公开符号、参数与返回值完全不变；
    只有直接 import 旧实现模块路径（如
    `from honet.metrics.calc_entropy import calc_entropy`）的代码需要改为从
    `honet.metrics` 顶层或新的三个模块导入。
  - `honet.attack` 的两处模块级 import 与 `execute_attack` 的度量调用同步改到新路径。

### Fixed

- **`_cache` 对时序网络完全失效，且一级校验会误判命中**
  - `_cache.structure` 通过 `H.edges.incidence_dict` 取成员总数，而
    `TemporalNetwork.edges` 是元组，没有该属性，异常被 `except Exception`
    吞掉后返回 `None`，导致**时序网络的指标结果从不缓存**（每次重算）。
  - 更严重的是修好取数之后暴露的隐患：结构特征只有
    `(事件数, 结点数, 成员总数)` 三个计数，无法区分「同一批事件换个时间顺序」
    的两个时序网络，而时序指标的结果恰恰由时间顺序决定——那样缓存会把
    另一个网络的答案当成自己的返回。现在时序对象（带 `times` 属性）会把
    逐事件的 `(时刻, 成员)` 时间线纳入结构特征，顺序敏感。
  - 非时序对象的结构特征保持不变，不影响既有缓存行为。

- **`initial_infecteds` 无法单独使用**：`random_discrete_sis` / `random_discrete_sir` /
  `random_gillespie_sir` / `random_gillespie_sis` 四个封装函数无条件把 `rho=None`
  改写成默认值 `0.1`，而底层要求 `rho` 与 `initial_infecteds` 互斥，导致只要传了
  `initial_infecteds` 就必然抛 `ValueError`——「从指定结点开始传播」这类分析完全
  无法进行。现在只在未显式给出初始感染者时才回落到默认比例。补 2 个回归测试。

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

- **控制框架新增能量积分、归一化与统计检验**（参考 nctpy）
  - `_energy_integration.py`：`integrate_control_energy`（Simpson 法则积分
    `∫u²dt`）、`minimum_energy_fast`（Simpson 型 Gramian 数值积分）、
    `minimum_energy_exact`（精确块指数解）。
  - `_normalization.py`：`normalize_state`（欧氏范数归一化）、
    `normalize_weights`（秩变换 + 缩放 + 常数）、`null_p_value`（加一平滑
    经验 p 值）、`fdr_p_values`（BH 步进法）、`label_to_state`（标签整数编码）。
  - **与 nctpy 逐位一致**：`normalize_weights` 误差 0.0。注意 nctpy 的
    `add_constant` 是整体 +1（结果落 `[1,2]`），不是缩放到 `(0,1]`；
    第一版按后者实现，对照后发现偏离，已修正。
  - nctpy 的 `integrate_u` 自身在 scipy >= 1.14 上崩溃（同样的 `simps`
    版本比较 bug），无法数值对照；本实现直接用现存的 `scipy.integrate.simpson`。

### Added

- **控制框架新增前向仿真与批量控制任务**（参考 nctpy）
  - `_simulate_system.py`：`simulate_system` 给定输入正算轨迹（连续时间用
    增广矩阵指数的精确分段推进，离散时间直接迭代）；`simulate_impulse`
    封装阶跃刺激场景。
  - `_batch_control.py`：`expand_states` 把标签向量展开成初态/终态矩阵；
    `control_energy_matrix` 一次算出所有状态对之间的控制能量，
    `BatchEnergyResult` 提供 `hardest()` / `easiest()` / `as_dict()`。
  - **闭环自检**：反解（`minimum_energy_control`）的输入喂回前向仿真，
    终点误差 1.5e-04；批量能量与逐条反解相对差 2.4e-07。这两个测试把
    「反解」与「正算」两个方向的模块绑定在一起，任一侧改动都会被捕获。

### Added

- **控制框架新增稳态层：网络控制入口 + 带约束最优控制**（参考 nctpy）
  - 补齐与 nctpy 对齐的三块能力，全部只用 numpy / scipy：
  - **稳态与能量分析**（`_steady_state.py`）：`matrix_normalization` 谱归一化、
    `is_stable` 稳定性判定、`infinite_gramian`（Lyapunov 方程，支持连续/离散）、
    `gramian_auto` 按时域自动选择。不稳定系统抛 `ValueError` 而非返回 `nan`。
  - **可控性度量**（`_metrics.py`）：`average_controllability`（平均可控性）、
    `modal_controllability`（模态可控性）、`control_energy_ranking`（结点排名）。
  - **带状态轨迹约束的最优控制**（`_optimal_control.py`）：`optimal_control`
    用 Hamiltonian 状态-协态法求解 `J = ∫[ρ‖u‖² + ‖S(x-x_r)‖²]dt`，
    支持连续/离散两种时间系统。
  - 与 nctpy 数值对照：归一化 / 无限 Gramian / 平均可控性 / 模态可控性
    误差 ≤1.1e-15；`optimal_control` 与 `get_control_inputs` 状态误差 ≤7.8e-16。
  - 发现 nctpy 自身 bug（scipy 版本字符串比较误判，调用已移除的
    `sp.integrate.simps`），已在文档记录，本实现不受影响。

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
