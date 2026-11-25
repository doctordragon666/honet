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

（暂无。下一个版本的改动将在此登记。）

## [dev] — `X.Y.Z.devN`（仅 TestPyPI）

（暂无。下一个开发快照将在此登记。）

## [rc] — `X.Y.ZrcN`（仅 TestPyPI）

（暂无。）

## [stable] — `X.Y.Z`（正式 PyPI）

### [0.2.0] - 2026-11-25

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
  0.1.0 为 MIT，自 0.2.0 起为 GPL-3.0-or-later。

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
