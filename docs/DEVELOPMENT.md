# 开发指南

本文档面向**贡献者与维护者**，只讲怎么改、怎么测、怎么构建。

- 只想使用 `honet`：看 [README](../README.md) 与 [`examples/`](../examples/)；
- 要发布版本：看 [PUBLISHING.md](PUBLISHING.md)（含 GitHub Actions 定时发布）。

## 0. 环境

| 项 | 要求 | 检查 |
| --- | --- | --- |
| Python | ≥ 3.12（与 `requires-python` 一致） | `uv run python -V` |
| uv | ≥ 0.10 | `uv --version` |

```bash
uv sync          # 安装运行依赖 + dev 组（pytest、ruff、pandas）
```

依赖与打包元数据**只**来自 `pyproject.toml`。不要新增 `requirements.txt`、
`setup.py`、`setup.cfg`、`Pipfile` 或 poetry 配置；锁文件只保留 `uv.lock`。

## 1. 目录结构

```
honet/
├── src/honet/                  发布到 PyPI 的包
│   ├── __init__.py             仅导出 + __version__（分析子包惰性挂载）
│   ├── _types.py               Node / Members 类型别名
│   ├── _frame.py               DataFrame 长表解析（内部，from_dataframe 共用）
│   ├── _lazy.py                PEP 562 惰性导出机制（分析子包共用）
│   ├── _deps.py                可选依赖导入 + 安装提示
│   ├── _hnx.py                 hypernetx 1.x / 2.x 兼容层
│   ├── _cache.py               结构指纹缓存（两级校验，见 PERFORMANCE.md）
│   ├── _topology.py            一次性拓扑索引（多指标共享）
│   ├── base/
│   │   ├── network.py          HigherOrderNetwork（抽象基类）
│   │   └── mixins/             WeightedMixin、DirectedMixin（各一个文件）
│   ├── core/                   Hyperedge、Hypergraph（各一个文件）
│   ├── simplicial/network.py   单纯复形基类 + W/D/WD
│   ├── hypergraph/network.py   超图基类 + W/D/WD
│   ├── multilayer/network.py   多层基类 + W/D/WD
│   ├── temporal/network.py     时序基类 + W/D/WD
│   ├── utils/                  每个度量函数一个模块
│   └── {io,convert,build,metrics,community,contagion,attack,control,plot,export}/
│                               分析子包（每个子包一个文件一个函数 + 惰性 __init__）
├── tests/                      pytest 测试（不随包分发）
├── examples/                   Jupyter notebook（不随包分发）
├── docs/                       开发文档（不随包分发）
├── llms.txt                    LLM 可读的包说明（llmstxt.org 风格）
├── AGENTS.md                   编码代理使用约定
└── .github/workflows/          CI / 定时发布
```

分层架构见 [ARCHITECTURE.md](ARCHITECTURE.md)（含架构图），
性能设计与缓存机制见 [PERFORMANCE.md](PERFORMANCE.md)，
传播模型的扩展方式见 [CONTAGION.md](CONTAGION.md)。

代码组织约定：

1. **一个文件只放一个类或一个函数**；子包的 `__init__.py` 只做 re-export；
2. 依赖方向固定为 `_types → base → base.mixins → 形态基类 → W/D/WD 变体`，
   不允许反向导入，出现互相依赖时用 `TYPE_CHECKING` 或下沉到 `_types`；
3. 四种形态各有一个子包，子包内只有一个 `network.py`，变体与基类同文件；
4. 公开 API 只由顶层 `honet/__init__.py` 的 `__all__` 维护，新增名字必须同步加进去，
   并补测试（`tests/test_package.py` 会校验可导入性与版本一致性）；
5. **分析子包不得引入硬依赖**：`dependencies` 保持为空，第三方包一律通过
   `honet._deps.require()` 获取，子包用 `honet._lazy` 惰性导出；
6. 访问 hypernetx 时优先用 `honet._hnx` 的兼容函数，不要直接用 1.x 已删除的接口；
7. **新指标优先复用 `honet._topology` 与 `honet._cache`**：
   不要各自遍历 `H.edges` / 构造关联矩阵；全局指标尽量由局部指标导出。
   缓存键带版本后缀（如 `"foo:v2"`），实现口径变化时递增以作废旧条目。

> 历史：分析子包由旧库 `hypertrade`（原 `src/src/`）全量迁入，旧目录已删除。
> 新旧等价性核查脚本归档在 `docs/_equiv_check.py.bak`，需要复核时可临时恢复
> 旧库后运行。

## 2. 日常命令

```bash
uv run pytest                      # 全部测试
uv run pytest tests/test_networks.py -q
uv run ruff check .                # 静态检查
uv run ruff format --check .       # 格式检查（CI 用的就是这个）
uv run ruff format .               # 需要时自动格式化
```

`ruff` 的 `target-version`、`line-length`、规则集都在 `pyproject.toml` 的
`[tool.ruff]` 下；改规则时不要另建配置文件。

## 3. 构建与本地校验

```bash
uv build                                   # 产出 dist/*.whl + dist/*.tar.gz
uvx twine check dist/*.whl dist/*.tar.gz   # README 渲染与元数据校验
```

`twine` 不进入项目依赖，一律通过 `uvx` 临时执行。wheel 只包含 `honet/` 模块与
`py.typed`；`tests/`、`docs/`、`examples/` 不会被打包。

## 4. 示例 notebook

`examples/` 里的 notebook 是用户文档的一部分，改动公共 API 后要确保它们仍能执行：

```bash
uv run --with jupyter --with ipykernel jupyter lab examples/
uv run --with jupyter --with ipykernel jupyter nbconvert \
  --to notebook --execute --inplace examples/01_build_hypergraph.ipynb
```

jupyter 不进项目依赖，用 `--with` 现拉即可；执行后的输出会写回 notebook，
提交前确认输出与代码一致。

## 5. 提交前检查清单

- [ ] `uv run pytest` 全绿；
- [ ] `uv run ruff check .` 与 `uv run ruff format --check .` 无输出；
- [ ] 新增/改名的公开符号已进入 `__all__`，并有对应测试；
- [ ] `CHANGELOG.md` 的 `[Unreleased]` 记录了本次变更；
- [ ] 面向用户的文档（`README.md`、`examples/`）与实际行为一致；
- [ ] 改动了版本号时才需要 `uv build` + `uvx twine check`。

## 6. 发布

版本号、通道（dev/rc/stable）、twine 上传、GitHub Actions 定时发布与常见失败原因，
全部见 [PUBLISHING.md](PUBLISHING.md)。
