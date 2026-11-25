# 发布指南（dev / rc → TestPyPI，stable → PyPI）

本文档描述 `honet` 从零到发布的完整流程。

**工具分工**：`uv` 负责依赖、环境、测试与**构建**；`twine` 负责**校验与上传**。
twine 不加入项目依赖，一律用 `uvx twine`（等价 `uv tool run twine`）临时执行。
不使用 `setup.py`、`setup.cfg`、`requirements.txt`、`Pipfile` 或 poetry。

三条发布通道严格分开：

| 通道 | 版本号形态 | 版本示例 | 只发布到 | 上传命令 | 对应分支 / 标签 |
| --- | --- | --- | --- | --- | --- |
| 开发版 dev | `X.Y.Z.devN` | `0.2.0.dev1` | TestPyPI | `uvx twine upload --repository testpypi dist/*` | `develop`，无标签 |
| 候选版 rc | `X.Y.ZrcN` | `0.2.0rc1` | TestPyPI | 同上 | `release/0.2`，标签 `v0.2.0rc1` |
| 稳定版 stable | `X.Y.Z` | `0.2.0` | **正式 PyPI** | `uvx twine upload dist/*` | `main`，标签 `v0.2.0` |

> 铁律：**dev 与 rc 绝不发到正式 PyPI**，正式 PyPI 只接受 `X.Y.Z`。
> twine 本身**不会**替你检查版本形态或版本是否已存在——它只管把文件推过去，
> 由远端索引在接收时拒绝。所以每次上传前必须 `uvx twine check` 通过，并自查通道。

---

## 0. 前置条件

| 项 | 要求 | 检查命令 |
| --- | --- | --- |
| Python | ≥ 3.12（与 `requires-python` 一致） | `uv run python -V` |
| uv | ≥ 0.10（`uv build` 需要） | `uv --version` |
| twine | 由 `uvx` 自动解析，无需预装 | `uvx twine --version` |
| 账号 | PyPI 与 TestPyPI 是**两套独立账号**，需分别注册 | — |
| 2FA | 两个站点都必须开启双因素认证后才能创建 token | 站点 Settings → 2FA |

注册地址：<https://pypi.org/account/register/> 与 <https://test.pypi.org/account/register/>。

首次执行 `uvx twine --version` 时，uv 会在隔离环境里下载 twine（只发生一次，
缓存在 `%LOCALAPPDATA%\uv\cache`），不会污染项目 `.venv`，也不会写进 `pyproject.toml`。

---

## 1. uv 与 twine 的分工

| 环节 | 命令 | 说明 |
| --- | --- | --- |
| 依赖安装 | `uv sync` | 依据 `pyproject.toml` + `uv.lock`，含 `[dependency-groups].dev` |
| 测试 | `uv run pytest` | 在项目虚拟环境中执行 |
| 静态检查 | `uv run ruff check .` | 同上 |
| 版本号修改 | `uv version X.Y.Z` | 只改 `pyproject.toml`（唯一真源） |
| 构建 | `uv build` | 后端 `uv_build`，产出 sdist + wheel 到 `dist/` |
| 校验 | `uvx twine check dist/*` | 检查 METADATA 与 README 能否被 PyPI 正确渲染 |
| 上传（测试） | `uvx twine upload --repository testpypi dist/*` | 走 `~/.pypirc` 的 `testpypi` 段 |
| 上传（正式） | `uvx twine upload dist/*` | 默认仓库 `pypi` |

twine **不做**的事：不解析 `pyproject.toml`、不构建、不检查版本是否已存在、
不改版本号。这些都归 uv 或人工。

---

## 2. 版本号体系（PEP 440）

### 2.1 三类版本号

```
稳定版    X.Y.Z              例如 0.2.0        → 正式 PyPI
候选版    X.Y.ZrcN           例如 0.2.0rc1     → TestPyPI
开发版    X.Y.Z.devN         例如 0.2.0.dev1   → TestPyPI
（可选）  X.Y.ZaN / X.Y.ZbN  例如 0.2.0b1      → TestPyPI
```

PEP 440 排序：`0.2.0.dev1 < 0.2.0a1 < 0.2.0b1 < 0.2.0rc1 < 0.2.0`。
因此 dev/rc 不会被解析器当成“比 0.2.0 更新”，普通用户不会误装半成品。

递增规则（语义化版本）：

| 变更类型 | 递增位 | 示例 |
| --- | --- | --- |
| 破坏性 API 变更 | MAJOR | `0.2.0` → `1.0.0` |
| 向后兼容的新功能 | MINOR | `0.2.0` → `0.3.0` |
| 向后兼容的修复 | PATCH | `0.2.0` → `0.2.1` |

### 2.2 版本号只有一个真源

版本号只写在 `pyproject.toml` 的 `[project].version`。`src/honet/__init__.py` 在运行期用
`importlib.metadata.version("honet")` 读取分发包元数据，不硬编码；
`tests/test_package.py` 断言 `honet.__version__` == 已安装元数据 == `pyproject.toml`。

```bash
uv version                      # 查看当前版本
uv version 0.2.0.dev1           # 开发版
uv version 0.2.0rc1             # 候选版
uv version 0.2.0                # 稳定版
uv version --bump patch         # 递增 PATCH（0.1.0 → 0.1.1）
```

> `uv version` 会顺带同步环境，改完可直接
> `uv run python -c "import honet; print(honet.__version__)"` 复核。

---

## 3. 分支与标签策略

```
main      ──●───────────────●──────────────▶  只保留“可发布为稳定版”的状态（X.Y.Z）
             ╲             ╱ (merge release)
develop   ────●──●──●──●──●──────────────▶  日常开发（X.Y.Z.devN）
                    ╲
feature/*             ●──●                   功能分支，PR 回 develop
```

1. **`main`**：只含已验证、可随时发布稳定版的状态；`pyproject.toml` 版本必须是
   `X.Y.Z`（不带 `rc` / `dev` 后缀）。合入前必须 `uv run pytest` + `uv run ruff check .` 全绿。
2. **`develop`**：日常开发主干，版本号常驻下一个目标的 `X.Y.Z.devN`；
   功能分支 `feature/<topic>` 从 `develop` 切出，PR 合回。
3. **`release/X.Y`**（可选）：功能冻结期从 `develop` 切出，只接受 bugfix；
   此处把版本改为 `X.Y.ZrcN` 并发 TestPyPI。
4. **标签**：只在**上传成功后**打，且与版本号一致：候选版 `v0.2.0rc1`、稳定版 `v0.2.0`。
   `git tag -a v0.2.0 -m "honet 0.2.0"`；推送标签需另行授权。
5. 禁止在 `main` 直接提交功能代码；禁止对已打标签的提交 rebase 或强制推送。

---

## 4. 凭据：API token、`~/.pypirc` 与环境变量

### 4.1 创建 token

1. <https://test.pypi.org/manage/account/token/>（正式站：<https://pypi.org/manage/account/token/>）
   → *Add API token*；
2. Scope 选 `Project: honet`；项目首次上传前只能选 *Entire account*，
   首次上传成功后立刻收窄到单项目并吊销旧 token；
3. 复制 `pypi-` 开头的字符串（只显示一次）。

**两个站点的 token 互不通用**：测试站 token 传正式站会 403，反之亦然。

### 4.2 `~/.pypirc` 配置（`--repository` 依赖它）

twine 的 `--repository <name>` 只能引用 `~/.pypirc` 里登记过的名字；
不写 `--repository` 时默认用 `pypi`。Windows 路径为 `%USERPROFILE%\.pypirc`。

仓库内提供了模板 [`.pypirc.example`](../.pypirc.example)，复制并填写：

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
```

`.pypirc` 已加入 `.gitignore`，**绝不提交**。Unix 上建议 `chmod 600 ~/.pypirc`。

### 4.3 密码用环境变量注入（推荐，避免明文落盘）

`~/.pypirc` 只写 `username`，密码从环境变量读：

```powershell
# PowerShell 7 —— 正式 PyPI
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-正式站token"

# 切到 TestPyPI 前先换成测试站 token
$env:TWINE_PASSWORD = "pypi-测试站token"
```

```bash
# bash / zsh
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="pypi-正式站token"
```

- `TWINE_USERNAME` **必须是 `__token__`**（这是 PyPI 约定的 token 用户名），不是账号名；
- 也可以用 `--username __token__ --password pypi-xxx` 显式传参，但会留在 shell 历史里；
- 还可以用 `TWINE_REPOSITORY=testpypi` 临时切换默认仓库，等价于 `--repository testpypi`。

不想用 `.pypirc` 时，可直接指定上传地址（与 `--repository` 二选一）：

```bash
uvx twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

---

## 5. 发布前检查清单

```bash
# 1) 依赖与 uv.lock 同步
uv sync

# 2) 测试全绿
uv run pytest

# 3) 静态检查
uv run ruff check .
uv run ruff format --check .

# 4) 版本号与目标通道匹配（dev/rc 只能发 TestPyPI，stable 才发 PyPI）
uv version

# 5) 名称是否被占用（404 = 可用，200 = 已存在）
uv run python -c "import urllib.request,urllib.error;\
try: urllib.request.urlopen('https://pypi.org/pypi/honet/json'); print('TAKEN')\
except urllib.error.HTTPError as e: print('FREE' if e.code==404 else e.code)"

# 6) 构建 + 校验（check 不通过绝不上传）
uv build
uvx twine check dist/*
```

人工复核：

- [ ] `CHANGELOG.md` 对应分区（`dev` / `rc` / `stable`）已登记本次内容；
- [ ] `README.md` 示例输出与实际行为一致；
- [ ] `LICENSE` 与 `license = "GPL-3.0-or-later"` + `license-files = ["LICENSE"]` 匹配；
- [ ] `classifiers` 中的 Python 版本不超出 `requires-python`；
- [ ] 上传目标与版本形态一致（这一点 twine 不会帮你把关）；
- [ ] 工作区干净；若使用 git，`uv.lock` 已纳入版本控制。

---

## 6. 构建（`uv build`）

```bash
uv build                                   # sdist + wheel → dist/
uv build --out-dir "$env:TEMP\honet-dist"  # 输出到仓库外（验证用）
uv build --sdist                           # 只构建源码包
uv build --wheel                           # 只构建 wheel
```

产物命名：`honet-0.2.0.tar.gz`、`honet-0.2.0-py3-none-any.whl`（dev/rc 同理带后缀）。

### 6.1 包里到底有什么（明确约定）

| 内容 | wheel（`honet-X.Y.Z-py3-none-any.whl`） | sdist（`honet-X.Y.Z.tar.gz`） |
| --- | --- | --- |
| `honet/` 全部模块（`base/`、`core/`、四个形态子包、`utils/`） | ✅ | ✅（在 `src/honet/` 下） |
| `honet/py.typed` | ✅ | ✅ |
| `LICENSE` | ✅（`dist-info/licenses/LICENSE`） | ✅ |
| `README.md` | ✅（作为 `METADATA` 的 `Description` 内联） | ✅（独立文件） |
| `CHANGELOG.md` | ❌（不进 wheel，避免运行期冗余） | ✅（`source-include` 显式收录） |
| `pyproject.toml`、`PKG-INFO` | ✅（`dist-info/METADATA`） | ✅ |
| `tests/`、`docs/`、`examples/` | ❌ | ❌（不随包分发） |

`CHANGELOG.md` 的收录由 `pyproject.toml` 决定：

```toml
[tool.uv.build-backend]
source-include = ["CHANGELOG.md"]
```

核对命令：

```powershell
# wheel 内容
uv run python -c "import zipfile,os;print('\n'.join(zipfile.ZipFile(os.path.join(os.environ['TEMP'],'honet-dist','stable','honet-0.2.0-py3-none-any.whl')).namelist()))"
# sdist 内容
uv run python -c "import tarfile,os;print('\n'.join(m.name for m in tarfile.open(os.path.join(os.environ['TEMP'],'honet-dist','stable','honet-0.2.0.tar.gz')).getmembers() if m.isfile()))"
```

### 6.2 三通道本地构建 + `twine check` 实操

逐条执行（产物统一放 `D:\TEMP`，不污染仓库）：

```powershell
uv version 0.1.0.dev1
uv build --out-dir "$env:TEMP\honet-dist\dev"
uvx twine check "$env:TEMP\honet-dist\dev\*"

uv version 0.1.0rc1
uv build --out-dir "$env:TEMP\honet-dist\rc"
uvx twine check "$env:TEMP\honet-dist\rc\*"

uv version 0.1.0
uv build --out-dir "$env:TEMP\honet-dist\stable"
uvx twine check "$env:TEMP\honet-dist\stable\*"
```

> twine 自带通配符展开，PowerShell 里写成 `"路径\*"` 即可（bash 用未加引号的 `dist/*`）。
> 只给目录会连 `.gitignore` 一起扫进去并报 `Unknown distribution format`，
> 建议显式写 `*.whl` 与 `*.tar.gz`。
>
> 验证完成后务必把版本改回发布目标（如 `uv version 0.1.0`），否则会误发。

---

## 7. 发布 dev / rc 到 TestPyPI

前提：版本为 `X.Y.Z.devN` / `X.Y.ZrcN`，`uvx twine check` 全 PASSED。

```bash
uvx twine upload --repository testpypi dist/*
```

- `--repository testpypi` 读取 `~/.pypirc` 的 `[testpypi]` 段；
- 等价写法：`uvx twine upload --repository-url https://test.pypi.org/legacy/ dist/*`；
- 凭据来自 `TWINE_USERNAME=__token__` + `TWINE_PASSWORD=<TestPyPI token>`。

从 TestPyPI 安装验证（TestPyPI 不镜像第三方依赖，需回退到 PyPI 取依赖）：

```powershell
$tmp = Join-Path $env:TEMP "honet-testpypi"
New-Item -ItemType Directory -Path $tmp -Force | Out-Null
uv init --bare --name verify --python 3.12 $tmp
uv add --project $tmp `
  --index https://test.pypi.org/simple/ `
  --default-index https://pypi.org/simple/ "honet==0.2.0rc1"
uv run --project $tmp python -c "import honet; print(honet.__version__)"
```

> `--python 3.12` 不可省略：`uv init` 默认生成 `requires-python = ">=3.11"`，
> 与本包 `>=3.12` 冲突会直接解析失败。

---

## 8. 从 dev 晋级到 stable：要改哪些地方

以 `0.2.0.dev1` → `0.2.0rc1` → `0.2.0` 为例：

| 步骤 | dev → rc | rc → stable |
| --- | --- | --- |
| 1. 分支 | 从 `develop` 切 `release/0.2`（功能冻结，只接受 bugfix） | `release/0.2` 合入 `main`（PR + 全部检查通过） |
| 2. 版本号 | `uv version 0.2.0rc1`（唯一改动点：`pyproject.toml`） | `uv version 0.2.0` |
| 3. CHANGELOG | `[dev]` 区内容移至 `[rc]` 区，`[dev]` 清空 | `[rc]` 区内容移至 `[stable]` 区并写日期，`[rc]` 清空 |
| 4. 测试/检查 | `uv run pytest`、`uv run ruff check .` | 同上 |
| 5. 构建 | `uv build` | `uv build` |
| 6. 校验 | `uvx twine check dist/*` | `uvx twine check dist/*` |
| 7. 上传 | `uvx twine upload --repository testpypi dist/*` | `uvx twine upload dist/*` |
| 8. 标签 | `v0.2.0rc1` | `v0.2.0` |
| 9. 收尾 | `develop` 推进到 `0.3.0.dev0` | `develop` 同步 `main` 的版本状态 |

**不需要**改的地方：`src/honet/__init__.py` 里没有硬编码版本，
`honet.__version__` 自动跟随 `pyproject.toml`；若两处不一致，说明有人手写了版本，应立即修正。

---

## 9. 发布 stable 到正式 PyPI

前提：`main` 分支、版本 `X.Y.Z`、rc 已在 TestPyPI 验证通过、`twine check` 全 PASSED。

```bash
# 确认凭据指向正式站 token
uvx twine upload dist/*                       # 默认仓库 pypi
# 或显式：uvx twine upload --repository pypi dist/*
```

上传成功后：

1. 打开 <https://pypi.org/project/honet/> 检查 README 渲染（标题、表格、代码块）；
2. 把 token scope 收窄为 *Project: honet*，吊销账号级旧 token；
3. 打标签 `vX.Y.Z`（推送需授权）；
4. 等索引刷新（约 1 分钟）后验证：

```bash
uv run --with "honet==0.2.0" python -c "import honet; print(honet.__version__)"
```

---

## 10. GitHub Actions 定时发布（可选）

除本地手动 twine 上传外，仓库还提供定时发布工作流
[`.github/workflows/publish-pypi.yml`](../.github/workflows/publish-pypi.yml)：

- **触发**：每周一 03:00 UTC（北京时间 11:00）自动运行；也可在
  *Actions → Scheduled PyPI Release → Run workflow* 手动触发；
- **行为**：读取 `pyproject.toml` 的版本号 → 预发布版本（`*.devN` / `*aN` / `*bN` / `*rcN`）
  直接跳过（按第 2 节策略只发 TestPyPI）→ PyPI 上已存在该版本则跳过 → 否则依次执行
  `uv sync`、`pytest`、`ruff check`、`ruff format --check`、`uv build`、`uvx twine check`，
  全部通过后用 Trusted Publishing 上传；
- **认证**：PyPI *Trusted Publishing*（OIDC），仓库不保存 API token，也不配置 GitHub Secret；
- **幂等**：同一版本重复运行只会跳过，不会尝试覆盖（PyPI 不允许覆盖已发布文件）。

### 10.1 一次性配置 PyPI 侧（必须手动做一次）

`honet` 0.1.0 已发布，项目已存在，因此：

1. 登录 PyPI → 项目页 <https://pypi.org/project/honet/> → *Manage → Publishing*；
2. 点 *Add a trusted publisher*（GitHub），填写：
   Owner `doctordragon666`、Repository name `honet`、
   Workflow name `publish-pypi.yml`、Environment name `pypi`；
3. 保存。之后该环境下的工作流即可用 OIDC 上传，无需 token。

注意 `environment: pypi` 必须与这里填写的 Environment name 完全一致，否则 OIDC 校验失败
（报 `invalid-publisher` 或 403）。

### 10.2 为什么“定时”而不是“每次提交”

发布由版本号驱动：只有把 `pyproject.toml` 的 `version` 改成新的稳定版本并推到 `main`，
下一次定时运行才会真正上传。这样 `main` 上的日常提交不会误发版本，
也天然保证“版本号单一来源”（第 2 节）。

---

## 11. 常见失败原因与处理

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `twine check` 报 `README rendering failed` / `long_description` 警告 | README 含 PyPI 不支持的 Markdown 语法或相对路径图片 | 改绝对 URL、简化表格；`twine check` 不通过就不要上传 |
| `twine check` 报 `InvalidDistribution: Unknown distribution format` | 把目录里的非分发包（如 `.gitignore`、`.DS_Store`）传给了 twine | 只传 `*.whl` 与 `*.tar.gz`，显式指定文件或用 `"dist\*"` 后确认 |
| `HTTPError: 403 File already exists` / `400 File already exists` | 版本号已被占用（含 dev/rc） | 递增版本号重新构建；已上传的版本无法覆盖。twine 上传前不查重，报错来自远端 |
| `HTTPError: 400 Bad Request` 并提示 metadata 非法 | `pyproject.toml` 元数据不合规（license 表达式与分类器冲突、classifier 不存在等） | 按提示修 `pyproject.toml`，重新 `uv build` + `uvx twine check` |
| `HTTPError: 401 Unauthorized` | `TWINE_USERNAME` 不是 `__token__`，或 `TWINE_PASSWORD` 为空/含换行/已吊销 | 固定用 `__token__` 作用户名；重新生成 token 并以环境变量注入 |
| `HTTPError: 403` 但 token 有效 | 用了 TestPyPI token 传正式站（或反之） | 两站 token 分开保存；上传前确认 `--repository` |
| `No such file or directory: dist/*` | 还没构建，或构建到了别的目录 | 先 `uv build`，确认 `dist/` 内有 `*.whl` 与 `*.tar.gz` |
| `Invalid or non-existent authentication information` 提示输入用户名 | 环境变量未生效（新终端、子进程） | 同一终端内 export/`$env:` 后再执行；或写入 `~/.pypirc` 的 `username` |
| 把 dev/rc 发到了正式 PyPI | 通道混用，且 twine 不校验版本形态 | 立即 **yank**；复核第 5 节清单，dev/rc 只能 `--repository testpypi` |
| 名称被占用（注册提示不可用） | PyPI 名称全局唯一，`-`/`_`/大小写归一化后等价 | 只能换名（`honet-py`、`honetx`、`pyhonet` 等）。`honet` 已由本项目发布（0.1.0） |
| README 在 PyPI 页面渲染错乱/空白 | PyPI 只支持有限 Markdown 子集 | 改用绝对图片 URL、简化表格；先发 TestPyPI 预览 |
| `license` 相关上传错误 | 同时使用 PEP 639 `license = "GPL-3.0-or-later"` 表达式与 `License :: OSI Approved :: ...` 分类器 | 二者互斥；本项目只保留表达式 + `license-files` |
| `uv build` 报 `glob ... did not match any files` | `license-files` / `source-include` 指向的文件不存在 | 确认 `LICENSE`、`CHANGELOG.md` 与 `pyproject.toml` 同级 |
| 版本号比较“乱序” | 用了 `0.2.0-rc1`、`0.2.0.dev-1` 等非 PEP 440 写法 | 严格用 `0.2.0rc1`、`0.2.0.dev1`；`+local` 后缀不允许上传 |
| 验证项目安装时报“只支持 >=3.12” | 临时项目继承 `uv init` 默认的 `requires-python = ">=3.11"` | 建临时项目时加 `--python 3.12`（见第 7 节） |
| Actions 报 `invalid-publisher` / 403 | PyPI pending publisher 的 Owner / Repo / Workflow / Environment 与仓库不一致 | 按第 10.1 节逐项核对，`environment` 名必须是 `pypi` |

### 撤销与补救

- **Yank（推荐）**：PyPI 项目页 *Manage → Releases* → 选版本 → *Yank*。文件不删除，
  解析器默认跳过，已锁定依赖的用户仍可安装。
- **删除**：自助删除基本不可用（仅极新/未被安装的项目有例外）。优先发新补丁版本。
- token 泄漏：立刻吊销该 token，检查 *Collaborators* 是否被加入陌生账号。

---

## 12. 一页速查

```bash
uv sync                                         # 依赖（含 dev 组）
uv run pytest                                   # 测试
uv run ruff check .                             # 静态检查

uv version 0.2.0.dev1                           # 开发版
uv build                                        # 构建
uvx twine check dist/*                          # 校验（必须全 PASSED）
uvx twine upload --repository testpypi dist/*   # → TestPyPI

uv version 0.2.0rc1
uv build
uvx twine check dist/*
uvx twine upload --repository testpypi dist/*   # → TestPyPI

uv version 0.2.0                                # 稳定版
uv build
uvx twine check dist/*
uvx twine upload dist/*                         # → 正式 PyPI
```

凭据（每个新终端都要设）：

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-对应站点的token"
```
