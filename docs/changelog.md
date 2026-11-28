# 更新日志

本项目的完整变更记录维护在仓库根目录的
[`CHANGELOG.md`](https://github.com/doctordragon666/honet/blob/main/CHANGELOG.md)，
按 `Unreleased` / `dev` / `rc` / `stable` 四个发布通道分区记录，格式参考
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## 版本号规则

版本号遵循 [PEP 440](https://peps.python.org/pep-0440/) 与
[语义化版本](https://semver.org/lang/zh-CN/)，只维护在 `pyproject.toml` 一处；
`honet.__version__` 在运行期从分发包元数据读取，因此不存在「两处版本不一致」。

| 通道 | 版本号形态 | 发布到 |
| --- | --- | --- |
| `Unreleased` | 无版本号 | 不发布 |
| `dev` | `X.Y.Z.devN` | 仅 TestPyPI |
| `rc` | `X.Y.ZrcN` | 仅 TestPyPI |
| `stable` | `X.Y.Z` | 正式 PyPI |

晋级规则与发布步骤见[发布指南](development/publishing.md)。

## 当前版本

```{eval-rst}
.. py:data:: honet.__version__
   :no-index:

   当前安装的版本号，由 ``pyproject.toml`` 的 ``[project].version`` 决定。
```

```python
import honet
print(honet.__version__)    # 例如 "0.1.1"
```
