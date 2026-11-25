"""包级别契约测试：公开 API 白名单与版本号一致性。

覆盖点：
* ``honet.__all__`` 中的每个名字都能实际导入；
* ``honet.__version__`` 与 ``pyproject.toml`` 的 ``[project].version`` 一致，
  避免发布时出现“元数据版本与运行期版本不符”。

运行方式：``uv run pytest``。
"""

from __future__ import annotations

import re
import tomllib
from importlib.metadata import version as metadata_version
from pathlib import Path

import honet


def _pyproject_path() -> Path:
    """定位仓库根目录下的 pyproject.toml（测试从仓库根运行时 cwd 即根目录）。"""
    return Path(__file__).resolve().parents[1] / "pyproject.toml"


class TestPublicApi:
    """__all__ 与实际导出对象的一致性。"""

    def test_all_names_importable(self) -> None:
        """__all__ 中的名字必须真实存在于模块命名空间。"""
        missing = [name for name in honet.__all__ if not hasattr(honet, name)]
        assert missing == []

    def test_expected_symbols(self) -> None:
        """核心公开符号齐全，防止误删导出。"""
        expected = {
            "Hyperedge",
            "Hypergraph",
            "Node",
            "clique_projection",
            "degree_sequence",
            "incidence_matrix",
        }
        assert expected <= set(honet.__all__)

    def test_star_import(self) -> None:
        """``from honet import *`` 只导入 __all__ 白名单内的名字。"""
        namespace: dict[str, object] = {}
        exec("from honet import *", namespace)  # noqa: S102 - 测试专用
        exported = set(namespace) - {"__builtins__"}
        assert exported == set(honet.__all__)


class TestVersionConsistency:
    """版本号单一来源校验：pyproject.toml → 分发包元数据 → ``honet.__version__``。"""

    def test_version_matches_pyproject(self) -> None:
        """运行期版本号与 pyproject.toml 的 version 完全一致（证明只有一处真源）。"""
        with _pyproject_path().open("rb") as fh:
            data = tomllib.load(fh)
        assert honet.__version__ == data["project"]["version"]

    def test_version_matches_installed_metadata(self) -> None:
        """``__version__`` 来自 importlib.metadata，二者必须相等。"""
        assert honet.__version__ == metadata_version("honet")

    def test_version_is_pep440(self) -> None:
        """版本号必须是 PEP 440 的常规形式，覆盖 stable / rc / dev 三类。

        允许的形式：``X.Y.Z``、``X.Y.ZrcN``、``X.Y.ZaN``、``X.Y.Z.devN``、
        ``X.Y.ZrcN.devM`` 等组合。
        """
        pattern = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?$")
        assert pattern.match(honet.__version__), f"非 PEP 440 版本号：{honet.__version__}"
