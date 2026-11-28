"""Sphinx 配置：honet 文档。

文档源就是本目录下的 Markdown（MyST）。设计取舍：

1. **不做 sphinx-apidoc 落盘**。API 页面由 ``docs/api/*.md`` 手写、每页只放
   ``.. automodule::`` 与显式的 ``autosummary`` 列表，因此文档结构可 review、
   可 diff；生成式 ``.rst`` 每次构建都会变，反而淹没真实改动。
2. **autodoc 必须能导入包**。``honet`` 的分析子包依赖 numpy / hypernetx / HAT，
   构建环境装的是 ``honet[all]``，与 CI 一致；核心层零依赖不受影响。
3. **中文 docstring 用 napoleon 别名映射**。仓库里 186 处写的是「参数:」「返回:」
   「抛出:」，而 napoleon 只认英文段名，故用 ``napoleon_custom_sections`` 把它们
   映射到内置的 parameters / returns / raises 解析器，无需改动源码 docstring。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让 autodoc 在没有安装包时也能导入（CI 与本地 uv sync 后其实都已安装，
# 这里只是兜底，避免「源码树可读但未装」的场景直接构建失败）。
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# -- 项目信息 ----------------------------------------------------------------

project = "honet"
author = "DoctorDragon"
copyright = "2026, DoctorDragon"  # noqa: A001 - Sphinx 约定的配置名
release = "0.1.1"
language = "zh_CN"

# -- 通用配置 ----------------------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
    "attrs_inline",
]
myst_heading_anchors = 3

# -- autodoc / napoleon ------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
# 不用 autosummary 生成存根页：API 页里的条目是手写的 automodule/autoclass，
# 同时开启 autosummary_generate 会让同一对象被描述两次（duplicate object
# description）。需要 autosummary 表格时再单独打开。
autosummary_generate = False

# 「参数:」「返回:」「抛出:」「属性:」「示例:」→ napoleon 内置段名的别名。
# 第二个元素是目标段名（不是样式），napoleon 会复用该段的解析器。
napoleon_custom_sections = [
    ("参数", "parameters"),
    ("返回", "returns"),
    ("抛出", "raises"),
    ("属性", "attributes"),
    ("示例", "examples"),
    ("用法", "examples"),
]
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
# Attributes 段改用 :ivar: 字段列表，而不是生成 .. attribute:: 指令。
# 后者会与 autodoc 的 :members: 对同一个类属性各描述一次，
# 触发「duplicate object description」警告。
napoleon_use_ivar = True

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}
intersphinx_timeout = 15

# -- 输出 --------------------------------------------------------------------

html_theme = "furo"
html_title = "honet 文档"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/architecture.png"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/doctordragon666/honet/",
    "source_branch": "main",
    "source_directory": "docs/",
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
