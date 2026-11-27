"""``honet.export`` 导出/读回函数的单元测试。

覆盖点：
* ``to_latex`` 的完整版式（``center`` + ``tabular`` + ``bicaption`` + ``label``）；
* ``_format_latex_value`` 的三种口径：缺失值留空、整数去小数、其余 3 位小数；
* ``beautify_global_metrics`` 按 csv / excel / latex 落盘，并验证往返一致；
* ``beautify_honcommunity`` 的社团聚合、成员排序与空映射报错；
* ``load_global_metrics`` / ``load_honcommunity`` 的读回与错误分支
  （LaTeX 读回未实现、未知扩展名、缺文件、缺列）；
* 惰性导出：``import honet.export`` 不会拉入 pandas。

数据规模：5 个结点、4 条超边的真实小超图，指标由它实际算出后导出，
再用 ``tmp_path`` 写临时文件验证往返一致性；不做 mock。
运行方式：``uv run pytest tests/test_export.py -q``。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import hypernetx as hnx
import pandas as pd
import pytest

import honet.export as export_pkg
from honet._hnx import edge_sizes, node_degrees
from honet.export import (
    _format_latex_value,
    beautify_global_metrics,
    beautify_honcommunity,
    load_global_metrics,
    load_honcommunity,
    to_latex,
)

#: pandas 写/读 ``.xlsx`` 需要 openpyxl，默认开发环境未安装，相关用例按需跳过。
_HAS_OPENPYXL = importlib.util.find_spec("openpyxl") is not None


@pytest.fixture()
def hg() -> hnx.Hypergraph:
    """5 结点 4 超边的真实小超图。

    拓扑：e1={CN,JP,KR}、e2={JP,KR}、e3={KR,SG}、e4={SG,VN}，
    超度依次为 CN=1、JP=2、KR=3、SG=2、VN=1，KR 是唯一的枢纽结点。
    """
    return hnx.Hypergraph(
        {
            "RCEP": ["CN", "JP", "KR"],
            "CPTPP": ["JP", "KR"],
            "ASEAN": ["KR", "SG"],
            "BILAT": ["SG", "VN"],
        }
    )


@pytest.fixture()
def metrics(hg: hnx.Hypergraph) -> pd.DataFrame:
    """由真实超图算出的全局指标表：结点数、超边数、平均度、最大超边规模。"""
    degrees = node_degrees(hg)
    sizes = edge_sizes(hg)
    return pd.DataFrame(
        {
            "超图": ["RCEP 子网"],
            "结点数": [len(hg.nodes)],
            "超边数": [len(hg.edges)],
            "平均度": [sum(degrees.values()) / len(degrees)],
            "最大超边": [max(sizes.values())],
        }
    )


class TestFormatLatexValue:
    """单元格格式化：缺失值 / 整数 / 小数 / 非数值。"""

    def test_missing_values_become_empty(self) -> None:
        """NaN、pd.NA、None 一律输出空串，表格里留白。"""
        assert _format_latex_value(float("nan")) == ""
        assert _format_latex_value(pd.NA) == ""
        assert _format_latex_value(None) == ""

    def test_integral_float_loses_decimal_part(self) -> None:
        """整数值的浮点数去掉小数部分（3.0 → 3），不写成 3.000。"""
        assert _format_latex_value(3.0) == "3"
        assert _format_latex_value(0.0) == "0"

    def test_fraction_keeps_three_decimals(self) -> None:
        """非整数保留 3 位小数，四舍五入由 float 格式化负责。"""
        assert _format_latex_value(0.12345) == "0.123"
        assert _format_latex_value(2 / 3) == "0.667"

    def test_bool_is_not_treated_as_number(self) -> None:
        """布尔值不参与数值格式化（旧实现同样排除 bool）。"""
        assert _format_latex_value(True) == "True"

    def test_non_numeric_is_stringified(self) -> None:
        """字符串原样输出，不做 LaTeX 转义。"""
        assert _format_latex_value("RCEP") == "RCEP"


class TestToLatex:
    """LaTeX 渲染：版式结构、标题、标签与数值格式。"""

    def test_structure(self, metrics: pd.DataFrame) -> None:
        """输出是完整的 center + tabular 环境，首尾行与列格式符合约定。"""
        text = to_latex(metrics)
        lines = text.splitlines()

        assert lines[0] == "\\begin{center}"
        assert lines[-1] == "\\end{center}"
        assert text.endswith("\n")
        # 5 列 → 首列左对齐、其余 4 列居中
        assert "\t\t\\begin{tabular}{l*{4}{c}}" in text
        assert text.count("\\hline") == 3

    def test_header_and_row(self, metrics: pd.DataFrame) -> None:
        """表头是列名以 `` & `` 连接；数据行的数值按 3 位小数口径渲染。"""
        text = to_latex(metrics)

        assert "\t\t\t超图 & 结点数 & 超边数 & 平均度 & 最大超边 \\\\" in text
        # 平均度 = (1+2+3+2+1)/5 = 1.8 → 1.800；最大超边 = 3 → 3
        assert "\t\t\tRCEP 子网 & 5 & 4 & 1.800 & 3 \\\\" in text

    def test_captions_and_label(self, metrics: pd.DataFrame) -> None:
        """中英文标题写入 bicaption，label 写入 label 行。"""
        text = to_latex(metrics, caption_cn="全局指标", caption_en="Global metrics", label="tab:gm")

        assert "\t\\bicaption[table]{全局指标}{Global metrics}" in text
        assert "\t\\label{tab:gm}" in text

    def test_no_caption_no_label(self, metrics: pd.DataFrame) -> None:
        """不给标题与标签时不输出对应行。"""
        text = to_latex(metrics)

        assert "bicaption" not in text
        assert "\\label" not in text

    def test_single_column_format(self) -> None:
        """只有一列时列格式退化为纯 ``l``（与旧实现一致）。"""
        text = to_latex(pd.DataFrame({"社团": ["A", "B"]}))

        assert "\t\t\\begin{tabular}{l}" in text

    def test_missing_value_renders_blank(self) -> None:
        """缺失值在表格里留白，不会写出 ``nan``。"""
        text = to_latex(pd.DataFrame({"a": [1.0], "b": [float("nan")]}))

        assert "1 &  \\\\" in text
        assert "nan" not in text


class TestBeautifyGlobalMetrics:
    """全局指标表落盘：三种格式 + 索引开关 + 未知扩展名。"""

    def test_csv_roundtrip(self, metrics: pd.DataFrame, tmp_path: Path) -> None:
        """写 CSV 再用 ``load_global_metrics`` 读回，列与值完全一致。"""
        written = beautify_global_metrics(metrics, tmp_path / "metrics")

        assert written == tmp_path / "metrics.csv"
        assert written.exists()

        restored = load_global_metrics(tmp_path / "metrics", "csv")
        assert list(restored.columns) == list(metrics.columns)
        assert restored.iloc[0].tolist() == metrics.iloc[0].tolist()

    def test_latex_written(self, metrics: pd.DataFrame, tmp_path: Path) -> None:
        """latex 格式写出 ``.tex``，内容与 ``to_latex`` 的渲染结果一致。"""
        written = beautify_global_metrics(metrics, tmp_path / "metrics", ext="latex")

        assert written == tmp_path / "metrics.tex"
        assert written.read_text(encoding="utf-8") == to_latex(metrics)

    def test_csv_index_flag(self, metrics: pd.DataFrame, tmp_path: Path) -> None:
        """``index=True`` 时行索引作为首列写出，读回多出一列 ``Unnamed: 0``。"""
        beautify_global_metrics(metrics, tmp_path / "with_index", index=True)
        plain = beautify_global_metrics(metrics, tmp_path / "plain")

        with_index = pd.read_csv(tmp_path / "with_index.csv")
        without_index = pd.read_csv(plain)

        assert list(with_index.columns)[0] == "Unnamed: 0"
        assert list(without_index.columns) == list(metrics.columns)

    def test_unknown_ext_raises(self, metrics: pd.DataFrame, tmp_path: Path) -> None:
        """未知扩展名抛 ValueError 并列出可选值（旧实现是静默什么都不做）。"""
        with pytest.raises(ValueError, match="ext 只能是"):
            beautify_global_metrics(metrics, tmp_path / "bad", ext="parquet")

    @pytest.mark.skipif(_HAS_OPENPYXL, reason="已安装 openpyxl，改由真实读写用例覆盖")
    def test_excel_without_engine_hints_openpyxl(
        self, metrics: pd.DataFrame, tmp_path: Path
    ) -> None:
        """缺 openpyxl 时导出 Excel 抛 ImportError，提示安装 openpyxl。"""
        with pytest.raises(ImportError, match="openpyxl"):
            beautify_global_metrics(metrics, tmp_path / "metrics", ext="excel")

    @pytest.mark.skipif(not _HAS_OPENPYXL, reason="需要 openpyxl 才能读写 .xlsx")
    def test_excel_roundtrip(self, metrics: pd.DataFrame, tmp_path: Path) -> None:
        """写 Excel 再读回，列与值一致。"""
        written = beautify_global_metrics(metrics, tmp_path / "metrics", ext="excel")

        assert written == tmp_path / "metrics.xlsx"
        restored = load_global_metrics(tmp_path / "metrics", "excel")
        assert list(restored.columns) == list(metrics.columns)
        assert restored["结点数"].tolist() == [5]


class TestBeautifyHoncommunity:
    """社区表整理：聚合、排序、空映射与落盘。"""

    def test_groups_and_sorts(self, tmp_path: Path) -> None:
        """同一社团的成员合并并按字母序拼接；社团按编号升序排列。"""
        frame = beautify_honcommunity({"KR": 1, "CN": 0, "JP": 0, "SG": 1, "VN": 2}, tmp_path / "c")

        assert list(frame.columns) == ["社团编号", "国家列表"]
        assert frame["社团编号"].tolist() == [0, 1, 2]
        assert frame["国家列表"].tolist() == ["CN,JP", "KR,SG", "VN"]

    def test_file_written(self, tmp_path: Path) -> None:
        """返回的表与磁盘上的 CSV 内容一致。"""
        frame = beautify_honcommunity({"CN": 0, "JP": 0, "KR": 1}, tmp_path / "c")

        on_disk = pd.read_csv(tmp_path / "c.csv")
        assert on_disk["社团编号"].tolist() == frame["社团编号"].tolist()
        assert on_disk["国家列表"].tolist() == frame["国家列表"].tolist()

    def test_non_string_nodes_are_stringified(self, tmp_path: Path) -> None:
        """结点不是字符串时按 ``str()`` 写入（旧实现同样如此）。"""
        frame = beautify_honcommunity({1: 0, 2: 0}, tmp_path / "n")

        assert frame["国家列表"].tolist() == ["1,2"]

    def test_empty_nodemap_raises(self, tmp_path: Path) -> None:
        """空映射抛 ValueError。"""
        with pytest.raises(ValueError, match="nodemap 不能为空"):
            beautify_honcommunity({}, tmp_path / "c")

    def test_unknown_ext_raises(self, tmp_path: Path) -> None:
        """未知扩展名抛 ValueError。"""
        with pytest.raises(ValueError, match="ext 只能是"):
            beautify_honcommunity({"CN": 0}, tmp_path / "c", ext="json")


class TestLoadHoncommunity:
    """社区表读回：往返一致、缺列校验、LaTeX 分支。"""

    def test_roundtrip(self, tmp_path: Path) -> None:
        """``beautify_honcommunity`` → ``load_honcommunity`` 得到原映射。"""
        nodemap = {"CN": 0, "JP": 0, "KR": 1, "SG": 1, "VN": 2}
        beautify_honcommunity(nodemap, tmp_path / "c")

        assert load_honcommunity(tmp_path / "c", "csv") == nodemap

    def test_drops_rows_with_missing_cells(self, tmp_path: Path) -> None:
        """关键列缺失的行整体跳过，不产生 ``'nan'`` 这样的伪结点。"""
        pd.DataFrame({"社团编号": [0, None], "国家列表": ["CN,JP", None]}).to_csv(
            tmp_path / "c.csv", index=False
        )

        assert load_honcommunity(tmp_path / "c", "csv") == {"CN": 0, "JP": 0}

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        """缺列时抛 ValueError 并列出实际列名（旧实现抛 pandas 的 KeyError）。"""
        pd.DataFrame({"community": [0], "members": ["CN"]}).to_csv(tmp_path / "c.csv", index=False)

        with pytest.raises(ValueError, match="社区表必须包含列"):
            load_honcommunity(tmp_path / "c", "csv")

    def test_latex_not_implemented(self, tmp_path: Path) -> None:
        """从 LaTeX 读回未实现（与旧实现一致）。"""
        with pytest.raises(NotImplementedError, match="LaTeX"):
            load_honcommunity(tmp_path / "c", "latex")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """文件不存在时抛 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="数据表文件不存在"):
            load_honcommunity(tmp_path / "nope", "csv")

    @pytest.mark.skipif(not _HAS_OPENPYXL, reason="需要 openpyxl 才能读写 .xlsx")
    def test_excel_roundtrip(self, tmp_path: Path) -> None:
        """Excel 格式的往返一致。"""
        nodemap = {"CN": 0, "JP": 0, "KR": 1}
        beautify_honcommunity(nodemap, tmp_path / "c", ext="excel")

        assert load_honcommunity(tmp_path / "c", "excel") == nodemap


class TestLoadGlobalMetrics:
    """全局指标表读回：往返一致与错误分支。"""

    def test_roundtrip_values(self, hg: hnx.Hypergraph, tmp_path: Path) -> None:
        """由真实超图算出的指标写盘再读回，数值保持一致。"""
        degrees = node_degrees(hg)
        frame = pd.DataFrame({"结点": list(degrees), "度": list(degrees.values())})
        beautify_global_metrics(frame, tmp_path / "deg")

        restored = load_global_metrics(tmp_path / "deg", "csv")

        assert restored["结点"].tolist() == list(degrees)
        assert restored["度"].tolist() == list(degrees.values())

    def test_latex_not_implemented(self, tmp_path: Path) -> None:
        """从 LaTeX 读回未实现（与旧实现一致）。"""
        with pytest.raises(NotImplementedError, match="LaTeX"):
            load_global_metrics(tmp_path / "m", "latex")

    def test_unknown_ext_raises(self, tmp_path: Path) -> None:
        """未知扩展名抛 ValueError（旧实现是 UnboundLocalError）。"""
        with pytest.raises(ValueError, match="ext 只能是"):
            load_global_metrics(tmp_path / "m", "tsv")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """文件不存在时抛 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="数据表文件不存在"):
            load_global_metrics(tmp_path / "nope", "csv")


class TestLazyImport:
    """惰性导出契约。"""

    def test_exports_are_callable(self) -> None:
        """``__all__`` 里每个名字都必须是函数，不能被同名子模块对象遮蔽。

        回归点：若某个实现模块在模块级 import 同包的另一个公开函数模块，
        import 机制会把**模块对象**写回 ``honet.export`` 的同名属性，导致惰性
        ``__getattr__`` 不再触发，调用时抛 ``TypeError: 'module' object is
        not callable``。
        """
        for name in export_pkg.__all__:
            value: Any = getattr(export_pkg, name)
            assert callable(value), f"{name} 不是函数：{value!r}"

    def test_private_helper_is_reachable_but_not_public(self) -> None:
        """``_format_latex_value`` 可访问，但不进入 ``__all__``。"""
        assert callable(_format_latex_value)
        assert "_format_latex_value" not in export_pkg.__all__
        assert len(export_pkg.__all__) == 5

    def test_importing_package_does_not_load_pandas(self) -> None:
        """子进程内单独 import honet.export，确认 pandas 没被连带导入。"""
        code = (
            "import sys, honet.export;"
            "assert 'pandas' not in sys.modules, 'pandas 被连带导入';"
            "assert 'numpy' not in sys.modules, 'numpy 被连带导入';"
            "assert len(honet.export.__all__) == 5"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )

        assert completed.returncode == 0, completed.stderr
