"""``honet.io`` 数据加载与合并函数的单元测试。

覆盖点：
* CSV / pickle 两种格式的读取分派；
* ``load_hypergraph_data`` 在有无权重路径时返回不同类型；
* 文件缺失时抛 ``FileNotFoundError``（而不是旧实现的“打印后返回空表”）；
* ``merge_ew`` 的归一化口径：按协议内参与国出口额占比分配权重并乘以 ``depth``；
* 缺列时抛 ``ValueError``，参与国贸易额之和为 0 的超边被跳过；
* ``load_map_data`` 读取真实 GeoJSON（需 geopandas；未安装时校验安装提示）。

数据规模：5 个国家 / 4 条协议，均为手工可验算的小样本。
运行方式：``uv run pytest tests/test_io.py -q``；
地图相关用例需 ``uv run --with geopandas pytest tests/test_io.py -q``。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from honet.io import (
    load_edge_data,
    load_hypergraph_data,
    load_map_data,
    load_weight_data,
    merge_ew,
)

#: geopandas 属于 ``honet[viz]``，默认开发环境未安装，相关用例按需跳过。
_HAS_GEOPANDAS = importlib.util.find_spec("geopandas") is not None


@pytest.fixture()
def trade_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """5 国 4 协议的贸易样本：``e`` 为成员表，``w`` 为出口额表。

    出口额（按 ``reporterISO`` 汇总）：
    CN=100、JP=50、KR=10、SG=5、VN=15；US=999 不参与任何协议。
    """
    e = pd.DataFrame(
        {
            "agreement": [
                "RCEP",
                "RCEP",
                "RCEP",
                "CPTPP",
                "CPTPP",
                "CPTPP",
                "ASEAN",
                "ASEAN",
                "ASEAN",
                "ZERO",
            ],
            "country": ["CN", "JP", "KR", "JP", "SG", "VN", "SG", "VN", "TH", "ZZ"],
            "depth": [3, 3, 3, 2, 2, 2, 1, 1, 1, 4],
        }
    )
    w = pd.DataFrame(
        {
            "reporterISO": ["CN", "CN", "JP", "JP", "KR", "SG", "VN", "US"],
            "primaryValue": [60.0, 40.0, 30.0, 20.0, 10.0, 5.0, 15.0, 999.0],
        }
    )
    return e, w


class TestLoadEdgeData:
    """超边数据读取：CSV / pickle 分派与缺文件报错。"""

    def test_read_csv(self, tmp_path: Path) -> None:
        """CSV 路径按 ``read_csv`` 读取，行列内容与写入一致。"""
        path = tmp_path / "edges.csv"
        pd.DataFrame(
            {
                "agreement": ["RCEP", "RCEP", "CPTPP"],
                "country": ["CN", "JP", "SG"],
            }
        ).to_csv(path, index=False)

        frame = load_edge_data(path)

        assert list(frame.columns) == ["agreement", "country"]
        assert len(frame) == 3
        assert frame["country"].tolist() == ["CN", "JP", "SG"]

    def test_read_pickle(self, tmp_path: Path) -> None:
        """``.pkl`` 路径按 ``read_pickle`` 读取。"""
        path = tmp_path / "edges.pkl"
        pd.DataFrame({"agreement": ["RCEP", "CPTPP"], "country": ["KR", "VN"]}).to_pickle(path)

        frame = load_edge_data(path)

        assert len(frame) == 2
        assert frame["agreement"].tolist() == ["RCEP", "CPTPP"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """文件不存在时抛 FileNotFoundError，而不是返回空表。"""
        with pytest.raises(FileNotFoundError, match="超边数据文件不存在"):
            load_edge_data(tmp_path / "nope.csv")


class TestLoadWeightData:
    """权重数据读取：CSV / pickle 分派与缺文件报错。"""

    def test_read_csv(self, tmp_path: Path) -> None:
        """权重 CSV 读到 ``reporterISO`` / ``primaryValue`` 两列。"""
        path = tmp_path / "weights.csv"
        pd.DataFrame(
            {"reporterISO": ["CN", "JP", "SG"], "primaryValue": [100.0, 50.0, 5.0]}
        ).to_csv(path, index=False)

        frame = load_weight_data(path)

        assert list(frame.columns) == ["reporterISO", "primaryValue"]
        assert frame["primaryValue"].sum() == 155.0

    def test_read_pickle(self, tmp_path: Path) -> None:
        """``.pkl`` 权重表同样可读。"""
        path = tmp_path / "weights.pkl"
        pd.DataFrame({"reporterISO": ["VN", "KR"], "primaryValue": [15.0, 10.0]}).to_pickle(path)

        frame = load_weight_data(path)

        assert len(frame) == 2
        assert frame["reporterISO"].tolist() == ["VN", "KR"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """文件不存在时抛 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="超边权重数据文件不存在"):
            load_weight_data(tmp_path / "nope.pkl")


class TestLoadHypergraphData:
    """组合入口：有无权重路径时返回不同类型。"""

    def test_with_weights_returns_pair(self, tmp_path: Path) -> None:
        """同时给边路径与权重路径时返回二元组。"""
        edge_path = tmp_path / "edges.csv"
        weight_path = tmp_path / "weights.csv"
        pd.DataFrame({"agreement": ["RCEP"], "country": ["CN"]}).to_csv(edge_path, index=False)
        pd.DataFrame({"reporterISO": ["CN"], "primaryValue": [100.0]}).to_csv(
            weight_path, index=False
        )

        result = load_hypergraph_data(edge_path, weight_path)

        assert isinstance(result, tuple)
        edges, weights = result
        assert edges["country"].tolist() == ["CN"]
        assert weights["primaryValue"].tolist() == [100.0]

    def test_without_weights_returns_frame(self, tmp_path: Path) -> None:
        """只给边路径时返回单个 DataFrame。"""
        edge_path = tmp_path / "edges.csv"
        pd.DataFrame({"agreement": ["CPTPP", "CPTPP"], "country": ["JP", "SG"]}).to_csv(
            edge_path, index=False
        )

        result = load_hypergraph_data(edge_path)

        assert isinstance(result, pd.DataFrame)
        assert result["country"].tolist() == ["JP", "SG"]

    def test_missing_weight_file_raises(self, tmp_path: Path) -> None:
        """权重路径缺失时抛 FileNotFoundError（旧实现会返回两个空表）。"""
        edge_path = tmp_path / "edges.csv"
        pd.DataFrame({"agreement": ["RCEP"], "country": ["CN"]}).to_csv(edge_path, index=False)

        with pytest.raises(FileNotFoundError):
            load_hypergraph_data(edge_path, tmp_path / "missing.csv")


class TestMergeEw:
    """贸易边数据与权重数据的合并口径。"""

    def test_columns(self, trade_frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """输出列固定为 agreement / country / weight / depth。"""
        merged = merge_ew(*trade_frames)
        assert list(merged.columns) == ["agreement", "country", "weight", "depth"]

    def test_weight_sums_to_depth_per_agreement(
        self, trade_frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """每条协议的权重之和等于该协议的 depth（归一化后整体缩放）。"""
        merged = merge_ew(*trade_frames)
        sums = merged.groupby("agreement")["weight"].sum().to_dict()

        assert sums["RCEP"] == pytest.approx(3.0)
        assert sums["CPTPP"] == pytest.approx(2.0)
        assert sums["ASEAN"] == pytest.approx(1.0)

    def test_weights_proportional_to_export(
        self, trade_frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """权重 = 该国出口额 / 协议内出口额之和 × depth。"""
        merged = merge_ew(*trade_frames)
        rcep = merged[merged["agreement"] == "RCEP"].set_index("country")["weight"].to_dict()

        # RCEP 参与国出口额：CN=100、JP=50、KR=10，合计 160，depth=3
        assert rcep["CN"] == pytest.approx(100 / 160 * 3)
        assert rcep["JP"] == pytest.approx(50 / 160 * 3)
        assert rcep["KR"] == pytest.approx(10 / 160 * 3)

    def test_non_member_export_ignored(
        self, trade_frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """未参与任何协议的国家（US）不出现在结果中。"""
        merged = merge_ew(*trade_frames)
        assert "US" not in set(merged["country"])

    def test_missing_export_treated_as_zero(
        self, trade_frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """协议成员在权重表中缺席时按 0 处理，仍保留该行。"""
        merged = merge_ew(*trade_frames)
        asean = merged[merged["agreement"] == "ASEAN"].set_index("country")["weight"].to_dict()

        assert asean["TH"] == pytest.approx(0.0)
        assert asean["SG"] == pytest.approx(5 / 20 * 1)
        assert asean["VN"] == pytest.approx(15 / 20 * 1)

    def test_zero_total_agreement_skipped(
        self, trade_frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """参与国贸易额之和为 0 的协议整体跳过（ZERO / ZZ）。"""
        merged = merge_ew(*trade_frames)
        assert "ZERO" not in set(merged["agreement"])
        assert "ZZ" not in set(merged["country"])

    def test_depth_recorded_per_row(self, trade_frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """depth 列取该协议第一行的值并回填到每一行。"""
        merged = merge_ew(*trade_frames)
        depths = merged.groupby("agreement")["depth"].unique().to_dict()

        assert depths["RCEP"].tolist() == [3]
        assert depths["CPTPP"].tolist() == [2]

    def test_all_zero_returns_empty_frame(self) -> None:
        """所有权重为 0 时返回空表（与旧实现一致，不抛异常）。"""
        e = pd.DataFrame({"agreement": ["A", "A"], "country": ["CN", "JP"], "depth": [2, 2]})
        w = pd.DataFrame({"reporterISO": ["KR"], "primaryValue": [1.0]})

        merged = merge_ew(e, w)

        assert merged.empty

    def test_missing_e_columns_raises(self) -> None:
        """``e`` 缺 agreement / country 列时抛 ValueError 并列出实际列。"""
        e = pd.DataFrame({"treaty": ["A"], "member": ["CN"], "depth": [1]})
        w = pd.DataFrame({"reporterISO": ["CN"], "primaryValue": [1.0]})

        with pytest.raises(ValueError, match="e 必须包含列"):
            merge_ew(e, w)

    def test_missing_w_columns_raises(self) -> None:
        """``w`` 缺 reporterISO / primaryValue 列时抛 ValueError。"""
        e = pd.DataFrame({"agreement": ["A"], "country": ["CN"], "depth": [1]})
        w = pd.DataFrame({"iso": ["CN"], "value": [1.0]})

        with pytest.raises(ValueError, match="w 必须包含列"):
            merge_ew(e, w)

    def test_old_semantics_regression(self) -> None:
        """与旧实现逐值对照：单一协议、等额出口、depth=1 时权重均为 0.5。"""
        e = pd.DataFrame({"agreement": ["A", "A"], "country": ["CN", "JP"], "depth": [1, 1]})
        w = pd.DataFrame({"reporterISO": ["CN", "JP"], "primaryValue": [10.0, 10.0]})

        merged = merge_ew(e, w)

        assert merged["weight"].tolist() == [0.5, 0.5]
        assert merged["depth"].tolist() == [1, 1]


class TestLoadMapData:
    """地图读取：真实 GeoJSON 文件；geopandas 未安装时校验安装提示。"""

    @staticmethod
    def _write_geojson(path: Path) -> Path:
        """写一个含 4 个点的最小 GeoJSON（4 个“结点”，对应 4 个国家）。"""
        features = [
            {
                "type": "Feature",
                "properties": {"iso": iso, "value": value},
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            }
            for iso, lon, lat, value in [
                ("CN", 116.4, 39.9, 100.0),
                ("JP", 139.7, 35.7, 50.0),
                ("KR", 127.0, 37.5, 10.0),
                ("SG", 103.8, 1.35, 5.0),
            ]
        ]
        payload = {"type": "FeatureCollection", "features": features}
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @pytest.mark.skipif(_HAS_GEOPANDAS, reason="已安装 geopandas，改由真实读取用例覆盖")
    def test_import_error_hints_viz_extra(self, tmp_path: Path) -> None:
        """未安装 geopandas 时抛 ImportError，且提示指向 honet[viz]。"""
        with pytest.raises(ImportError, match="viz"):
            load_map_data(self._write_geojson(tmp_path / "map.geojson"))

    @pytest.mark.skipif(not _HAS_GEOPANDAS, reason="需要 geopandas（honet[viz]）")
    def test_read_geojson(self, tmp_path: Path) -> None:
        """读取真实 GeoJSON：得到 GeoDataFrame，几何与属性列都正确。"""
        frame = load_map_data(self._write_geojson(tmp_path / "map.geojson"))

        assert len(frame) == 4
        assert "geometry" in frame.columns
        assert set(frame["iso"]) == {"CN", "JP", "KR", "SG"}
        assert frame["value"].sum() == pytest.approx(165.0)

    @pytest.mark.skipif(not _HAS_GEOPANDAS, reason="需要 geopandas（honet[viz]）")
    def test_missing_map_raises(self, tmp_path: Path) -> None:
        """地图文件不存在时抛 FileNotFoundError（旧实现返回空 GeoDataFrame）。"""
        with pytest.raises(FileNotFoundError, match="地图数据文件不存在或无法读取"):
            load_map_data(tmp_path / "nope.shp")


class TestLazyImport:
    """``import honet.io`` 不得拉入 pandas / geopandas。"""

    def test_all_exports_resolve_to_callables(self) -> None:
        """五个导出名都必须是函数，不能被同名子模块对象遮蔽。

        回归点：若某个实现模块在模块级 import 同包的另一个公开函数模块，
        import 机制会把**模块对象**写回 ``honet.io`` 的同名属性，导致惰性
        ``__getattr__`` 不再触发，调用时抛 ``TypeError: 'module' object is
        not callable``。这里逐个取值并确认可调用。
        """
        import honet.io as io_pkg

        for name in io_pkg.__all__:
            value = getattr(io_pkg, name)
            assert callable(value), f"{name} 不是函数：{value!r}"

    def test_importing_package_does_not_load_pandas(self) -> None:
        """子进程内单独 import honet.io，确认可选依赖没被连带导入。"""
        code = (
            "import sys, honet.io;"
            "assert 'pandas' not in sys.modules, 'pandas 被连带导入';"
            "assert 'geopandas' not in sys.modules, 'geopandas 被连带导入';"
            "assert 'merge_ew' in honet.io.__all__;"
            "assert len(honet.io.__all__) == 5"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )

        assert completed.returncode == 0, completed.stderr
