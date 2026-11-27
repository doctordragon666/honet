"""honet.plot 的统一样式配置（私有）。

旧库 ``hypertrade.draw`` 在模块 import 时就改 matplotlib 的全局 ``rcParams``，
这会让「导入绘图模块」产生隐藏副作用。本模块把同一套配置收进
:func:`apply_style`，由每个绘图函数在入口处调用（幂等，只生效一次），
既保留旧实现的中文字体与网格风格，又不在 import 阶段改动全局状态。

**与旧实现的差异**：旧代码在每个函数里硬编码 ``fontname='SimSun'``，
没装宋体的系统上会刷出大量缺字形告警；这里改为配置
``font.sans-serif`` 的候选字体链（微软雅黑 → 宋体 → 黑体 → DejaVu Sans），
交给 matplotlib 的字形回退机制挑一个真正可用的中文字体。
"""

from __future__ import annotations

from honet._deps import require

#: 中文字体候选链，按可用性从前往后排（matplotlib 3.6+ 支持逐字形回退）
FONT_CANDIDATES = (
    "Microsoft YaHei",
    "SimSun",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "DejaVu Sans",
)

#: 是否已经应用过样式；模块级标记保证 rcParams 只被改一次
_applied = False


def apply_style() -> None:
    """把旧库的全局绘图风格应用到 matplotlib（幂等）。

    配置项与旧实现一致：中文字体链、负号不转义、stix 数学字体、
    默认网格（虚线、透明度 0.6）、刻度朝内、PDF 内嵌 Type 42 字体。

    调用方是各个绘图函数，因此 ``import honet.plot`` 本身不会 import
    matplotlib，也不会改动全局状态。
    """
    global _applied
    if _applied:
        return

    matplotlib = require("matplotlib", extra="viz")
    rc = matplotlib.rcParams
    rc["font.sans-serif"] = list(FONT_CANDIDATES)
    rc["axes.unicode_minus"] = False  # 关闭 unicode 负号转义，避免负号显示为方块
    rc["mathtext.fontset"] = "stix"
    rc["axes.grid"] = True
    rc["grid.linestyle"] = "--"
    rc["grid.alpha"] = 0.6
    rc["pdf.fonttype"] = 42
    rc["xtick.direction"] = "in"
    rc["ytick.direction"] = "in"
    _applied = True
