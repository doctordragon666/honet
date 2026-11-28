"""生成 honet 重构后的架构图。

用 matplotlib 精确控制布局（Graphviz 对 CJK 文本的尺寸估算不准，会导致
文字溢出边框）。坐标单位即英寸，字体磅值与框高按 1pt = 1/72 英寸换算，
因此框一定能容纳文字。

产出：
1. docs/architecture.svg —— 矢量图，适合文档
2. docs/architecture.png —— 位图，适合预览

布局：从左到右三层 —— 分析层（10 子包）→ 适配层（3 个内部模块）→ 核心层（零依赖）。
节点信息取自仓库真实结构（子包文件数、导出符号数、所属 extras）。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

# 核心层：零第三方依赖，`import honet` 只会加载这里
CORE = [
    ("utils/", "关联矩阵 / 度序列 / 团投影"),
    ("temporal/", "时序网络 + W/D/WD"),
    ("multilayer/", "多层网络 + W/D/WD"),
    ("hypergraph/", "超图网络 + W/D/WD"),
    ("simplicial/", "单纯复形 + W/D/WD"),
    ("core/", "Hyperedge / Hypergraph 容器"),
    ("base/mixins/", "Weighted / Directed Mixin"),
    ("base/", "HigherOrderNetwork 抽象基类"),
    ("_frame.py", "DataFrame 长表解析"),
    ("_types.py", "Node / Members 类型别名"),
]

# 适配层：重构新增，隔离「可选依赖」与「hypernetx 版本差异」，并提供共享缓存
ADAPTER = [
    ("_hnx.py", "hypernetx 1.x / 2.x 兼容层", "analysis"),
    ("_deps.py", "可选依赖导入 + 安装提示", "—"),
    ("_lazy.py", "PEP 562 惰性导出", "—"),
    ("_cache.py", "结构指纹缓存（两级校验）", "—"),
    ("_topology.py", "一次性拓扑索引（共享）", "—"),
]

# 分析层：10 个新子包，全部惰性加载
ANALYSIS = [
    ("export/", "导出 LaTeX / CSV", "5", "analysis"),
    ("plot/", "绘图", "13", "viz"),
    ("control/", "控制 · 驱动+动力学+单纯形+稳态+仿真+统计", "51", "hat"),
    ("attack/", "攻击鲁棒性", "6", "analysis"),
    ("contagion/", "传播动力学 · 规格驱动，可扩 SIS/SIR/SIER", "32", "analysis"),
    ("community/", "社区发现", "15", "analysis"),
    ("metrics/", "结点 / 超边 / 全局指标", "14", "analysis"),
    ("build/", "构建与随机生成", "6", "analysis"),
    ("convert/", "超图格式互转", "7", "analysis"),
    ("io/", "数据加载与合并", "5", "analysis"),
]

STYLE = {
    "core": {"fill": "#e8f5e9", "edge": "#2e7d32"},
    "adapter": {"fill": "#fff3e0", "edge": "#e65100"},
    "analysis": {"fill": "#e3f2fd", "edge": "#1565c0"},
}

FONT = "Microsoft YaHei"

# 全部单位：英寸。字体 10.5pt ≈ 0.146in，框高 0.52in 容纳三行文字
# （层名 / 中文说明 / 导出元信息），三者纵向错开，避免水平重叠。
BOX_H = 0.52
GAP = 0.075
PAD = 0.20
HEAD = 0.44


def column_height(n: int) -> float:
    """一列节点的总高度（含外框内边距与标题区）。"""
    return HEAD + n * BOX_H + (n - 1) * GAP + 2 * PAD


def draw_column(ax, x, top, items, style, title, subtitle, box_w, show_meta):
    """在 (x, top) 处自上而下画一列节点，并加外框与标题。返回外框底边 y。"""
    n = len(items)
    body_h = n * BOX_H + (n - 1) * GAP
    total_h = column_height(n)

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, top - total_h),
            box_w,
            total_h,
            boxstyle="round,pad=0,rounding_size=0.05",
            linewidth=1.8,
            edgecolor=style["edge"],
            facecolor=style["fill"],
            zorder=1,
        )
    )

    # 标题：粗体层名 + 浅色说明，同一基线
    ax.text(
        x + 0.16, top - 0.20, title,
        fontsize=13.5, fontweight="bold", color=style["edge"],
        ha="left", va="center", zorder=3, family=FONT,
    )
    ax.text(
        x + 0.16, top - 0.355, subtitle,
        fontsize=8.2, color="#607d8b",
        ha="left", va="center", zorder=3, family=FONT,
    )

    body_top = top - HEAD - PAD + PAD  # = top - HEAD
    for i, item in enumerate(items):
        name, desc = item[0], item[1]
        y_top = body_top - i * (BOX_H + GAP)
        y = y_top - BOX_H

        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x + 0.14, y),
                box_w - 0.28,
                BOX_H,
                boxstyle="round,pad=0,rounding_size=0.035",
                linewidth=1.0,
                edgecolor="#b0bec5",
                facecolor="#ffffff",
                zorder=2,
            )
        )
        cx = x + box_w / 2
        ax.text(
            cx, y + BOX_H * 0.72, name,
            fontsize=10.5, fontweight="bold", color="#263238",
            ha="center", va="center", zorder=3, family=FONT,
        )
        ax.text(
            cx, y + BOX_H * 0.40, desc,
            fontsize=8.0, color="#607d8b",
            ha="center", va="center", zorder=3, family=FONT,
        )
        if show_meta and len(item) > 2:
            ax.text(
                cx, y + BOX_H * 0.13,
                f"导出 {item[2]} · {item[3]}",
                fontsize=7.0, color="#90a4ae",
                ha="center", va="center", zorder=4, family=FONT,
            )

    return top - total_h


def main() -> None:
    width, height = 14.6, 9.3
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")

    top = 7.55
    analysis_x, analysis_w = 0.45, 2.90
    adapter_x, adapter_w = 5.55, 2.55
    core_x, core_w = 10.15, 2.90

    # 三层：核心层最高，分析层与适配层对齐到同一顶部
    bottom_core = draw_column(
        ax, core_x, top, CORE, STYLE["core"],
        "① 核心层", "零第三方依赖 · import honet 只加载这一层", core_w, False,
    )

    adapter_top = top - 2.05
    bottom_adapter = draw_column(
        ax, adapter_x, adapter_top, ADAPTER, STYLE["adapter"],
        "② 适配层", "重构新增 · 隔离可选依赖与版本差异", adapter_w, False,
    )

    bottom_analysis = draw_column(
        ax, analysis_x, top, ANALYSIS, STYLE["analysis"],
        "③ 分析层", "10 个子包 · 107 个导出 · 访问时才 import", analysis_w, True,
    )

    # ---- 依赖箭头：③ → ② → ① ----
    mid_analysis = (top + bottom_analysis) / 2
    mid_adapter = (adapter_top + bottom_adapter) / 2
    mid_core = (top + bottom_core) / 2

    ax.annotate(
        "",
        xy=(adapter_x - 0.10, mid_adapter),
        xytext=(analysis_x + analysis_w + 0.10, mid_analysis),
        arrowprops=dict(arrowstyle="-|>", linewidth=2.0, color="#1565c0",
                        connectionstyle="arc3,rad=-0.12"),
    )
    ax.text(
        (analysis_x + analysis_w + adapter_x) / 2,
        (mid_analysis + mid_adapter) / 2 + 0.30,
        "惰性加载",
        fontsize=9, color="#1565c0", ha="center", va="bottom", family=FONT,
    )

    ax.annotate(
        "",
        xy=(core_x - 0.10, mid_core),
        xytext=(adapter_x + adapter_w + 0.10, mid_adapter),
        arrowprops=dict(arrowstyle="-|>", linewidth=2.0, color="#e65100"),
    )
    ax.text(
        (adapter_x + adapter_w + core_x) / 2,
        (mid_adapter + mid_core) / 2 + 0.16,
        "统一访问容器",
        fontsize=9, color="#e65100", ha="center", va="bottom", family=FONT,
    )

    # ---- 标题 ----
    ax.text(
        0.45, 8.42, "honet 架构（重构后）",
        fontsize=20, fontweight="bold", color="#1a237e",
        ha="left", va="center", family=FONT,
    )
    ax.text(
        0.45, 8.06,
        "核心零依赖 · 分析能力惰性加载 · 旧库 hypertrade 全量迁入",
        fontsize=10.5, color="#546e7a", ha="left", va="center", family=FONT,
    )

    # ---- 图例 ----
    ax.text(
        0.45, 0.48, "依赖方向：③ → ② → ①（不允许反向导入）",
        fontsize=9.5, color="#37474f", ha="left", va="center", family=FONT,
    )
    ax.text(
        0.45, 0.18,
        "验证：796 passed · ruff check / format 全通过 · "
        "指标缓存命中后提速 55–21000× · 传播模型可注册扩展",
        fontsize=8.5, color="#78909c", ha="left", va="center", family=FONT,
    )

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for ext in ("svg", "png"):
        out = DOCS / f"architecture.{ext}"
        fig.savefig(out, format=ext, bbox_inches="tight", pad_inches=0.12,
                    dpi=160, facecolor="white")
        print(f"{out.name}: {out.stat().st_size} 字节")
    plt.close(fig)


if __name__ == "__main__":
    main()
