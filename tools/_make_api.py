"""生成 ``docs/api/analysis.md``：十个分析子包的 autodoc 汇总页。

为什么用脚本而不是手写：子包导出是**运行期事实**（``__all__``），手写清单会漂移。
本脚本 import 各子包、读 ``__all__``，再拼出 ``autofunction`` / ``autoclass`` 条目。
产物是普通 Markdown，进 git、可 review；只有增删导出时才需要重跑。

用法::

    uv run --group docs python tools/_make_api.py
"""

from __future__ import annotations

import importlib
import inspect
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET = ROOT / "docs" / "api" / "analysis.md"

#: (子包名, 一句话说明, 依赖 extras)
SUBPACKAGES: list[tuple[str, str, str]] = [
    ("io", "数据加载与合并", "analysis"),
    ("convert", "hypernetx / xgi / hypergraphx / NetworkX / HAT 互转", "analysis"),
    ("build", "超图构建与随机生成（随机、无标度、聚类、小世界）", "analysis"),
    ("metrics", "结点 / 超边 / 全局指标", "analysis"),
    ("community", "社区发现（Random / Basic / NMF / Recursive / Hierarchy / Kumar）", "analysis"),
    ("contagion", "传播动力学 SIS / SIR（离散、随机、Gillespie、相图）", "analysis"),
    ("attack", "攻击鲁棒性（随机 / 度 / 中心性 / 目标）", "analysis"),
    ("control", "驱动节点控制 + 控制律设计", "hat / analysis"),
    ("plot", "绘图（SIS/SIR 曲线、社区、地图、攻击结果）", "viz"),
    ("export", "导出 LaTeX / CSV / Excel", "analysis"),
]

HEADER = """# 分析子包

本页由 `tools/_make_api.py` 读取各子包的 `__all__` 自动生成，**不要手改**。
新增或删除公开符号后重跑：

```bash
uv run --group docs python tools/_make_api.py
```

| 子包 | 能力 | 导出数 | 需要 |
| --- | --- | ---: | --- |
"""


def render_entry(module_path: str, name: str, obj: object) -> str:
    """按对象类型选择 autofunction / autoclass / autodata。"""
    if inspect.isclass(obj):
        directive = "autoclass"
    elif callable(obj):
        directive = "autofunction"
    else:
        directive = "autodata"
    return f".. {directive}:: {module_path}.{name}"


def main() -> int:
    lines: list[str] = [HEADER.rstrip("\n")]
    exports: dict[str, list[str]] = {}

    for subpackage, _, _ in SUBPACKAGES:
        module = importlib.import_module(f"honet.{subpackage}")
        exports[subpackage] = list(module.__all__)

    # 汇总表
    for subpackage, summary, extra in SUBPACKAGES:
        names = exports[subpackage]
        lines.append(f"| `{subpackage}` | {summary} | {len(names)} | `{extra}` |")
    lines.append("")

    # 每子包一节
    for subpackage, summary, _ in SUBPACKAGES:
        module = importlib.import_module(f"honet.{subpackage}")
        lines.append(f"## `honet.{subpackage}`")
        lines.append("")
        lines.append(f"{summary}。")
        lines.append("")
        docstring = (module.__doc__ or "").strip()
        if docstring:
            # 取首段作为子包说明，避免把整段长文档塞进 API 页
            first = docstring.split("\n\n")[0].strip()
            lines.append(first)
            lines.append("")
        for name in exports[subpackage]:
            obj = getattr(module, name)
            lines.append("```{eval-rst}")
            lines.append(render_entry(f"honet.{subpackage}", name, obj))
            lines.append("```")
            lines.append("")

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    total = sum(len(v) for v in exports.values())
    print(f"已写入 {TARGET.relative_to(ROOT)}（{len(SUBPACKAGES)} 个子包 / {total} 个符号）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
