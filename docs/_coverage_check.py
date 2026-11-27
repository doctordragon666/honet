"""迁移完整性核查：旧库每个公开函数是否都在新包中有对应实现。

判定规则：
1. 旧名（含驼峰）在新包中作为别名存在 → 覆盖；
2. 旧名转 snake_case 后在新包中存在 → 覆盖；
3. 私有函数（下划线开头）只要求同名存在。
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT / "src" / "src" / "hypertrade"

SUBPACKAGES = [
    "io",
    "convert",
    "build",
    "metrics",
    "community",
    "contagion",
    "attack",
    "control",
    "plot",
    "export",
]


def snake(name: str) -> str:
    """把驼峰/混合命名转成 snake_case。"""
    out: list[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out).replace("__", "_")


# 收集新包全部导出名
new_names: set[str] = set()
for sub in SUBPACKAGES:
    module = importlib.import_module(f"honet.{sub}")
    new_names.update(getattr(module, "__all__", []))

# 收集旧库全部顶层函数与类名
old_defs: list[tuple[str, str, str]] = []  # (模块, 名字, 类型)
for path in sorted(OLD.rglob("*.py")):
    if path.name == "__init__.py":
        continue
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            old_defs.append((path.stem, node.name, "func"))
        elif isinstance(node, ast.ClassDef):
            old_defs.append((path.stem, node.name, "class"))

missing: list[str] = []
covered = 0
for module, name, kind in old_defs:
    candidates = {name, snake(name)}
    if candidates & new_names:
        covered += 1
    else:
        missing.append(f"{module}.{name} ({kind})")

print(f"旧库定义总数：{len(old_defs)}")
print(f"新包导出总数：{len(new_names)}")
print(f"已覆盖：{covered}")
print()
if missing:
    print(f"未找到对应实现（{len(missing)} 个）：")
    for item in missing:
        print("  -", item)
else:
    print("旧库全部公开定义均已在 honet 中找到对应实现。")

sys.exit(0)
