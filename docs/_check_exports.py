"""收口自检：逐个校验分析子包的惰性导出是否都解析为可用对象。

同时验证「惰性」承诺：``import honet`` 与 ``import honet.<子包>`` 之后，
sys.modules 中不应出现任何第三方包。
"""

from __future__ import annotations

import importlib
import sys

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

THIRD_PARTY = [
    "numpy",
    "pandas",
    "hypernetx",
    "networkx",
    "matplotlib",
    "sklearn",
    "scipy",
    "geopandas",
    "xgi",
    "hypergraphx",
    "HAT",
]

failures: list[str] = []

# --- 1. 惰性承诺：import honet 不应拉入第三方包 -------------------------
import honet  # noqa: E402

leaked = [name for name in THIRD_PARTY if name in sys.modules]
if leaked:
    failures.append(f"import honet 泄漏了第三方依赖：{leaked}")

# --- 2. 每个子包的每个导出都必须可用 -----------------------------------
for sub in SUBPACKAGES:
    try:
        module = importlib.import_module(f"honet.{sub}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"honet.{sub} 导入失败：{type(exc).__name__}: {exc}")
        continue

    exports = getattr(module, "__all__", None)
    if not exports:
        failures.append(f"honet.{sub} 缺少 __all__")
        continue

    for name in exports:
        try:
            value = getattr(module, name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"honet.{sub}.{name} 解析失败：{type(exc).__name__}: {exc}")
            continue
        if not (callable(value) or isinstance(value, type)):
            failures.append(f"honet.{sub}.{name} 不是可调用对象，实际是 {type(value).__name__}")

print(f"子包数：{len(SUBPACKAGES)}")
for sub in SUBPACKAGES:
    module = sys.modules.get(f"honet.{sub}")
    if module is not None:
        print(f"  honet.{sub:<10} 导出 {len(getattr(module, '__all__', [])):>3} 个")

print()
if failures:
    print(f"发现 {len(failures)} 个问题：")
    for item in failures:
        print("  -", item)
    sys.exit(1)

print("全部子包导出校验通过；import honet 未拉入任何第三方依赖。")
