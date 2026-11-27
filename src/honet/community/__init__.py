"""honet.community —— 高阶网络社区划分与划分结果比较。

本子包迁移自旧库的 ``hypertrade.honcommunity`` 模块，属于 ``honet[analysis]``
可选组。全部符号都是惰性导出的：只有真正访问某个符号时才会 import 对应的
实现模块，因此 ``import honet`` 不会因此拉入 numpy / networkx / hypernetx /
scikit-learn / hypergraphx。

实现分两层，原因是 Python 的导入语义：

* ``_`` 前缀的私有模块放共享逻辑（``_hmod`` 封装 hypernetx 的超图模块度算法，
  ``_subhypergraph`` 提供诱导子超图），它们之间可以互相 import；
* 公开模块（``random_method`` / ``basic_method`` 等）只放一个同名公开函数，
  需要复用别的公开函数时**在函数体内延迟 import**。若在模块顶层互相 import，
  ``import honet.community.basic_method`` 会把子模块对象写回包属性，遮蔽本文件
  的惰性导出，使 ``honet.community.basic_method`` 从函数变成模块。

旧的驼峰命名（``Random_Method`` / ``Basic_Method`` / ``NMF_Method`` /
``Recursive_Method`` / ``Hierarchy_Method``）保留为别名，指向同一个函数对象，
便于旧代码平滑迁移；新代码请使用 snake_case 名字。``Method`` 枚举的类名与成员
名（``KUMAR`` / ``NMF`` / ``Random`` 等）是常量标识符，保持原样。
"""

from __future__ import annotations

from honet._lazy import lazy_getattr

_EXPORTS = {
    # 方法枚举
    "Method": "honet.community.method:Method",
    # 各划分方法（snake_case 新名）
    "random_method": "honet.community.random_method:random_method",
    "basic_method": "honet.community.basic_method:basic_method",
    "nmf_method": "honet.community.nmf_method:nmf_method",
    "recursive_method": "honet.community.recursive_method:recursive_method",
    "hierarchy_method": "honet.community.hierarchy_method:hierarchy_method",
    # 划分结果的检查与比较
    "check_partition": "honet.community.check_partition:check_partition",
    "extract_community_partition": (
        "honet.community.extract_community_partition:extract_community_partition"
    ),
    "compare_methods": "honet.community.compare_methods:compare_methods",
    "compare_graphs": "honet.community.compare_graphs:compare_graphs",
    # 旧名别名（值指向同一实现模块里的同一属性）
    "Random_Method": "honet.community.random_method:random_method",
    "Basic_Method": "honet.community.basic_method:basic_method",
    "NMF_Method": "honet.community.nmf_method:nmf_method",
    "Recursive_Method": "honet.community.recursive_method:recursive_method",
    "Hierarchy_Method": "honet.community.hierarchy_method:hierarchy_method",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """按需加载实现模块（PEP 562）。

    额外处理一处命名冲突：实现模块的文件名与公开函数同名（如
    ``honet/community/random_method.py`` 与函数 ``random_method``）。Python 在
    import 子模块后会把包属性 ``random_method`` 指向**模块对象**，遮蔽惰性导出
    的函数，表现为 ``TypeError: 'module' object is not callable``。因此在解析出
    值之后，把该模块对应的其它导出名（含旧名别名）也一并绑定成同一个对象。
    """
    value = lazy_getattr(globals(), __name__, _EXPORTS, name, "analysis")
    module_path = _EXPORTS[name].partition(":")[0]
    for exported, target in _EXPORTS.items():
        if target.partition(":")[0] == module_path:
            globals()[exported] = value
    return value
