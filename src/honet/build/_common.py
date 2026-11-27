"""honet.build 的私有工具：可选依赖入口、随机抽样与超图装配。

文件名以下划线开头，本模块不对外导出任何公开函数，只供同子包的实现模块
复用。单独抽出一层的原因有两个：

1. 实现模块之间若互相 ``import``，同名子模块会被写回父包属性，从而遮蔽
   子包的 PEP 562 惰性导出；把共用逻辑放进私有模块即可避开这个坑。
2. hypernetx 2.4.3 的构造语义有若干坑（见 ``depth_edge_properties`` 与
   ``head_tail_hypergraph`` 的说明），集中到一处便于统一维护。

随机生成类函数统一使用两个互不干扰的随机源：``random.Random(seed)`` 负责
均匀抽样与整数抽样，``numpy.random.default_rng(seed)`` 负责带权抽样。两者
都由同一个 ``seed`` 派生，因此同一 ``seed`` 下结果完全可复现，并且不会像
旧实现那样污染调用方的全局随机状态。
"""

from __future__ import annotations

import random
from importlib import import_module
from typing import TYPE_CHECKING, Any

from honet._deps import require

if TYPE_CHECKING:
    import pandas as pd


def hypernetx() -> Any:
    """返回 hypernetx 模块，缺失时抛出带安装提示的 ``ImportError``。"""
    return require("hypernetx")


def numpy() -> Any:
    """返回 numpy 模块，缺失时抛出带安装提示的 ``ImportError``。"""
    return require("numpy")


def lazy_export(package: str, name: str) -> Any:
    """从子包的惰性导出中取出实现函数。

    直接 ``import honet.io.merge_ew`` 会把同名子模块写回父包属性，使
    ``honet.io.merge_ew`` 从函数变成模块对象；这里统一经父包属性取值，
    并在拿到模块对象时退回模块内的同名函数，两种状态都能用。
    """
    module = import_module(package)
    value = getattr(module, name)
    if not callable(value):
        value = getattr(value, name)
    return value


def sample_head(rng: random.Random, remaining: list[int], k: int) -> list[int]:
    """从 ``remaining`` 中抽 ``1..k-1`` 个结点作为超边的 head。

    抽取规模与旧实现一致：``hsize = randint(1, min(k - 1, len(remaining)))``。
    """
    hsize = rng.randint(1, min(k - 1, len(remaining)))
    return rng.sample(remaining, hsize)


def head_tail_hypergraph(edges: list[dict[str, list[int]]], *, name: str) -> Any:
    """把 ``{"head": [...], "tail": [...]}`` 列表装配成 hypernetx 超图。

    超边 uid 依次取 ``e0``、``e1``……，成员为 ``tail + head``；``head`` 与
    ``tail`` 原样保留为超边属性，因此旧实现里 head/tail 的区分仍然可查
    （``H.edges[e].head`` / ``H.edges[e].tail``）。

    这里刻意先构造 ``{边: 成员}`` 字典再传入 ``edge_properties``：2.4.3 的
    DataFrame 构造路径会把 ``edge_weight_prop_col`` 指定的列改名成 ``weight``，
    而字典路径不会，head/tail 因此能原样保留。
    """
    hnx = hypernetx()
    incidence = {f"e{index}": [*edge["tail"], *edge["head"]] for index, edge in enumerate(edges)}
    properties = {
        f"e{index}": {"head": list(edge["head"]), "tail": list(edge["tail"])}
        for index, edge in enumerate(edges)
    }
    return hnx.Hypergraph(incidence, edge_properties=properties, name=name)


def depth_edge_properties(frame: pd.DataFrame) -> dict[Any, dict[str, Any]]:
    """由边表的 ``depth`` 列生成 ``{超边: {"weight": 深度, "depth": 深度}}``。

    hypernetx 2.4.3 在 DataFrame 构造路径下会把 ``edge_weight_prop_col`` 指定
    的列重命名成 ``weight``，导致 ``H.edges[e].depth`` 变成 ``None``。因此这
    里同时写入 ``weight``（供 ``edge_weight_prop_col="weight"`` 读取，使边权重
    等于深度）和 ``depth``（供调用方继续按旧名字查询深度），两者取值相同。
    """
    depths = frame.drop_duplicates("agreement").set_index("agreement")["depth"].to_dict()
    return {edge: {"weight": depth, "depth": depth} for edge, depth in depths.items()}
