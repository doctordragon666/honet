"""随机均分基线方法。"""

from __future__ import annotations

from typing import Any

from honet.community._methods import random_method as _impl


def random_method(hnx_g: Any, n_components: int = 5, seed: int | None = None) -> list[list[Any]]:
    """把结点随机打乱后均匀分配到 ``n_components`` 个社区。

    该方法不利用任何结构信息，只作为其他划分方法的对照基线。

    参数:
        hnx_g: hypernetx 超图对象。
        n_components: 社区数量，必须为正整数。结点数少于该值时多出来的社区
            是空列表（与旧实现一致）。
        seed: 随机种子。``None`` 表示每次调用重新随机（旧实现行为）；传入
            整数则结果可复现。

    返回:
        长度为 ``n_components`` 的列表，每个元素是该社区的结点列表。结点按
        打乱后的顺序轮转分配，因此各社区大小最多相差 1。

    抛出:
        ValueError: ``n_components`` 小于 1。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"]})
        >>> partition = random_method(H, n_components=2, seed=0)
        >>> len(partition)
        2
    """
    return _impl(hnx_g, n_components=n_components, seed=seed)
