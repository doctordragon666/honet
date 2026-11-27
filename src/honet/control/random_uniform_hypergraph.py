"""生成随机 k-均匀有向超图。"""

from __future__ import annotations

from honet.control._random_hypergraph import random_uniform_hypergraph as _impl


def random_uniform_hypergraph(
    n_local: int, k_local: int, m_local: int
) -> list[dict[str, list[int]]]:
    """生成随机 k-均匀有向超图。

    每条超边先抽 ``k`` 个结点作为 tail，再从剩余结点里抽 ``1..k-1`` 个作为 head，
    因此 head 与 tail 不相交、head 非空；重复的 ``(head, tail)`` 会被丢弃重抽。

    参数:
        n_local: 结点总数（编号 ``0..n_local-1``）。
        k_local: 每条超边 tail 的大小。
        m_local: 需要生成的超边条数。

    返回:
        长度为 ``m_local`` 的列表，每项形如 ``{"head": [...], "tail": [...]}``，
        两个列表均按结点下标升序。

    抛出:
        ValueError: ``k_local`` 不满足 ``1 <= k_local < n_local``。

    注意:
        旧实现在 ``k_local >= n_local`` 时 ``remaining`` 恒为空、``continue``
        死循环；这里改为抛 ``ValueError``。随机性来自标准库 ``random``，需要复现
        时先 ``random.seed(...)``。

    示例:
        >>> random.seed(0)  # doctest: +SKIP
        >>> edges = random_uniform_hypergraph(6, 3, 2)  # doctest: +SKIP
    """
    return _impl(n_local, k_local, m_local)
