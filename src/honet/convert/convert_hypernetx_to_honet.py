"""hypernetx 超图 → :class:`honet.Hypergraph`。

这是两个「同名但不同物」的超图类型之间的桥：

* ``hypernetx.Hypergraph`` 是分析层（``metrics`` / ``contagion`` / ``community`` /
  ``attack``）的输入，由全部 ``honet.build.*`` 生成器产出；
* ``honet.Hypergraph`` 是零依赖的通用容器，``incidence_matrix`` /
  ``degree_sequence`` / ``clique_projection`` 只接受它。

在此之前两者之间没有官方通道，使用者只能手写
``honet.Hypergraph(edges=[sorted(H.edges[e]) for e in H.edges])``。这个朴素写法
有三个坑：

1. **丢标签**：边名全部丢失，无法回溯是哪条超边。
2. **丢权重**：``Hypergraph.__init__`` 的 ``edges`` 参数只接受成员列表。
3. **重复成员集冲突**：hypernetx 允许 ``{"e1": ["a","b"], "e2": ["b","a"]}``
   这样的两条边，而 ``honet.Hypergraph`` 的判重键是「成员集合 + 标签」，
   标签都为空时第二条会抛「重复的超边」。

本模块把这些细节收敛到一处。
"""

from __future__ import annotations

from typing import Any

from honet.core.hypergraph import Hypergraph


def _edge_members(hnx_hg: Any, edge: Any) -> list[Any]:
    """取一条边的成员列表。

    优先走 ``incidence_dict``：hypernetx 2.4.3 里 ``H.edges[e]`` 每次调用都会
    重建 ``AttrList``（500 条边约 3.2 ms），而 ``incidence_dict`` 是一次性
    DataFrame 查询，接近 0 ms。取不到时再退回 ``H.edges[e]``。
    """
    incidence = getattr(hnx_hg.edges, "incidence_dict", None)
    if isinstance(incidence, dict) and edge in incidence:
        return list(incidence[edge])
    return list(hnx_hg.edges[edge])


def _edge_weights(hnx_hg: Any) -> dict[Any, float]:
    """取 ``{边名: 权重}``。

    hypernetx 2.4.3 **没有** ``H.edges.weights`` 属性（旧版本有），权重存在
    ``H.edges.properties`` 这个 DataFrame 的 ``weight`` 列里。取不到时返回空
    字典，调用方按默认权重 1.0 处理。
    """
    properties = getattr(hnx_hg.edges, "properties", None)
    if properties is None or not hasattr(properties, "get"):
        return {}
    column = properties.get("weight")
    if column is None:
        return {}
    return {edge: float(value) for edge, value in column.items()}


def convert_hypernetx_to_honet(
    hnx_hg: Any,
    *,
    keep_labels: bool = True,
    keep_weights: bool = True,
    merge_duplicates: bool = True,
) -> Hypergraph:
    """把 hypernetx 超图转换为 :class:`honet.Hypergraph`。

    参数:
        hnx_hg: hypernetx 超图对象。
        keep_labels: 是否把 hypernetx 的边名保留为 ``label``。边名若不是字符串
            （例如整数 id），会转成 ``str`` 以符合 ``Hyperedge.label`` 的类型
            约束；为 ``False`` 时全部留空。
        keep_weights: 是否迁移边权重。
        merge_duplicates: 如何处理**成员集合相同**的边。

            * ``True``（默认）：合并为一条，权重相加。重复边各自代表一次独立
              交互，丢弃会让总权重凭空减少，因此累加而不是取其一。
            * ``False``：原样保留为多条边（依靠不同的 ``label`` 区分）。边名
              若恰好也相同，``honet.Hypergraph`` 仍会抛「重复的超边」——那种
              情况下数据本身就无法无损表达。

    返回:
        ``honet.Hypergraph``，成员以 ``frozenset`` 保存（无序），结点顺序
        沿用 hypernetx 的 ``nodes`` 迭代顺序。

    抛出:
        TypeError: ``hnx_hg`` 缺少 ``nodes`` / ``edges`` 属性。
        ValueError: ``merge_duplicates=False`` 且重复边的边名也相同。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["c", "d"]})
        >>> g = convert_hypernetx_to_honet(H)
        >>> g.num_nodes, g.num_edges
        (4, 2)
        >>> g.degree("c")
        2

        重复成员集默认合并：

        >>> H_dup = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "a"]})
        >>> convert_hypernetx_to_honet(H_dup).num_edges
        1
        >>> convert_hypernetx_to_honet(H_dup, merge_duplicates=False).num_edges
        2

    备注:
        转换是**单向且有损**的：hypernetx 的 ``misc_properties`` 等自定义属性
        不会迁移，只有结构、边名与权重被保留。

        **关于「权重」的口径**：本函数迁移的是 hypernetx 的**边级权重**
        （``H.edges.properties["weight"]``，每条边一个标量，默认 1.0），
        对应 ``Hyperedge.weight``。它与**关联矩阵权重**（长表
        ``(edge, node, weight)`` 里的逐结点关联强度，列和即
        :func:`~honet.convert.get_hyperedge_weights` 的返回值）是两回事：
        后者描述「某个结点在某条超边里占多大份量」，``honet.Hypergraph``
        没有对应字段，因此不迁移。需要保留关联权重时请直接读取
        ``H.edges.properties`` 或调用 ``get_hyperedge_weights``。
    """
    # 类型检查：不能只看 nodes/edges 属性——honet.Hypergraph 也有这两个属性，
    # 那样会在后续访问 incidence_dict / properties 时抛出难以理解的
    # AttributeError。这里直接看类的全名，报错信息才能指向真正的原因。
    if isinstance(hnx_hg, Hypergraph):
        msg = (
            "输入已经是 honet.Hypergraph，无需转换；"
            "本函数用于把 hypernetx.Hypergraph 转成 honet.Hypergraph"
        )
        raise TypeError(msg)
    if not hasattr(hnx_hg, "nodes") or not hasattr(hnx_hg, "edges"):
        msg = f"需要 hypernetx 超图（含 nodes/edges 属性），收到 {type(hnx_hg).__name__}"
        raise TypeError(msg)

    weights = _edge_weights(hnx_hg) if keep_weights else {}

    # 先聚合成「待写入的边」列表，再统一构建。
    # 之所以不边遍历边 add_edge：Hypergraph 没有修改已有边的方法
    # （只有 add_edge / remove_node / subhypergraph），合并重复边时无法就地更新权重。
    planned: list[tuple[list[Any], str | None, float]] = []
    index_by_members: dict[frozenset[Any], int] = {}

    for edge in hnx_hg.edges:
        members = _edge_members(hnx_hg, edge)
        key = frozenset(members)
        weight = weights.get(edge, 1.0)
        label = str(edge) if keep_labels else None

        if key in index_by_members:
            if merge_duplicates:
                position = index_by_members[key]
                kept_members, kept_label, kept_weight = planned[position]
                planned[position] = (kept_members, kept_label, kept_weight + weight)
                continue
            # 不合并：靠不同 label 区分。label 也相同则由 add_edge 抛错，
            # 此时数据本身无法无损表达，报错是正确行为。

        index_by_members.setdefault(key, len(planned))
        planned.append((members, label, weight))

    result = Hypergraph()
    for members, label, weight in planned:
        result.add_edge(members, label=label, weight=weight)
    return result
