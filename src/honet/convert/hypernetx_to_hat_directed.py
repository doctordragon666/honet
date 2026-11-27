"""hypernetx 无向超图 → 有向 HAT 超图。

HAT（Hypergraph Analysis Toolbox）用「结点表 + 边表」描述有向超图：边表需要
``Nodes`` / ``head`` / ``tail`` / ``Edges`` 四列。hypernetx 的超边本身没有方向，
这里把每条超边的全部结点同时放进 head 与 tail，表示边内结点之间的双向影响。

依赖 ``HypergraphAnalysisToolbox``（导入名 ``HAT``），属于 ``honet[hat]``
可选组；注意 PyPI 上名为 ``hat`` 的包是无关的 Theano 工具箱。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require


def hypernetx_to_hat_directed(hnx_hg: Any) -> Any:
    """把 hypernetx 无向超图转换为有向 HAT 超图。

    结点按**名字排序**后重新编号为 ``0..n-1``（``Names`` 列保留原名字），
    每条超边的成员按编号升序放入 ``Nodes``、``head``、``tail`` 三列。

    参数:
        hnx_hg: 输入的 hypernetx 无向超图对象。

    返回:
        ``HAT.Hypergraph``，构造参数为
        ``nodes=<结点表>, edges=<边表>, directed=True, compress=False``。
        ``compress=False`` 是必须的：压缩会按边表内容去重并重排边号，
        改变与 hypernetx 超边的对应关系。

    注意:
        结点名必须可排序（旧实现用 ``sorted``）；混合类型结点名会抛
        ``TypeError``。边表按 hypernetx 的边名排序分组，HAT 侧不保留原边名，
        边号是 ``0..m-1`` 的连续整数。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["c", "d"]})
        >>> HG = hypernetx_to_hat_directed(H)
        >>> HG.nnodes, HG.nedges
        (4, 2)
    """
    HAT = require("HAT", extra="hat")
    np = require("numpy")
    pd = require("pandas")

    # 1. 结点表：名字排序后编号，Names 列保留原名字供反查
    unique_nodes = sorted(hnx_hg.dataframe["nodes"].unique())
    node_df = pd.DataFrame({"Nodes": np.arange(len(unique_nodes)), "Names": unique_nodes})
    name_to_idx = {name: index for index, name in enumerate(unique_nodes)}

    # 2. 边表：按 hypernetx 的边分组，成员编号升序，head 与 tail 取同一集合
    edge_nodes_list: list[list[int]] = []
    head_list: list[list[int]] = []
    tail_list: list[list[int]] = []
    for _, group in hnx_hg.dataframe.groupby("edges"):
        node_names = list(group["nodes"].unique())
        node_idxs = sorted(name_to_idx[name] for name in node_names)
        edge_nodes_list.append(node_idxs)
        head_list.append(node_idxs)
        tail_list.append(node_idxs)

    edge_df = pd.DataFrame(
        {
            "Nodes": edge_nodes_list,
            "head": head_list,
            "tail": tail_list,
            "Edges": np.arange(len(edge_nodes_list)),
        }
    )

    return HAT.Hypergraph(nodes=node_df, edges=edge_df, directed=True, compress=False)
