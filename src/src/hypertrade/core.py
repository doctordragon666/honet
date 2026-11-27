"""高阶网络通用核心函数

支持加载数据，以及不同对象之间的转换
"""

from geopandas import GeoDataFrame, read_file
from pyogrio.errors import DataSourceError
from pandas import DataFrame, read_csv, read_pickle
import xgi
from .third import *

####################  导入外部数据方法 ####################


def load_map_data(path: str) -> GeoDataFrame:
    """加载地图数据

    Parameters
    ----------
    path : str
        地图路径，shp格式

    Returns
    -------
    GeoDataFrame
        地图数据
    """
    try:
        return read_file(path)
    except DataSourceError:
        print(f"File not found: {path}, Please check the file path.")
        return GeoDataFrame()  # 返回一个空的GeoDataFrame


def load_edge_data(path: str) -> DataFrame:
    """加载超边数据

    Parameters
    ----------
    path : str
        超边数据存放地址

    Returns
    -------
    DataFrame
        超边数据
    """
    try:
        if path.endswith('.pkl'):
            return read_pickle(path)
        else:
            return read_csv(path)
    except FileNotFoundError:
        print(f"File not found: {path}, Please check the file path.")
        return DataFrame()  # 返回一个空的DataFrame


def load_weight_data(path: str) -> DataFrame:
    """加载超边权重数据

    Parameters
    ----------
    path : str
        超边权重路径

    Returns
    -------
    DataFrame
        超边权重
    """
    try:
        if path.endswith('.pkl'):
            return read_pickle(path)
        else:
            return read_csv(path)
    except FileNotFoundError:
        print(f"File not found: {path}, Please check the file path.")
        return DataFrame()  # 返回一个空的DataFrame


def load_hypergraph_data(e: str, w: str = None) -> tuple[DataFrame, DataFrame] | DataFrame:
    """加载超图数据

    Parameters
    ----------
    e : str
        超图边数据路径
    w : str, optional
        超图权重数据路径，默认权重为 None

    Returns
    -------
    tuple[DataFrame, DataFrame] | DataFrame
        如果提供了权重数据路径，则返回一个包含边数据和权重数据的元组；
        否则仅返回边数据。
    """
    if w is not None:
        try:
            return load_edge_data(e), load_weight_data(w)
        except FileNotFoundError:
            print(f"File not found: {e} or {w}, Please check the file path.")
            return DataFrame(), DataFrame()  # 返回两个空的DataFrame
    else:
        return load_edge_data(e)


def merge_ew(e: DataFrame, w: DataFrame) -> DataFrame:
    """合并超图边数据和权重数据

    Parameters
    ----------
    e : DataFrame
        超图边数据
    w : DataFrame
        超图权重数据

    Returns
    -------
    DataFrame
        合并后的超图数据
    """
    # 仅使用出口数据（reporterISO 为出口国）的贸易额作为国家贸易额
    exporter_sum = w.groupby('reporterISO')['primaryValue'].sum()

    weighted_hyperedges = []

    if 'agreement' not in e.columns or 'country' not in e.columns:
        raise ValueError(
            "e must contain 'agreement' and 'country' columns")

    for agr, grp in e.groupby('agreement'):
        depth = grp['depth'].iloc[0]
        countries = grp['country'].unique().tolist()
        trades = [float(exporter_sum.get(c, 0.0)) for c in countries]
        total = sum(trades)
        if total <= 0:
            continue
        weights = {c: (float(exporter_sum.get(c, 0.0)) / total)*depth
                   for c in countries}
        for c, weight in weights.items():
            weighted_hyperedges.append(
                {'agreement': agr, 'country': c, 'weight': weight, 'depth': depth})

    return DataFrame(weighted_hyperedges)


def get_hyperedge_weights(H):
    """获取超图中每条边的权重（即边内所有节点关联权重之和）。

    Parameters:
        H (hypernetx.Hypergraph): 已构建好的超图对象。

    Returns:
        dict: {边名称: 边权重}
    """
    # 1. 获取关联矩阵 (行=节点, 列=边，自动包含构建时 cell_weight_col 的权重)
    B, _, edge_names = H.incidence_matrix(
        index=True, weights=True)  # 返回稀疏矩阵，行=节点，列=边，值为权重

    # 2. 对行(axis=1)求和，兼容 scipy.sparse 稀疏矩阵和 numpy 稠密矩阵
    weights = np.asarray(B.sum(axis=0)).flatten()

    # 3. 组合成字典返回
    return dict(zip(edge_names, weights))


def convert_hypernetx_to_xgi(hnx_g):
    """将 HyperNetX 超图转换为 XGI 超图

    HyperNetX 和 XGI 都是用列表的列表来表示超边

    Parameters
    ----------
    hnx_g : hypernetx.Hypergraph
        输入的 HyperNetX 超图对象

    Returns
    -------
    xgi.Hypergraph
        转换后的 XGI 超图对象
    """
    # 获取所有节点并转换为整数索引
    nodes = list(hnx_g.nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for i, node in enumerate(nodes)}

    edge_weights = get_hyperedge_weights(hnx_g)  # 获取超边权重

    # 转换超边：每个超边的节点列表转换为索引
    xgi_edges = []
    for edge in hnx_g.edges():
        edge_nodes = list(hnx_g.edges[edge])
        edge_indices = [node_to_idx[node] for node in edge_nodes]
        edge_indices.append(edge_weights.get(edge, 1.0))  # 添加权重，默认为1.0
        xgi_edges.append(tuple(edge_indices))

    # 创建 XGI 超图
    H_xgi = xgi.Hypergraph()
    H_xgi.add_weighted_edges_from(xgi_edges)

    return H_xgi, idx_to_node


def convert_xgi_to_hypernetx(xgi_hg):
    """将 XGI 超图 转换为 HyperNetX 的超图

    Parameters
    ----------
    xgi_hg : xgi.core.Hypergraph
        输入的 XGI 超图对象

    Returns
    -------
    hypernetx.Hypergraph
        转换后的 HyperNetX 超图对象
    """
    # 获取 XGI 超图的边信息
    hyperedges = {}

    # 兼容不同版本的 XGI
    edges_obj = xgi_hg.edges
    members_attr = getattr(edges_obj, "members")
    if callable(members_attr):
        edges_dict = members_attr()
    else:
        edges_dict = members_attr

    # 构建 HyperNetX 格式的超边字典
    # 处理 edges_dict 可能是列表或字典的情况
    if isinstance(edges_dict, dict):
        for edge_id, nodes in edges_dict.items():
            hyperedges[edge_id] = list(nodes)
    elif isinstance(edges_dict, list):
        # 如果是列表，自动生成边ID
        for idx, nodes in enumerate(edges_dict):
            hyperedges[f'e{idx}'] = list(nodes)
    else:
        # 尝试其他方式获取边信息
        for edge_id in xgi_hg.edges:
            nodes = xgi_hg.edges.members(edge_id)
            hyperedges[edge_id] = list(nodes)

    return hnx.Hypergraph(hyperedges)


def convert_hypernetx_to_hypergraphx(hnx_g):
    """将 HyperNetX 超图对象转换为 HyperGraphX 超图对象

    Parameters
    ----------
    hnx_hypergraph: hnx.Hypergraph
        HyperNetX 超图对象

    Returns
    -------
    hgx.Hypergraph
        HyperGraphX 超图对象
    """
    edge_list = []
    weights = get_hyperedge_weights(hnx_g)  # 获取超边权重
    valid_weights = []
    for edge_id in hnx_g.edges:
        edge_nodes = tuple(hnx_g.edges[edge_id])
        if len(edge_nodes) >= 2:
            edge_list.append(edge_nodes)
            valid_weights.append(weights.get(edge_id, 1.0))  # 默认权重为1.0
    return hgx.Hypergraph(edge_list, weighted=True, weights=list(valid_weights))


def convert_hypergraphx_to_hypernetx(hgx_hypergraph):
    """将 HyperGraphX 超图对象转换为 HyperNetX 超图对象

    Parameters
    ----------
    hgx_hypergraph: hgx.Hypergraph
        HyperGraphX 超图对象

    Returns
    -------
    hnx.Hypergraph
        HyperNetX 超图对象
    """
    if not isinstance(hgx_hypergraph, hgx.Hypergraph):
        raise TypeError("输入必须是 hypergraphx.Hypergraph 对象")

    edge_dict = {}
    for edge in hgx_hypergraph.edges:
        edge_id = f"e{len(edge_dict) + 1}"
        edge_dict[edge_id] = list(edge)

    return hnx.Hypergraph(edge_dict)


def convert_hypernetx_to_nxgraph(hnx_hypergraph):
    """将 HyperNetX 超图对象转换为 NetworkX 图对象

    Parameters
    ----------
    hnx_hypergraph: hnx.Hypergraph
        HyperNetX 超图对象

    Returns
    -------
    nx.Graph
        NetworkX 图对象
    """
    if not isinstance(hnx_hypergraph, hnx.Hypergraph):
        raise TypeError("输入必须是 hypernetx.Hypergraph 对象")

    # 将超图的边转换为 NetworkX 图的边
    edges = []
    for edge in hnx_hypergraph.edges:
        nodes = list(hnx_hypergraph.edges[edge])
        if len(nodes) >= 2:
            # 将超边中的节点两两组合成图的边
            edges.extend([(nodes[i], nodes[j])
                          for i in range(len(nodes)) for j in range(i + 1, len(nodes))])

    # 创建 NetworkX 图对象
    import networkx as nx
    G_nx = nx.Graph()
    G_nx.add_edges_from(edges)

    return G_nx


def hypernetx_to_hat_directed(hnx_hg):
    """将 HyperNetX 的无向超图转换为有向 HAT 超图。

    对于 HyperNetX 中的每条无向边，将其转换为有向 HAT 超边，
    其中所有节点同时属于 head 和 tail 集合，表示边内节点间的双向影响关系。

    Parameters
    ----------
    hnx_hg : hypernetx.Hypergraph
        输入的 HyperNetX 无向超图。

    Returns
    -------
    HAT.Hypergraph
        转换后的有向 HAT 超图。
    """
    unique_nodes = sorted(hnx_hg.dataframe['nodes'].unique())
    node_df = DataFrame({
        'Nodes': np.arange(len(unique_nodes)),
        'Names': unique_nodes
    })
    name_to_idx = {name: idx for idx, name in enumerate(unique_nodes)}

    edge_nodes_list = []
    head_list = []
    tail_list = []
    for _, group in hnx_hg.dataframe.groupby('edges'):
        node_names = list(group['nodes'].unique())
        node_idxs = sorted([name_to_idx[n] for n in node_names])
        edge_nodes_list.append(node_idxs)
        head_list.append(node_idxs)
        tail_list.append(node_idxs)

    edge_df = DataFrame({
        'Nodes': edge_nodes_list,
        'head': head_list,
        'tail': tail_list,
        'Edges': np.arange(len(edge_nodes_list))
    })

    return HAT.Hypergraph(
        nodes=node_df,
        edges=edge_df,
        directed=True,
        compress=False
    )
