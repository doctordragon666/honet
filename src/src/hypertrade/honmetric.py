"""计算高阶网络指标并输出

提供批量导出，支持同配系数的测试
"""

from hypertrade.core import DataFrame
from hypertrade import honcommunity as hc
from .metric.globalmetric import calc_reducibility, calc_entropy, calc_clustering_coefficient_g, calc_assortativity
from .metric.nodemetric import excess_average_degree
from .third import np, hnx
from hypernetx import dist_stats


def hnxs_global(hnxs: dict) -> DataFrame:
    """计算多个高阶网络的整体性质

    Parameters
    ----------
    hnxs : dict
        字典形式的超图对象

    Returns
    -------
    DataFrame
        返回多个超图的全局指标表格
    """
    result = DataFrame({
        year: {
            # '超图的均匀程度': dist_stats(g)['aspect ratio'],
            # '国家数量': len(g.nodes),
            # '贸易协定数量': len(g.edges),
            # '高阶网络平均度': dist_stats(g)['node degree centrality stats']['mean'],
            # '高阶网络平均聚类系数': calc_clustering_coefficient_g(g),
            # '高阶网络密度': dist_stats(g)['density'],
            # '高阶网络熵': calc_entropy(g),
            # '高阶网络模块度': hc.extract_community_partition(g, method=hc.Method.LAST_STEP)['qh'],
            # '高阶网络同配系数': calc_assortativity(g)[0],
            # '同配系数显著性': calc_assortativity(g)[1],
            '高阶网络可约性': calc_reducibility(g, weighted=True)
        }
        for year, g in hnxs.items()
    }).T
    # 国家和贸易协定数量只保留整数
    # result['国家数量'] = result['国家数量'].astype(int)
    # result['贸易协定数量'] = result['贸易协定数量'].astype(int)
    return result


def assortativity_test(H: hnx.Hypergraph):
    """测试该超图是否具有度相关性

    通过判断原始超图中度为k的节点的余平均度是否与k相关，
    如果与k为无关的常数，则认为不具有度相关性，否则认为具有度相关性。

    Parameters
    ----------
    H : Hypergraph
        输入的超图对象

    Returns
    -------
    dict
        包含度相关性的字典，键为度，值为余平均度
    """
    degree_dict = {node: H.degree(node) for node in H.nodes}
    excess_dict = excess_average_degree(H)

    # 计算相同度的节点
    from collections import defaultdict
    degree_excess_points = defaultdict(list)
    for node, degree in degree_dict.items():
        excess = excess_dict.get(node, 0)
        degree_excess_points[degree].append(excess)

    # 计算每个度的平均余平均度
    degree_avg_excess = {degree: np.mean(
        excesses) for degree, excesses in degree_excess_points.items()}

    return degree_avg_excess
