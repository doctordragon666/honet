"""网络攻击内容

包括采样，执行攻击，生成攻击序列等功能
"""

import random
from .metric.nodemetric import excess_average_degree, clustering_coefficient, eigenvector_centrality
from .metric.globalmetric import calc_natural_connectivity
from .third import hnx

# ####################### 生成攻击序列 #######################


def random_attack_sequence(hg, target='node'):
    """生成随机攻击序列

    Parameters
    ----------
    hg: 
        超图对象
    target: 
        攻击目标 ('node' 或 'edge')

    Returns
    -------
    list:
        随机攻击序列
    """
    if target != 'node':
        # items = list(hg.nodes)
    # else:
        items = list(hg.edges)

    if target == 'node':
        degree_dict = {node: hg.degree(node) for node in hg.nodes}
        sorted_nodes = sorted(degree_dict.items(),
                              key=lambda x: x[1], reverse=False)
        return [node for node, _ in sorted_nodes]
    random.shuffle(items)
    return items


def degree_attack_sequence(hg, target='node'):
    """生成基于度的攻击序列

    Parameters:
    ----------
    hg : hnx.Hypergraph
        超图对象
    target : str, optional
        攻击目标 ('node' 或 'edge')

    Returns:
    ----------
    list
        基于度的攻击序列
    """
    if target == 'node':
        degree_dict = {node: hg.degree(node) for node in hg.nodes}
        sorted_nodes = sorted(degree_dict.items(),
                              key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes]
    elif target == 'edge':
        degree_dict = {edge: len(hg.edges.elements[edge]) for edge in hg.edges}
        sorted_edges = sorted(degree_dict.items(),
                              key=lambda x: x[1], reverse=True)
        return [edge for edge, _ in sorted_edges]
    elif target == 'excess':
        # 余平均度
        excess_dict = excess_average_degree(hg)
        sorted_excess = sorted(excess_dict.items(),
                               key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_excess]
    elif target == 'cluster':
        # 聚类系数
        # import honmetric as hm
        cluster_dict = clustering_coefficient(hg)
        sorted_items = sorted(cluster_dict.items(),
                              key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_items]


def centrality_attack_sequence(hg, metric='betweenness') -> list:
    """基于中心性的攻击序列

    生成包括介数中心性和特征向量中心性的攻击序列

    Parameters
    ----------
    hg : hnx.Hypergraph
        超图对象
    metric : str, optional
        中心性指标, 可选值为 'betweenness' 或 'eigenvector', 默认为 'betweenness'

    Returns
    -------
    list
        返回基于中心性的攻击序列
    """
    if metric == 'betweenness':
        # 介数中心性
        result = hnx.s_betweenness_centrality(hg, edges=False)
        return [node for node, _ in sorted(result.items(), key=lambda x: x[1], reverse=True)]
    elif metric == 'eigenvector':
        # 特征向量中心性
        result = eigenvector_centrality(hg)
        return [node for node, _ in sorted(result.items(), key=lambda x: x[1], reverse=True)]


def main_country(attack_sample, num=None, partion=None) -> list:
    """输出攻击样本中最重要的前num个或前partion比例的节点

    Parameters
    ----------
    attack_sample : list or dict
        攻击样本，可以是节点列表或字典形式的攻击样本
    num : int, optional
        输出的节点数量，默认为None
    partion : float, optional
        输出的节点比例，默认为None

    Returns
    -------
    list
        返回选择的节点的uid列表
    """
    if num == None:
        if partion != None:
            num = int(len(attack_sample) * partion)
        else:
            num = 10  # 默认输出前10个重要节点或超边
    else:
        if partion != None:
            print("不可同时设置num和partion参数，请选择其中一个进行设置")

    # 返回选择的节点的uid列表
    if isinstance(attack_sample, dict):
        return list(attack_sample.values())[:num]
    else:
        return attack_sample[:num]


def execute_attack(hg, attack_sequence, target='node', func=None, rho=0.01):
    """执行攻击

    Parameters
    ----------
    hg : hnx.Hypergraph
        超图对象
    attack_sequence : list or dict
        攻击序列，可以是节点列表或字典形式的攻击样本
    target : str, optional
        攻击目标 ('node' 或 'edge')
    func : callable, optional
        如果为None，自动使用自然连通度来衡量当前网络的情况
    rho : float, optional
        每次攻击的比例，默认为0.01

    Returns
    -------
    dict
        返回攻击结果的字典，格式为 {策略名称: {steps: nc_vals}}
    """
    results = {}  # 存储结果的字典，格式为 {策略名称: {steps: nc_vals}}
    if not isinstance(attack_sample, dict):
        attack_sample = {'默认策略': attack_sample}

    if hg is None or not isinstance(hg, hnx.Hypergraph):
        return results  # 如果hg为空或不是Hypergraph对象，直接返回空结果
    for strategy_name, attack_sequence in attack_sample.items():
        nc_vals = [1.0]  # 存储每次攻击后的自然连通度
        steps = [0.0]  # 存储已经攻击的数量，存储的是比例
        hg_copy = hg.copy()  # 创建超图的副本，以便在攻击过程中修改

        while len(attack_sequence) > 0:
            # 执行攻击
            if target == 'node':
                num_to_remove = max(1, int(len(hg_copy.nodes) * rho))
                nodes_to_remove = attack_sequence[:num_to_remove]
                hg_copy.restrict_to_nodes(nodes_to_remove)
                attack_sequence = attack_sequence[num_to_remove:]
            elif target == 'edge':
                num_to_remove = max(1, int(len(hg_copy.edges) * rho))
                edges_to_remove = attack_sequence[:num_to_remove]
                hg_copy.restrict_to_edges(edges_to_remove)
                attack_sequence = attack_sequence[num_to_remove:]

            # 计算当前网络的自然连通度
            if func is None:
                nc_val = calc_natural_connectivity(hg_copy)
            else:
                nc_val = func(hg_copy)

            # 存储结果
            nc_vals.append(nc_val)
            steps.append(len(hg.nodes) - len(hg_copy.nodes)/len(hg.nodes) if target ==
                         'node' else len(hg.edges) - len(hg_copy.edges)/len(hg.edges))

            if nc_val == 0:
                break  # 如果自然连通度为0，提前结束攻击

        results[strategy_name] = {'steps': steps, 'nc_vals': nc_vals}
    return results


def basic_attack_sequence(g) -> dict:
    """生成普通图的攻击序列，包括基于度(不分入度和出度，就用普通的)、余平均度、聚类系数，介数中心性，特征向量中心性
    按照从大到小的顺序排列，返回一个字典，键为攻击策略名称，值为对应的攻击序列列表。

    Parameters
    ----------
    g : nx.Graph
        输入的普通图对象

    Returns
    -------
    dict
        返回一个字典，键为攻击策略名称，值为对应的攻击序列列表。
    """
    import networkx as nx
    degree_dict = dict(g.degree())
    excess_dict = nx.average_neighbor_degree(g)
    clustering_dict = nx.clustering(g)
    betweenness_dict = nx.betweenness_centrality(g)
    eigenvector_dict = nx.eigenvector_centrality(g)

    def _sort_desc(d):
        return [k for k, _ in sorted(d.items(), key=lambda x: x[1], reverse=True)]
    attack_sequences = {
        '度': _sort_desc(degree_dict),
        '余平均度': _sort_desc(excess_dict),
        '聚类系数': _sort_desc(clustering_dict),
        '介数': _sort_desc(betweenness_dict),
        '特征向量中心性': _sort_desc(eigenvector_dict),
    }
    return attack_sequences
