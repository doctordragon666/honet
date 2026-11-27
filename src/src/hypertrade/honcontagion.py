"""高阶网络动力学内容

"""

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from .third import np, hnx
from hypernetx.algorithms.contagion import collective_contagion, individual_contagion, threshold, discrete_SIS, majority_vote, Gillespie_SIS, Gillespie_SIR, discrete_SIR
from enum import unique, auto, Enum
from .metric.edgemetric import get_avg_degree_by_edge_size
from .contagion.honSIS import discrete_SIS_control, RandomDiscrete_SIS, RandomGillespie_SIS
from .contagion.honSIR import discrete_SIR_control, RandomDiscrete_SIR, RandomGillespie_SIR


@unique
class Contagion_Method(Enum):
    """高阶网络动力学方法枚举类

    Attributes
    ----------
    Gillespie : auto()
        Gillespie方法
    RandomGillespie : auto()
        随机Gillespie方法
    Discrete : auto()
        离散时间方法
    RandomDiscrete : auto()
        随机离散时间方法
    """
    Gillespie = auto()
    RandomGillespie = auto()
    Discrete = auto()
    RandomDiscrete = auto()


def generate_tau(H, beta: list, type='fixed'):
    """根据超边大小生成感染率tau
    """
    if beta is None or len(beta) == 0:
        raise ValueError("beta参数不能为空或长度为0")
    if type == 'fixed':
        # 所有节点的感染率都相等
        beta = beta[0]
        tau = {int(size): beta
               for size in set(H.size(edge) for edge in H.edges)}
    elif type == 'fixed_compare':
        # m=2的感染率不等于其他超边大小的感染率，但其他超边的感染率固定
        tau = {int(size): beta[1]
               for size in set(H.size(edge) for edge in H.edges)}
        tau[2] = beta[0]  # 对于超边大小为2的感染率进行调整
    elif type == 'size_based':
        # 所有超边的感染率与超边大小成正比，beta[0]为比例系数
        beta = beta[0]
        tau = {int(size): beta*int(size)
               for size in set(H.size(edge) for edge in H.edges)}
    elif type == 'size_based_compare':
        # m=2的感染率不等于其他超边大小的感染率，但其他超边的感染率与超边大小成正比，beta[0]为比例系数
        tau = {int(size): beta[1]*int(size)
               for size in set(H.size(edge) for edge in H.edges)}
        tau[2] = beta[0]  # 对于超边大小为2的感染率进行调整
    elif type == 'sum_based':
        # 所有超边的感染率的和为beta[0]，并且与超边大小成正比
        total_size = sum(set(H.size(edge) for edge in H.edges))
        tau = {int(size): beta[0]*int(size)/total_size
               for size in set(H.size(edge) for edge in H.edges)}
    elif type == 'sum_based_compare':
        # m=2的感染率不等于其他超边大小的感染率，但其他超边的感染率的和为beta[1]，并且与超边大小成正比
        total_size = sum(set(H.size(edge) for edge in H.edges))
        tau = {int(size): beta[1]*int(size)/total_size
               for size in set(H.size(edge) for edge in H.edges)}
        tau[2] = beta[0]  # 对于超边大小为2的感染率进行调整
    else:
        raise ValueError(f"Unknown type: {type}")
    return tau


def is_stable(H, tau, gamma):
    """判断高阶网络传播模型是否稳定

    等式右边可以看成基本再生数，去掉gamma
    """
    stats = hnx.dist_stats(H)
    k_lst = stats['node degree list']
    n = len(k_lst)
    if n == 0:
        return True

    k_mean = sum(k_lst) / n
    k_mean_sq = get_avg_degree_by_edge_size(H).get(
        2, 0.0)  # 使用超边大小为 2 的平均度数作为 k_mean_sq
    if k_mean_sq == 0:
        return True

    RHS = gamma * k_mean / k_mean_sq

    LHS = 0.0
    for m, count in get_avg_degree_by_edge_size(H).items():
        if m < 2:
            continue
        beta_m = tau.get(m, 0.0)
        if beta_m == 0:
            continue
        LHS += (m - 1) * beta_m * count / k_mean

    return LHS < RHS


def _parse_strategy(strategy):
    """根据传入的策略字符串返回对应的传播函数
    """
    if strategy == 'individual':
        return individual_contagion
    elif strategy == 'collective':
        return collective_contagion
    elif strategy == 'default' or strategy == 'threshold':
        return threshold
    elif strategy == 'majority_vote':
        return majority_vote
    # 这些参数是写给discrete_SIS_control区分的，还是采用同样的策略。
    elif strategy == 'pulse':
        # 脉冲控制
        return individual_contagion  # 这里可以根据需要返回不同的函数
    elif strategy == 'target':
        # 目标控制
        return individual_contagion  # 这里可以根据需要返回不同的函数
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def honSIS(H, gamma: float, tau: dict, method=Contagion_Method.Discrete, initial_infection=None, strategy='collective', tmax=1000, seed=42, return_all_data=False, control_nodes: list = None) -> tuple:
    """SIS模型接口函数

    根据传入的method参数选择不同的SIS模型算法

    Parameters
    -----------
    H : hypernetx.Hypergraph
        超图结构
    gamma : float, optional
        恢复率，默认为0.3
    tau : dict, optional
        感染率，键为超边大小，值为对应的感染率
    method : Contagion_Method, optional
        选择SIS模型算法，默认为Contagion_Method.Discrete
    initial_infection : list, optional
        初始感染者列表，默认为None

    Returns
    --------
    t : list
        时间点列表
    S : list
        易感者数量历史
    I : list
        感染者数量历史
    """
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))

    dt = 1  # 离散时间步长

    if method == Contagion_Method.Discrete:
        if control_nodes is not None:
            return discrete_SIS_control(
                H, tau, gamma, strategy, rho=0.1, tmin=0, tmax=tmax, dt=dt, initial_recovereds=control_nodes, return_full_data=return_all_data)

        if initial_infection is not None:
            return discrete_SIS(
                H, tau, gamma, tmin=0, tmax=tmax, dt=dt, initial_infecteds=initial_infection, transmission_function=_parse_strategy(strategy),
                return_full_data=return_all_data)
        else:
            return discrete_SIS(
                H, tau, gamma, rho=0.1, tmin=0, tmax=tmax, dt=dt, transmission_function=_parse_strategy(strategy),
                return_full_data=return_all_data)

    elif method == Contagion_Method.Gillespie:
        if initial_infection is not None:
            return Gillespie_SIS(H, tau, gamma, transmission_function=_parse_strategy(strategy), initial_infecteds=initial_infection, tmax=tmax, return_all_data=return_all_data)
        else:
            return Gillespie_SIS(H, tau, gamma, transmission_function=_parse_strategy(strategy), rho=0.1, tmax=tmax, return_all_data=return_all_data)

    elif method == Contagion_Method.RandomDiscrete:
        return RandomDiscrete_SIS(
            H, tau, gamma, rho=0.1, tmax=tmax, dt=dt, transmission_function=_parse_strategy(strategy),
            return_full_data=return_all_data)

    elif method == Contagion_Method.RandomGillespie:
        return RandomGillespie_SIS(H, tau, gamma, transmission_function=_parse_strategy(strategy), rho=0.1, tmax=tmax, return_all_data=return_all_data)


def honSIR(H, gamma: float, tau: dict, method=Contagion_Method.Discrete, initial_infection=None, initial_recovereds=None, strategy='collective', tmax=1000, seed=42, return_all_data=False) -> tuple:
    """SIR模型接口函数

    根据传入的method参数选择不同的SIR模型算法

    Parameters
    -----------
    H : hypernetx.Hypergraph
        超图结构
    gamma : float, optional
        恢复率，默认为0.3
    tau : dict, optional
        感染率，键为超边大小，值为对应的感染率
    method : Contagion_Method, optional
        选择SIR模型算法，默认为Contagion_Method.Discrete
    initial_infection : list, optional
        初始感染者列表，默认为None
    initial_recovereds : list, optional
        初始康复者列表，默认为None

    Returns
    --------
    t : list
        时间点列表
    S : list
        易感者数量历史
    I : list
        感染者数量历史
    R : list
        康复者数量历史
    """
    dt = 1  # 离散时间步长

    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))

    if method == Contagion_Method.Discrete:
        if initial_recovereds is not None:
            return discrete_SIR_control(
                H, tau, gamma, _parse_strategy(strategy), initial_infecteds=initial_infection, initial_recovereds=initial_recovereds, tmin=0, tmax=tmax, dt=dt, return_full_data=return_all_data)

        if initial_infection is None:
            return discrete_SIR(
                H, tau, gamma, rho=0.1, tmin=0, tmax=tmax, dt=dt, transmission_function=_parse_strategy(strategy), return_full_data=return_all_data)
        else:
            return discrete_SIR(
                H, tau, gamma, tmin=0, tmax=tmax, dt=dt, initial_infecteds=initial_infection, transmission_function=_parse_strategy(strategy), return_full_data=return_all_data)

    elif method == Contagion_Method.Gillespie:
        if initial_infection is None:
            return Gillespie_SIR(H, tau, gamma, transmission_function=_parse_strategy(strategy), rho=0.1, tmax=tmax, return_full_data=return_all_data)
        else:
            return Gillespie_SIR(H, tau, gamma, transmission_function=_parse_strategy(strategy), initial_infecteds=initial_infection, tmax=tmax, return_full_data=return_all_data)

    elif method == Contagion_Method.RandomDiscrete:
        return RandomDiscrete_SIR(
            H, tau, gamma, rho=0.1, tmax=tmax, dt=dt, transmission_function=_parse_strategy(strategy), return_full_data=return_all_data)
    elif method == Contagion_Method.RandomGillespie:
        return RandomGillespie_SIR(
            H, tau, gamma, transmission_function=_parse_strategy(strategy), rho=0.1, tmax=tmax, return_full_data=return_all_data)


def classify_point(beta2, beta_higher, gamma_global, H_global, kind, function=None):
    """根据传入的参数分类相图点
    """
    if function is not None:
        tau = function(H_global, beta=[float(
            beta2), float(beta_higher)], type=kind)
    else:
        tau = generate_tau(H_global, beta=[float(
            beta2), float(beta_higher)], type=kind)
    return is_stable(H_global, tau, gamma=gamma_global)


def sis_phase(
    H,
    gamma=0.3,
    beta2_lst=None,
    beta_higher_lst=None,
    kind='fixed',
    function=None
):
    """提供SIS相图数据

    该函数用于计算并返回SIS（可能是某种自旋系统或传染病模型）的相图数据。
    通过并行计算不同参数组合下的相态分类，生成相图矩阵。

    """
    # 将输入参数beta2_lst和beta_higher_lst转换为numpy数组，并指定数据类型为浮点数
    beta2_lst = np.asarray(beta2_lst, dtype=float)
    beta_higher_lst = np.asarray(beta_higher_lst, dtype=float)

    # 初始化相图矩阵，维度为(len(beta_higher_lst), len(beta2_lst))，数据类型为整数
    phase_map = np.zeros(
        (len(beta_higher_lst), len(beta2_lst)), dtype=int)

    # 创建工作项列表，每个元素是一个元组，包含索引iy、ix和对应的beta2、beta_higher值
    work_items = [
        (iy, ix, float(beta2), float(beta_higher))
        for iy, beta_higher in enumerate(beta_higher_lst)
        for ix, beta2 in enumerate(beta2_lst)
    ]

    # 设置线程池的最大工作线程数，最多32个，至少1个，根据工作项数量动态调整
    max_workers = min(32, max(1, len(work_items)))
    # 使用线程池执行器进行并行计算
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务到线程池，并创建future到索引的映射
        future_map = {
            executor.submit(classify_point, beta2, beta_higher, gamma, H, kind, function): (iy, ix)
            for iy, ix, beta2, beta_higher in work_items
        }
        # 遍历已完成的任务，并将结果存储到相图矩阵的相应位置
        for future in as_completed(future_map):
            iy, ix = future_map[future]
            phase_map[iy, ix] = int(future.result())

    # 返回计算得到的相图矩阵
    return phase_map
