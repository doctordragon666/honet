import random
from . import np
from hypernetx.algorithms.contagion import threshold


def discrete_SIR_control(
    H,
    tau,
    gamma,
    transmission_function=threshold,
    initial_infecteds=None,
    initial_recovereds=None,
    rho=None,
    tmin=0,
    tmax=float("Inf"),
    dt=1.0,
    return_full_data=False,
    **args,
):
    """离散时间SIR模型，支持控制节点的设置。
    """

    if rho is not None and initial_infecteds is not None:
        raise Exception("Cannot define both initial_infecteds and rho")

    if initial_infecteds is None:
        if rho is None:
            initial_number = 1
        else:
            initial_number = int(round(len(H.nodes) * rho))
        initial_infecteds = random.sample(list(H.nodes), initial_number)
    else:
        initial_infecteds = list(
            set(H.nodes).intersection(set(initial_infecteds)))

    initial_recovereds = list(
        set(H.nodes).intersection(set(initial_recovereds)))

    from collections import defaultdict
    status = defaultdict(lambda: "S")

    if return_full_data:
        transition_events = dict()
        transition_events[tmin] = list()

    for node in initial_infecteds:
        status[node] = "I"
        if return_full_data:
            transition_events[tmin].append(("I", node, None))

    I = [len(initial_infecteds)]
    R = [0]
    S = [len(H.nodes) - I[-1]]

    t = tmin
    times = [t]
    newStatus = status.copy()

    def edge_neighbors(node): return H.nodes.memberships[node]

    while t < tmax and I[-1] != 0:
        S.append(S[-1])
        I.append(I[-1])
        R.append(R[-1])
        if return_full_data:
            transition_events[t + dt] = list()

        for node in H.nodes:
            if status[node] == "I":
                # recover the node. If it is not healed, it stays infected.
                if random.random() <= gamma * dt or node in initial_recovereds:
                    newStatus[node] = "R"
                    I[-1] += -1
                    R[-1] += 1
                    if return_full_data:
                        transition_events[t + dt].append(("R", node))
            elif status[node] == "S":
                for edge_id in edge_neighbors(node):
                    members = H.edges[edge_id]

                    if (
                        random.random()
                        <= tau[len(members)]
                        * transmission_function(node, status, members, **args)
                        * dt
                    ):
                        newStatus[node] = "I"
                        S[-1] += -1
                        I[-1] += 1
                        if return_full_data:
                            transition_events[t +
                                              dt].append(("I", node, edge_id))
                        break
                else:
                    newStatus[node] = "S"
        status = newStatus.copy()
        t += dt
        times.append(t)
    if return_full_data:
        return transition_events
    else:
        return np.array(times), np.array(S), np.array(I), np.array(R)


    
def RandomDiscrete_SIR(
    H,
    tau,
    gamma,
    transmission_function=threshold,
    initial_infecteds=None,
    initial_recovereds=None,
    rho=None,
    tmin=0,
    tmax=float("Inf"),
    dt=1.0,
    return_full_data=False,
    **args,
):
    """随机离散时间SIR模型，支持控制节点的设置。
    """
    
def RandomGillespie_SIR(
    H,
    tau,
    gamma,
    sigma1=0.1,
    sigma2=0.1,
    transmission_function=threshold,
    initial_infecteds=None,
    initial_recovereds=None,
    rho=None,
    tmin=0,
    tmax=float("Inf"),
    return_full_data=False,
    **args,
):
    """随机Gillespie SIR模型
    
    Parameters
    ----------
    H : hypernetx.Hypergraph
        超图对象，表示传播网络
    tau : dict
        传播率字典，键为超边大小，值为对应的传播率
    gamma : float
        恢复率
    sigma1 : float, optional
        感染率的噪声强度，取值范围为[0, 1]，默认为0.1
    sigma2 : float, optional
        恢复率的噪声强度，取值范围为[0, 1]，默认为0.1
    transmission_function : function, optional
        传播函数，默认为threshold函数
    initial_infecteds : list, optional
        初始感染节点列表，默认为None
    initial_recovereds : list, optional
        初始恢复节点列表，默认为None
    rho : float, optional
        初始感染节点比例，默认为None
    tmin : float, optional
        模拟开始时间，默认为0
    tmax : float, optional
        模拟结束时间，默认为无穷大
    return_full_data : bool, optional
        是否返回完整数据，默认为False
    **args : dict
        其他参数，传递给传播函数
    """
    # 初始化节点状态
    beta = np.mean(tau.values()) # beta为平均传播率
    nodes = list(H.nodes)
    N = len(nodes)

    # 随机选择初始感染者
    infected = np.random.choice(nodes, size=int(rho * N), replace=False)
    state = {node: 'I' if node in infected else 'S' for node in nodes}

    # 计算度分布
    degrees = {node: len(list(H.neighbors(node))) for node in nodes}

    # 初始化结果
    t = [tmin]
    s_count = sum(1 for node in nodes if state[node] == 'S')
    i_count = sum(1 for node in nodes if state[node] == 'I')
    r_count = sum(1 for node in nodes if state[node] == 'R')
    S = [s_count]
    I = [i_count]
    R = [r_count]

    current_time = tmin
    while current_time < tmax:
        # 计算Theta
        if i_count > 0:
            theta_numerator = sum(degrees[node]
                                  for node in nodes if state[node] == 'I')
            theta_denominator = sum(degrees.values())
            theta = theta_numerator / theta_denominator if theta_denominator > 0 else 0
        else:
            theta = 0

        # 计算所有可能事件的速率
        infection_rates = []
        recovery_rates = []

        for node in nodes:
            if state[node] == 'S':
                # 易感者感染速率
                rate = beta * degrees[node] * theta
                # 加入噪声扰动
                noise = np.random.normal(0, sigma1 * degrees[node] * theta)
                rate = max(0, rate + noise)
                infection_rates.append(rate)

            elif state[node] == 'I':
                # 感染者恢复速率
                rate = gamma
                # 加入噪声扰动
                noise = np.random.normal(0, sigma2)
                rate = max(0, rate + noise)
                recovery_rates.append(rate)

        total_infection_rate = sum(infection_rates)
        total_recovery_rate = sum(recovery_rates)
        total_rate = total_infection_rate + total_recovery_rate

        if total_rate <= 0:
            # 没有事件发生，跳出循环
            break

        # 生成事件时间
        tau = np.random.exponential(1 / total_rate)
        current_time += tau

        if current_time > tmax:
            break

        # 确定发生哪种事件
        event_type = np.random.choice(['infection', 'recovery'], p=[
                                      total_infection_rate/total_rate, total_recovery_rate/total_rate])

        if event_type == 'infection':
            # 选择一个易感者感染
            susceptible_nodes = [node for node in nodes if state[node] == 'S']
            if susceptible_nodes and total_infection_rate > 0:
                # 根据感染速率选择节点
                node_rates = [beta * degrees[node] *
                              theta for node in susceptible_nodes]
                node_probs = [
                    rate / total_infection_rate for rate in node_rates]
                # 确保概率和为1（处理浮点数精度问题）
                node_probs = [p / sum(node_probs) for p in node_probs]
                selected_node = np.random.choice(
                    susceptible_nodes, p=node_probs)
                state[selected_node] = 'I'

        else:
            # 选择一个感染者恢复
            infected_nodes = [node for node in nodes if state[node] == 'I']
            if infected_nodes:
                # 随机选择一个感染者
                selected_node = np.random.choice(infected_nodes)
                state[selected_node] = 'R'

        # 更新状态计数
        s_count = sum(1 for node in nodes if state[node] == 'S')
        i_count = sum(1 for node in nodes if state[node] == 'I')
        r_count = sum(1 for node in nodes if state[node] == 'R')

        # 记录结果
        t.append(current_time)
        S.append(s_count)
        I.append(i_count)
        R.append(r_count)

    return t, S, I, R