"""计算高阶网络驱动节点

提供驱动节点计算，检测超边膨胀和可达性分析的功能。
"""

from itertools import combinations
import random
import pandas as pd
import networkx as nx
from .core import hypernetx_to_hat_directed
from .third import HAT, np


def brute_force_min_driver_sets(HG, verbose=False):
    """暴力搜索最小驱动节点集合。
    
    Parameters
    ----------
    HG : HAT.Hypergraph
        输入的有向超图。
    verbose : bool
        是否输出详细信息。

    Returns
    -------
    tuple
        (最小驱动节点集合的大小, 所有最小驱动节点集合)
    """
    n = HG.nnodes
    all_sys_nodes = set(range(n))

    if verbose:
        print("Start brute-forcing...\n")

    min_size = None
    min_sets = []

    for k in range(1, n+1):
        if verbose:
            print(f"=== Trying driver set size k = {k} ===")

        found_k = []

        for subset in combinations(range(n), k):
            drivers = set(subset)

            # ------------------------------
            # Step 1: dilation check
            # ------------------------------
            if detect_dilation_with_given_drivers(HG, drivers):
                continue
                # if verbose:
                #    print("  skip (dilation) drivers =", sorted(drivers))
                # continue

            # ------------------------------
            # Step 2: accessibility check
            # ------------------------------
            reachable = hypergraph_reachable(HG, drivers)
            reachable_sys = reachable & all_sys_nodes

            if reachable_sys == all_sys_nodes:
                found_k.append(drivers)
                if verbose:
                    print("  Feasible driver set:", sorted(drivers))

        if found_k:
            min_size = k
            min_sets = found_k
            if verbose:
                print(f"\n>>> Minimum driver size = {k}")
                for s in min_sets:
                    print("    ", sorted(s))
            break

    if min_size is None and verbose:
        print("No feasible driver set found: either dilation exists or accessibility cannot be satisfied.")

    return min_size, min_sets


def add_node_fields(HG_local):
    """为超图的节点 DataFrame 添加 edges、head、tail字段，记录每个节点参与的边及其角色。

    Parameters
    ----------
    HG_local : HAT.Hypergraph
        输入的有向超图，其 edges 属性需包含'Edges'、'Nodes'、'head'、'tail'列。

    Returns
    -------
    HAT.Hypergraph
        添加字段后的超图对象。
    """
    HG_local.nodes["edges"] = [[] for _ in range(len(HG_local.nodes))]
    HG_local.nodes["head"] = [[] for _ in range(len(HG_local.nodes))]
    HG_local.nodes["tail"] = [[] for _ in range(len(HG_local.nodes))]
    node_to_idx = dict(zip(HG_local.nodes["Nodes"], HG_local.nodes.index))
    for _, edge_row in HG_local.edges.iterrows():
        edge_id = edge_row["Edges"]
        for n in edge_row["Nodes"]:
            HG_local.nodes.at[node_to_idx[n], "edges"].append(edge_id)
        for n in edge_row["head"]:
            HG_local.nodes.at[node_to_idx[n], "head"].append(edge_id)
        for n in edge_row["tail"]:
            HG_local.nodes.at[node_to_idx[n], "tail"].append(edge_id)
    return HG_local


def compute_drivers_from_hypernetx(hnx_hg, out_csv="drivers_from_hnx.csv", HAT_type=False):
    """从 HyperNetX 超图出发，转换为有向 HAT 超图后计算驱动节点集合并导出 CSV。

    Parameters
    ----------
    hnx_hg : hypernetx.Hypergraph
        输入的 HyperNetX 超图。
    out_csv : str
        输出 CSV 文件路径。

    Returns
    -------
    dict
        包含算法名称、dilation 标记和驱动节点列表的字典。
    """
    if not HAT_type:
        HG = hypernetx_to_hat_directed(hnx_hg)
    HG = add_node_fields(HG)
    return _compute_drivers_on_hat(HG, out_csv)


def _compute_drivers_on_hat(HG, out_csv="drivers.csv"):
    """在已构建好的有向 HAT 超图上执行驱动节点计算。

    Parameters
    ----------
    HG : HAT.Hypergraph
        已添加节点字段的有向 HAT 超图。
    out_csv : str
        输出 CSV 文件路径。

    Returns
    -------
    dict
        包含算法名称、dilation 标记和驱动节点列表的字典。
    """
    node_count = HG.nnodes

    if node_count <= 8:
        algorithm = "brute_force"
        all_nodes = set(range(node_count))
        best_set = None
        dilation_exists = False
        for size in range(1, node_count + 1):
            for subset in combinations(range(node_count), size):
                drivers = set(subset)
                if detect_dilation_with_given_drivers(HG, drivers):
                    continue
                reachable = hypergraph_reachable(HG, drivers)
                if (reachable & all_nodes) == all_nodes:
                    best_set = drivers
                    break
            if best_set is not None:
                break
        if best_set is None:
            dilation_exists = True
            drivers_final = set()
        else:
            drivers_final = best_set
    else:
        dilation_exists, matched_nodes, driver_nodes, _ = detect_dilation_and_drivers(
            HG, node_count)
        all_nodes = set(range(HG.nnodes))
        dilation_exists_sub = detect_dilation_with_given_drivers(HG, driver_nodes)
        if dilation_exists_sub:
            print("超图完全不可控")
        
        initial_drivers = set(driver_nodes)
        if node_count <= 30:
            algorithm = "matching_plus_greedy"
            drivers_final = greedy_accessibility(HG, initial_drivers)
        else:
            algorithm = "matching_only"
            drivers_final = initial_drivers

    drivers_sorted = sorted(list(drivers_final))
    if drivers_sorted:
        df = pd.DataFrame({"driver_node": drivers_sorted})
        df["algorithm"] = algorithm
        df["dilation_exists"] = dilation_exists
    else:
        df = pd.DataFrame(
            columns=["driver_node", "algorithm", "dilation_exists"])
    if out_csv:
        df.to_csv(out_csv, index=False)
    return {"algorithm": algorithm, "dilation_exists": dilation_exists, "drivers": drivers_sorted}


def random_uniform_hypergraph(n_local, k_local, m_local):
    """生成随机 k-均匀有向超图。

    Parameters
    ----------
    n_local : int
        节点总数。
    k_local : int
        每条边 tail 的大小。
    m_local : int
        需要生成的边数。

    Returns
    -------
    list of dict
        每条边包含 'head' 和 'tail' 列表。
    """
    edge_set = set()
    edge_list_local = []
    while len(edge_list_local) < m_local:
        tail = tuple(sorted(random.sample(range(n_local), k_local)))
        remaining = list(set(range(n_local)) - set(tail))
        if not remaining:
            continue
        hsize = random.randint(1, min(k_local - 1, len(remaining)))
        head = tuple(sorted(random.sample(remaining, hsize)))
        edge = (head, tail)
        if edge not in edge_set:
            edge_set.add(edge)
            edge_list_local.append({"head": list(head), "tail": list(tail)})
    return edge_list_local


def edge_list_to_incidence(edge_list_local):
    """将 head/tail 格式的边列表转换为关联矩阵。

    Parameters
    ----------
    edge_list_local : list of dict
        每条边包含 'head' 和 'tail' 列表。

    Returns
    -------
    np.ndarray
        关联矩阵，head 节点为 1，tail 节点为 -1。
    """
    nodes = set()
    for e in edge_list_local:
        nodes.update(e["head"])
        nodes.update(e["tail"])
    nodes = sorted(list(nodes))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n_local = len(nodes)
    m_local = len(edge_list_local)
    incidence = np.zeros((n_local, m_local), dtype=int)
    for j, e in enumerate(edge_list_local):
        for h in e["head"]:
            incidence[node_to_idx[h], j] = 1
        for t in e["tail"]:
            incidence[node_to_idx[t], j] = -1
    return incidence


def star_graph(HG):
    """构建超图的星形图（二分图）表示。

    Parameters
    ----------
    HG : HAT.Hypergraph
        有向超图。

    Returns
    -------
    nx.DiGraph or nx.Graph
        星形图，边节点 (e*) 连接系统节点 (n*)。
    """
    G = nx.DiGraph() if HG.directed else nx.Graph()
    system_nodes = [f"n{i}" for i in range(HG.nnodes)]
    G.add_nodes_from(system_nodes, bipartite=0)

    for edge_id in HG.edges.index:
        e_node = f"e{edge_id}"
        G.add_node(e_node, bipartite=1)

        heads = HG.edges.loc[edge_id, "head"]
        tails = HG.edges.loc[edge_id, "tail"]
        heads = [f"n{h}" for h in heads]
        tails = [f"n{t}" for t in tails]

        if HG.directed:
            for t in tails:
                G.add_edge(t, e_node)
            for h in heads:
                G.add_edge(e_node, h)
        else:
            for v in heads + tails:
                G.add_edge(e_node, v)

    return G


def detect_dilation_and_drivers(HG_local, num_nodes):
    """通过二分图最大匹配检测 dilation 并识别驱动节点。

    Parameters
    ----------
    HG_local : HAT.Hypergraph
        有向超图。
    num_nodes : int
        系统节点总数。

    Returns
    -------
    tuple
        (dilation_exists, matched_heads, driver_nodes, matching)，分别代表是否存在超边膨胀，匹配的头节点集合，驱动节点集合，以及最大匹配字典。
    """
    G_star = star_graph(HG_local)
    G_bip = nx.Graph()
    E_side = set()
    from networkx.algorithms import bipartite
    for u, v in G_star.edges():
        if isinstance(u, str) and u.startswith("e") and isinstance(v, str) and v.startswith("n"):
            G_bip.add_edge(u, v)
            E_side.add(u)
    matching = bipartite.hopcroft_karp_matching(G_bip, E_side)
    matched_heads = {int(v[1:]) for v in matching if isinstance(
        v, str) and v.startswith("n")}
    all_nodes = set(range(HG_local.nnodes))
    driver_nodes = all_nodes - matched_heads
    dilation_exists = len(matched_heads) < num_nodes
    return dilation_exists, matched_heads, driver_nodes, matching


def detect_dilation_with_given_drivers(HG_local, drivers):
    """在给定驱动节点集下检测 dilation 是否仍然存在。

    Parameters
    ----------
    HG_local : HAT.Hypergraph
        有向超图。
    drivers : set of int
        候选驱动节点集合。

    Returns
    -------
    bool
        是否存在 dilation。
    """
    G_star = star_graph(HG_local)
    from networkx.algorithms import bipartite
    for d in drivers:
        u_name = f"u{d}"
        v_name = f"n{d}"
        if not G_star.has_node(u_name):
            G_star.add_node(u_name)
        G_star.add_edge(u_name, v_name)

    G_bip = nx.Graph()
    E_side = set()
    V_side = set()
    for u, v in G_star.edges():
        if (isinstance(u, str) and (u.startswith("e") or u.startswith("u")) and isinstance(v, str) and v.startswith("n")):
            G_bip.add_edge(u, v)
            E_side.add(u)
            V_side.add(v)
    matching = bipartite.hopcroft_karp_matching(G_bip,E_side)
    matched_heads = {int(v[1:]) for v in matching if isinstance(
        v, str) and v.startswith("n")}
    dilation_exists = len(matched_heads) < HG_local.nnodes
    return dilation_exists


def hypergraph_reachable(HG_local, start_nodes):
    """计算从起始节点集合出发，通过超边传播可达的节点集。

    Parameters
    ----------
    HG_local : HAT.Hypergraph
        有向超图。
    start_nodes : set of int
        起始节点集合。

    Returns
    -------
    set of int
        可达节点集合。
    """
    head_list = HG_local.edges["head"].tolist()
    tail_list = HG_local.edges["tail"].tolist()
    accessible = set(start_nodes)
    changed = True
    while changed:
        changed = False
        for h, t in zip(head_list, tail_list):
            if set(t).issubset(accessible):
                before = len(accessible)
                accessible.update(h)
                if len(accessible) > before:
                    changed = True
    return accessible


def _fast_propagate(accessible, newly_activated, already_active, edge_missing, edge_heads, node_edges_list):
    """快速传播：从新激活的边出发，递归激活更多边和节点。

    Parameters
    ----------
    accessible : set
        当前已可达节点集。
    newly_activated : list
        新激活的边索引列表。
    already_active : set
        已激活的边索引集合。
    edge_missing : list of set
        每条边还缺少的 tail 节点集。
    edge_heads : list of set
        每条边的 head 节点集。
    node_edges_list : list of list
        每个节点参与的边索引列表。

    Returns
    -------
    set
        传播后的可达节点集。
    """
    result = set(accessible)
    to_process = list(newly_activated)
    processed = set(already_active)
    i = 0
    while i < len(to_process):
        edge_idx = to_process[i]
        i += 1
        processed.add(edge_idx)
        new_nodes = edge_heads[edge_idx] - result
        result.update(new_nodes)
        for node in new_nodes:
            for check_edge_idx in node_edges_list[node]:
                if check_edge_idx not in processed:
                    if edge_missing[check_edge_idx].issubset(result):
                        to_process.append(check_edge_idx)
                        processed.add(check_edge_idx)
    return result


def greedy_accessibility(HG_local, initial_drivers=None):
    """贪心算法：在初始驱动节点基础上，选择最少的额外驱动节点使全系统可达。

    Parameters
    ----------
    HG_local : HAT.Hypergraph
        有向超图。
    initial_drivers : set of int, optional
        初始驱动节点集合。

    Returns
    -------
    set of int
        最终的驱动节点集合。
    """
    if initial_drivers is None:
        initial_drivers = set()
    all_sys_nodes = set(range(HG_local.nnodes))
    head_list = HG_local.edges["head"].tolist()
    tail_list = HG_local.edges["tail"].tolist()
    edge_tails = [set(t) for t in tail_list]
    edge_heads = [set(h) for h in head_list]
    node_edges_list = HG_local.nodes['edges'].tolist()
    edge_missing = [set(tail) for tail in edge_tails]
    current_drivers = set(initial_drivers)
    accessible = set(current_drivers)
    active_edges = set()
    for node in list(accessible):
        for edge_idx in node_edges_list[node]:
            edge_missing[edge_idx].discard(node)
            if not edge_missing[edge_idx] and edge_idx not in active_edges:
                active_edges.add(edge_idx)
                accessible.update(edge_heads[edge_idx])
    reachable_sys = accessible & all_sys_nodes
    remaining = all_sys_nodes - reachable_sys
    while remaining:
        best_node = None
        best_size = len(remaining)
        best_new_accessible = None
        candidate_pool = set()
        for v in remaining:
            for edge_idx in node_edges_list[v]:
                if edge_idx not in active_edges and edge_missing[edge_idx] == {v}:
                    candidate_pool.add(v)
                    break
        if not candidate_pool:
            candidate_pool = remaining
        for cand in candidate_pool:
            new_accessible = accessible | {cand}
            newly_activated = []
            for edge_idx in node_edges_list[cand]:
                if edge_idx not in active_edges:
                    if edge_missing[edge_idx] <= new_accessible:
                        newly_activated.append(edge_idx)
            if newly_activated:
                new_accessible = _fast_propagate(
                    new_accessible, newly_activated, active_edges, edge_missing, edge_heads, node_edges_list)
            new_remaining_size = len(
                all_sys_nodes - (new_accessible & all_sys_nodes))
            if new_remaining_size < best_size:
                best_node = cand
                best_size = new_remaining_size
                best_new_accessible = new_accessible
        current_drivers.add(best_node)
        accessible = best_new_accessible
        for edge_idx in node_edges_list[best_node]:
            edge_missing[edge_idx].discard(best_node)
            if not edge_missing[edge_idx] and edge_idx not in active_edges:
                active_edges.add(edge_idx)
        remaining = all_sys_nodes - (accessible & all_sys_nodes)
    return current_drivers


def compute_and_export_drivers(edge_list=None,
                               n=10, k=4, m=12, out_csv="drivers.csv"):
    """计算驱动节点集合并导出 CSV 文件。

    根据节点数自动选择算法之一：暴力，matching+greedy，matching，将驱动节点集合写入out_csv，每一行一个驱动节点，附带算法和dilation标记

    Parameters
    ----------
    edge_list : _type_, optional
        给定 edge_list 或者 head-tail 列表，默认 None 时使用 n,k,m 生成随机超图。
    n : int, optional
        节点总数，默认 10。
    k : int, optional
        每条边 tail 的大小，默认 4。
    m : int, optional
        默认的超边数量，默认 12。
    out_csv : str, optional
        输出 CSV 文件路径，默认 "drivers.csv"。

    Returns
    -------
    dict
        包含算法名称、dilation 标记和驱动节点列表的字典。
    """
    # ------------------ 构建或接收超图 ------------------
    if edge_list is None:
        edge_list = random_uniform_hypergraph(n, k, m)
    incidence = edge_list_to_incidence(edge_list)
    nodes = pd.DataFrame({"Nodes": np.arange(
        len(set().union(*[set(e['head']) | set(e['tail']) for e in edge_list])))})
    HG = HAT.Hypergraph(incidence_matrix=incidence, directed=True, nodes=nodes)
    HG = add_node_fields(HG)
    return _compute_drivers_on_hat(HG, out_csv)


def calc_drivers(hgs: dict, to_df=True) -> pd.DataFrame:
    """批量计算多个超图的驱动节点集合

    Parameters
    ----------
    hgs : dict
        包含多个 HAT.Hypergraph 对象的字典，键为超图名称，值为 HAT.Hypergraph 对象。

    to_df : bool, optional
        是否将结果转换为 DataFrame，默认为 True。

    Returns
    -------
    pd.DataFrame
        包含每个超图的驱动节点集合、算法和 dilation 标记的 DataFrame。
    """
    result = {}
    for year, hg in hgs.items():
        res = compute_drivers_from_hypernetx(hg, out_csv=None)
        node_lst = list(hg.nodes)
        res_country = [node_lst[driver] for driver in res["drivers"]]
        result[year] = res_country

    if to_df:
        return pd.DataFrame({
            "year": list(result.keys()),
            "drivers": [','.join(drivers) for drivers in result.values()],
            "partition": [len(drivers)/hg.shape[0] for drivers in result.values()]
        })
    return result
