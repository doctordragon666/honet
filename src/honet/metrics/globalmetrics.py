"""全局级指标：整张超图压成一个标量（或一张汇总表）。

本模块把原先「一个函数一个文件」的全局指标合并到一处：

* :func:`calc_entropy` —— 超度分布的香农熵；
* :func:`calc_information_gain` —— 高阶网络相对成对（投影）网络的信息增益；
* :func:`calc_clustering_coefficient_g` —— 全部结点局部聚类系数的均值；
* :func:`calc_natural_connectivity` —— 邻接矩阵特征值的指数平均取对数；
* :func:`assortativity_test` —— 按超度分组的余平均度（度相关性诊断）；
* :func:`calc_assortativity` —— 高阶同配系数及其零模型显著性；
* :func:`sub_reducibility` / :func:`calc_reducibility` —— （加权）可约性；
* :func:`hnxs_global` —— 多个超图的全局指标汇总成 ``pandas.DataFrame``。

依赖分布：可约性两个函数与同配系数的零模型需要 ``hypergraphx``；其余函数只需
``honet[analysis]`` （numpy / scipy / hypernetx / networkx / pandas）。
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from honet import _cache
from honet._deps import require
from honet._topology import Topology, topology
from honet.metrics._assortativity import assortativity_rank_correlation
from honet.metrics._reuse import package_function
from honet.metrics.nodemetrics import clustering_coefficient, excess_average_degree

#: 可约性允许的优化方式（与 HyperGraphX 官方一致）
_OPTIMIZATIONS = frozenset({"exact", "greedy"})
#: 可约性允许的熵估计口径（与 HyperGraphX 官方一致）
_ENTROPY_METHODS = frozenset({"count", "project"})


def _degree_entropy(degrees: Iterable[float]) -> float:
    """对一组度值求「度分布」的香农熵。

    这是 :func:`calc_entropy` 与 :func:`calc_information_gain` 共用的公式：

    .. math::

        S = -\\sum_i p_i \\ln p_i, \\qquad p_i = \\frac{k_i}{\\sum_j k_j}

    参数:
        degrees: 可迭代的度值序列（超度或成对度）。

    返回:
        熵值 ``float`` （自然对数单位）；度之和为 0 或只有一个结点时返回
        ``0.0``（分布退化，且避免出现 ``-0.0``）。
    """
    np = require("numpy")

    values = np.asarray(list(degrees), dtype=float)
    total = values.sum()
    if total <= 0.0:
        return 0.0

    p = values / total
    if p.size <= 1:
        return 0.0
    # 度为 0 的结点 p=0，0·ln0 约定为 0，用掩码排除避免 log(0) 警告
    positive = p[p > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def calc_entropy(H: Any) -> float:
    r"""计算超图的度分布香农熵。

    以结点超度 :math:`k_i` 为分布，取

    .. math::

        S = -\sum_i p_i \ln p_i, \qquad p_i = \frac{k_i}{\sum_j k_j}

    熵衡量「超度分布有多不均匀」：全部结点超度相同（规则超图）时熵最大，少数
    结点垄断全部关联时熵接近 0。

    参数:
        H: hypernetx 超图对象。

    返回:
        熵值 ``float``；没有任何关联（度之和为 0）时返回 ``0.0``。

    说明:
        旧实现调用 ``hypernetx.dist_stats(H)['node degree list']`` 取度序列，
        但 ``dist_stats`` 不属于 hypernetx 的稳定接口（2.x 各小版本字段名
        有变动），因此这里改用拓扑索引里的度值。两者在 2.4.3 上给出同一组
        度值，熵是顺序无关量，结果完全一致。
        结果按超图结构指纹缓存；结构未变时重复调用直接返回上次结果。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> round(calc_entropy(H), 4)
        1.0397
    """
    return _cache.cached(
        H, "calc_entropy:v2", lambda: _degree_entropy(topology(H).degrees.values())
    )


def _projected_degrees(top: Topology) -> list[float]:
    """返回团投影（成对网络）中每个结点的度。

    团投影把每条大小为 ``k`` 的超边展开成 ``k`` 团，因此结点 ``i`` 的成对度
    就是与它共边的结点数 ``|N(i)|``，可直接由拓扑索引的邻居表得到。
    """
    return [float(len(top.neighbors_of(node))) for node in top.nodes]


def _information_gain(H: Any) -> dict[str, float]:
    """计算成对（投影）熵、高阶熵与两者之差。"""
    top = topology(H)
    pairwise = _degree_entropy(_projected_degrees(top))
    higher_order = _degree_entropy(top.degrees.values())
    return {
        "pairwise_entropy": pairwise,
        "hypergraph_entropy": higher_order,
        "information_gain": pairwise - higher_order,
    }


def calc_information_gain(H: Any, detail: bool = False) -> Any:
    r"""计算高阶网络相对成对（投影）网络的信息增益。

    公式参考「基于引文大数据的高阶网络建模及信息增益比较研究」
    （李佳旭等，系统科学与数学 41(10), 2021）的信息增益定义：把低阶网络模型
    的熵减去高阶网络模型的熵，差值即为高阶表示带来的信息增益。本实现把
    「低阶模型」取为该超图的**团投影成对网络**，「高阶模型」取为超图本身，
    两者用同一套度分布熵公式（:func:`calc_entropy`）计算：

    .. math::

        G = S_{\mathrm{pair}} - S_{\mathrm{high}}
          = -\sum_i p_i^{\mathrm{pair}} \ln p_i^{\mathrm{pair}}
            + \sum_i p_i^{\mathrm{high}} \ln p_i^{\mathrm{high}}

    其中高阶侧 ``p_i = k_i / Σ_j k_j`` 用结点超度，成对侧用团投影里结点
    的成对度 ``|N(i)|``。两个熵都是归一化分布上的量，结点集合相同、对数底
    相同，因此可以直接相减。

    :math:`G > 0` 表示成对投影把信息摊得更均匀——高阶表示比投影更集中，
    一阶模型存在高阶信息损失；:math:`G = 0` 表示投影与高阶表示携带的多样性
    相同（例如全部超边都是 2 元边时，投影就是原网络本身）。

    参数:
        H: hypernetx 超图对象。
        detail: 为 ``False`` 时只返回信息增益标量；为 ``True`` 时返回包含
            ``pairwise_entropy`` / ``hypergraph_entropy`` / ``information_gain``
            三个键的字典，便于核对两侧熵值。

    返回:
        ``detail=False`` 返回 ``float``；否则返回 ``dict[str, float]``。
        空超图两侧熵均为 0，信息增益为 ``0.0``。

    说明:
        结果按超图结构指纹缓存；``detail`` 两种口径分开缓存，不共用条目。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["b", "c"]})
        >>> round(calc_information_gain(H), 4)
        0.0437
        >>> calc_information_gain(hnx.Hypergraph({"e1": ["a", "b"]}))
        0.0
    """
    values = _cache.cached(
        H, f"calc_information_gain:v1:{int(detail)}", lambda: _information_gain(H)
    )
    return values if detail else values["information_gain"]


def calc_clustering_coefficient_g(H: Any) -> float:
    """计算超图的全局聚类系数。

    全局值是全部结点局部聚类系数的算术平均，因此口径完全由
    :func:`honet.metrics.clustering_coefficient` 决定：邻居对是否至少共享一条
    超边。

    参数:
        H: hypernetx 超图对象。

    返回:
        全部结点局部聚类系数的算术平均；超图没有结点时返回 ``0.0``。

    说明:
        局部值直接复用 :func:`honet.metrics.clustering_coefficient`（纯 Python
        实现），因此本函数只额外依赖 numpy 求均值。旧实现调用 XGI 的
        ``clustering_coefficient``，其口径是「邻居对共享超边的平均比例」
        （按共享超边数加权），与这里的「邻居对是否连通」口径在超边重叠时
        数值不同；新口径与 honet 既有局部实现保持一致。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"]})
        >>> calc_clustering_coefficient_g(H)
        1.0
    """
    np = require("numpy")

    local = clustering_coefficient(H)
    if not local:
        return 0.0
    return float(np.mean(list(local.values())))


def _natural_connectivity(H: Any) -> float:
    """计算自然连通度（邻接矩阵特征值的指数平均取对数）。"""
    np = require("numpy")

    if not H.nodes:
        return 0.0

    adjacency = H.adjacency_matrix()
    # hypernetx 2.x 返回稀疏矩阵；1.x 曾返回 (矩阵, 结点索引, 超边索引) 元组
    if isinstance(adjacency, tuple):
        adjacency = adjacency[0]
    dense = adjacency.toarray() if hasattr(adjacency, "toarray") else np.asarray(adjacency)

    eigenvalues = np.linalg.eigvalsh(dense)
    return float(np.log(np.mean(np.exp(eigenvalues))))


def calc_natural_connectivity(H: Any) -> float:
    """计算超图的自然连通度。

    用 hypernetx 的团展开邻接矩阵（两结点共享超边则连一条边），取全部特征值
    :math:`\\lambda_i` 的指数平均的对数：

    .. math::

        \\lambda = \\ln\\left(\\frac{1}{N}\\sum_{i=1}^{N} e^{\\lambda_i}\\right)

    值越大表示高阶连通结构越冗余：去掉少量结点后仍能维持连通。

    参数:
        H: hypernetx 超图对象。

    返回:
        自然连通度 ``float``；超图没有结点时返回 ``0.0``。

    说明:
        旧实现写 ``H.adjacency_matrix().todense()``。hypernetx 2.4.3 的
        ``adjacency_matrix()`` 直接返回稀疏矩阵（不是 tuple），因此这里用
        ``toarray()`` 取稠密数组，同时兼容返回 tuple 的旧版本（取第一项）。
        结果按超图结构指纹缓存；结构未变时重复调用直接返回上次结果。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"]})
        >>> round(calc_natural_connectivity(H), 4)
        0.4338
    """
    return _cache.cached(H, "calc_natural_connectivity:v2", lambda: _natural_connectivity(H))


def assortativity_test(H: Any) -> dict[int, float]:
    """返回每个超度取值下结点的平均余平均度。

    做法：把结点按超度 ``k`` 分组，看每组内的余平均度（邻居的平均超度）是否随
    ``k`` 变化。若余平均度对 ``k`` 是常数，说明超度与邻居超度无关（无度相关性）；
    若单调变化，则存在同配（随 k 上升）或异配（随 k 下降）。

    参数:
        H: hypernetx 超图对象。

    返回:
        ``{超度: 该度全部结点的余平均度均值}``，键按超度升序排列。

    说明:
        余平均度直接复用 :func:`honet.metrics.excess_average_degree`，因此与
        全局余平均度口径完全一致；度值取自共享拓扑索引
        （:mod:`honet._topology`），不再逐点求度。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> sorted(assortativity_test(H))
        [1, 2]
    """
    np = require("numpy")

    # 度取自共享拓扑索引；余平均度复用局部指标（其结果同样带缓存）
    degrees = topology(H).degrees
    excess = excess_average_degree(H)

    # 1) 按超度分桶收集余平均度
    buckets: dict[int, list[float]] = defaultdict(list)
    for node, degree in degrees.items():
        buckets[int(degree)].append(excess.get(node, 0.0))

    # 2) 每个超度取平均；键升序，便于直接观察随 k 的变化趋势
    return {degree: float(np.mean(values)) for degree, values in sorted(buckets.items())}


def calc_assortativity(
    H: Any,
    n_rand: int = 1000,
    n_steps: int = 1000,
    label: str = "edge",
    seed: int = 42,
) -> tuple[float, float]:
    """计算高阶同配系数与相对配置模型零模型的双侧经验 p 值。

    观测值由 :mod:`honet.metrics._assortativity` 计算（超度最大与第二大结点
    的平均秩相关）；零模型用 HyperGraphX 的配置模型（``configuration_model``，
    保持度序列与超边规模分布的 MCMC 重连）抽样。

    参数:
        H: hypernetx 超图对象（也可以是 ``hypergraphx.Hypergraph``）。
        n_rand: 零模型抽样次数。
        n_steps: 每个零模型的配置模型 MCMC 步数。
        label: 配置模型标签空间，``"edge"``（较快）或 ``"vertex"``。
        seed: 随机种子，用于结果复现。

    返回:
        ``(r, p)``：``r`` 为观测同配系数；``p`` 为双侧经验 p 值，用
        ``(计数+1)/(n+1)`` 平滑后取 ``min(1, 2·min(p_ge, p_le))``。

    说明:
        **与旧实现的差异**：旧代码返回 ``(r_obs, p)`` 但 docstring 写的是
        ``(r, z)``，并且算了 z 又丢掉；这里按实际返回值的语义写文档，不再
        计算 z（需要 z 时可用 ``n_rand`` 次抽样自行复现）。
        旧实现用 ``hypergraphx.get_nodes()/get_edges()`` 在 HGX 对象上算同配
        系数；现在核心计算改为接收 hypernetx 超图（``honet._hnx`` 的
        ``node_degrees`` / ``edge_members``），零模型样本转回 hypernetx 后再
        计算，定义不变（超度最大与第二大结点的平均秩相关）。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b", "c"], "e2": ["b", "c"]})
        >>> r, p = calc_assortativity(H, n_rand=10, n_steps=10)
        >>> 0.0 <= p <= 1.0
        True
    """
    np = require("numpy")
    hgx = require("hypergraphx")
    # 直接从实现子模块取函数：``hypergraphx.generation`` 是惰性转发模块，其
    # 同名属性在子模块被 import 后会被模块对象覆盖，之后再取就不可调用了
    configuration_model = require(
        "hypergraphx.generation.configuration_model", attr="configuration_model"
    )

    # 1) 观测值：统一在 hypernetx 上计算
    to_hgx = package_function("honet.convert", "convert_hypernetx_to_hypergraphx")
    to_hnx = package_function("honet.convert", "convert_hypergraphx_to_hypernetx")
    if isinstance(H, hgx.Hypergraph):
        observed_graph = H
        hnx_graph = to_hnx(H)
    else:
        hnx_graph = H
        observed_graph = to_hgx(H)
    r_obs = assortativity_rank_correlation(hnx_graph)

    # 2) 零模型：保持度序列与超边规模分布，逐次 MCMC 重连
    rng = np.random.default_rng(seed)
    nulls = np.asarray(
        [
            assortativity_rank_correlation(
                to_hnx(
                    configuration_model(
                        observed_graph,
                        n_steps=n_steps,
                        label=label,
                        seed=int(rng.integers(0, 2**32 - 1)),
                    )
                )
            )
            for _ in range(n_rand)
        ],
        dtype=float,
    )

    # 3) 双侧经验 p 值；+1/(n+1) 平滑，规避 p=0 与有限样本偏小
    n = nulls.size
    p_ge = (float(np.sum(nulls >= r_obs)) + 1.0) / (n + 1.0)
    p_le = (float(np.sum(nulls <= r_obs)) + 1.0) / (n + 1.0)
    p = float(min(1.0, 2.0 * min(p_ge, p_le)))

    return float(r_obs), p


def _to_hgx(H: Any) -> Any:
    """把 hypernetx 超图转成 HyperGraphX 超图，转换结果按结构指纹缓存。

    可约性与 ``sub_reducibility`` 都需要 HyperGraphX 对象，重复转换是纯浪费；
    转换只依赖拓扑，因此可以安全地按指纹复用。
    """
    return _cache.cached(
        H,
        "to_hypergraphx:v1",
        lambda: package_function("honet.convert", "convert_hypernetx_to_hypergraphx")(H),
    )


def _layer_weight_costs(hg: Any, layers: dict[int, set[Any]]) -> dict[int, float]:
    """计算每层的权重异质性成本 ``W_l``。

    参数:
        hg: ``hypergraphx.Hypergraph``，需支持 ``get_weight(edge)``。
        layers: ``{超边规模: 该规模的边集合}``，由 HyperGraphX 的私有函数
            ``_get_layers`` 给出。

    返回:
        ``{超边规模: W_l}``，其中 ``W_l = E_l · [ln E_l − H(p)] ≥ 0``，
        层内权重全相等时取 0；自然对数单位与官方熵估计保持一致。
    """
    costs: dict[int, float] = {}
    for size, edges in layers.items():
        keys = list(edges)
        count = len(keys)
        if count <= 1:
            # 0/1 条边无从谈异质性
            costs[size] = 0.0
            continue

        weights = [float(hg.get_weight(edge)) for edge in keys]
        total = sum(weights)
        if total <= 0.0:
            costs[size] = 0.0
            continue

        p = [weight / total for weight in weights]
        entropy = -sum(pi * math.log(pi) for pi in p if pi > 0.0)
        costs[size] = count * (math.log(count) - entropy)
    return costs


def sub_reducibility(
    hg: Any,
    partition: Any = None,
    optimization: str = "exact",
    entropy_method: str = "count",
) -> tuple[float, tuple[int, ...]]:
    """计算（加权）超图的可约性及代表层集合。

    可约性回答「一个多层超图能否只用少数几个代表性层（超边规模）无损描述」：
    把超边按规模分层，若层间信息高度冗余，则可约性接近 1。

    本实现是对 HyperGraphX 官方 ``reducibility`` 的**加权修正版**：在官方无权
    熵的基础上，额外叠加每层的「权重异质性成本」——层内权重越不均，该层越
    难以被其它层代表。

    参考文献：Kirkley, A., Felippe, H., & Battiston, F. PRL 135, 247401 (2025).

    参数:
        hg: ``hypergraphx.Hypergraph``。加权时权重自动参与；无权时权重全为 1，
            退化为官方结果。
        partition: 结点分区（``list`` / ``dict`` / ``None``），用于多尺度可约性。
        optimization: ``"exact"`` （枚举全部代表层组合）或 ``"greedy"`` （贪心）。
        entropy_method: ``"count"`` 或 ``"project"``，对应官方两种熵估计。

    返回:
        ``(reducibility, representative_layers)``：可约性取值在 ``[0, 1]``，
        代表层是升序排列的超边规模元组。

    抛出:
        ImportError: 未安装 hypergraphx。
        hypergraphx.exceptions.InvalidParameterError: 参数非法，或输入不是
            无向 ``Hypergraph`` （由 HyperGraphX 抛出）。

    说明:
        **依赖 hypergraphx 的内部接口**：``_get_entropies_count`` /
        ``_get_entropies_project`` / ``_get_layers`` / ``_powerset`` 都是下划线
        开头的私有函数，上游版本变化时可能改名或改语义而失效。这是为了在
        官方无权熵之上叠加权重成本所付出的代价；升级 hypergraphx 后请重跑
        本子包测试。

    示例:
        >>> import hypergraphx as hgx
        >>> G = hgx.Hypergraph([("a", "b"), ("a", "b", "c")], weighted=True, weights=[1.0, 1.0])
        >>> value, reps = sub_reducibility(G, optimization="greedy")
        >>> 0.0 <= value <= 1.0
        True
    """
    np = require("numpy")
    require("hypergraphx")
    reducibility_module = require("hypergraphx.measures.reducibility")
    invalid_parameter = require("hypergraphx.exceptions", attr="InvalidParameterError")

    if entropy_method not in _ENTROPY_METHODS:
        raise invalid_parameter('entropy_method must be either "count" or "project".')
    if optimization not in _OPTIMIZATIONS:
        raise invalid_parameter('optimization must be either "exact" or "greedy".')

    # 1) 复用官方无权熵：Q 为层自身成本，M[k][l] 为高层 k 对低层 l 的条件成本
    if entropy_method == "project":
        M, Q = reducibility_module._get_entropies_project(hg, partition)
    else:
        M, Q = reducibility_module._get_entropies_count(hg, partition)

    # 2) 加权修正：把各层权重异质性成本 W_l 叠加到 Q 与 M 上
    weight_costs = _layer_weight_costs(hg, reducibility_module._get_layers(hg))
    for layer in Q:
        cost = weight_costs.get(layer, 0.0)
        Q[layer] = Q[layer] + cost
        for higher in M:
            # M[k][l] 只对 k > l 存在
            if higher > layer:
                M[higher][layer] += cost

    max_layer = max(Q.keys())

    # 3) 最小化总信息成本，寻找代表层集合（与官方逻辑一致）
    if optimization == "exact":
        costs_by_reps: dict[tuple[int, ...], float] = {}
        for reps in reducibility_module._powerset(list(Q.keys())):
            chosen = set(reps)
            if max_layer not in chosen:
                # 最高层必须作为代表层（它没有更高层可以代表它）
                continue
            total = sum(Q[r] for r in chosen)
            for layer in set(Q.keys()) - chosen:
                higher = [r for r in chosen if r > layer]
                best = higher[int(np.argmin([M[r][layer] for r in higher]))]
                total += M[best][layer]
            costs_by_reps[tuple(sorted(chosen))] = total
        best_reps = min(costs_by_reps, key=costs_by_reps.get)
        best_cost = costs_by_reps[best_reps]
    else:
        rep_history: list[list[int]] = []
        cost_history: list[float] = []
        chosen = {max_layer}
        remaining = set(Q.keys()) - chosen
        rep_history.append(sorted(chosen))
        cost_history.append(Q[max_layer] + sum(M[max_layer][layer] for layer in remaining))
        while remaining:
            candidates: list[int] = []
            scores: list[float] = []
            for candidate in remaining:
                trial = chosen | {candidate}
                still_out = remaining - {candidate}
                total = sum(Q[r] for r in trial)
                for layer in still_out:
                    higher = [r for r in trial if r > layer]
                    best = higher[int(np.argmin([M[r][layer] for r in higher]))]
                    total += M[best][layer]
                candidates.append(candidate)
                scores.append(total)
            winner = int(np.argmin(scores))
            chosen.add(candidates[winner])
            remaining.remove(candidates[winner])
            rep_history.append(sorted(chosen))
            cost_history.append(scores[winner])
        best_index = int(np.argmin(cost_history))
        best_reps = tuple(rep_history[best_index])
        best_cost = cost_history[best_index]

    # 4) 归一化到 [0, 1]：完全不可约（每层都必需）时为 0，单层即可代表时为 1
    baseline = sum(Q[layer] for layer in Q)
    if baseline == Q[max_layer]:
        return 1.0, tuple(best_reps)
    return float((baseline - best_cost) / (baseline - Q[max_layer])), tuple(best_reps)


def calc_reducibility(H: Any, no_layer: bool = True, weighted: bool = False) -> Any:
    """计算超图的可约性。

    可约性的含义是「能否只用少数代表性层（超边规模）描述整个多层超图而不损失
    信息」。``weighted=False`` 时直接调用 HyperGraphX 的官方实现；
    ``weighted=True`` 时改用 :func:`sub_reducibility` （在官方熵上叠加层内权重
    异质性成本）。

    参数:
        H: ``hypernetx.Hypergraph`` （会被转换成 HyperGraphX 超图）或
            ``hypergraphx.Hypergraph``。
        no_layer: 为 ``True`` 时只返回可约性数值；为 ``False`` 时返回
            ``(可约性, 代表层集合)`` 二元组。
        weighted: 为 ``True`` 时用加权修正版（:func:`sub_reducibility`），
            为 ``False`` 时用 HyperGraphX 官方无权实现。

    返回:
        ``no_layer=True`` 返回 ``float``；否则返回 ``(float, tuple[int, ...])``。

    抛出:
        ImportError: 未安装 hypergraphx。
        TypeError: 传入既不是 hypernetx 也不是 hypergraphx 的超图对象。

    说明:
        **与旧实现的差异**：旧代码用 ``isinstance(H_hnx, hnx.Hypergraph)`` 判断
        是否转换，其它类型会被原样传给 HyperGraphX 并在更深处抛出难懂的异常；
        这里对非 HGX 输入统一先做 hypernetx 转换，类型不符时在入口处抛
        ``TypeError``。转换沿用 ``honet.convert`` 的口径（丢掉成员数 < 2 的
        退化超边）。

    示例:
        >>> import hypernetx as hnx
        >>> H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["a", "b", "c"]})
        >>> value = calc_reducibility(H)
        >>> 0.0 <= value <= 1.0
        True
    """
    hgx = require("hypergraphx")
    hnx = require("hypernetx")
    reducibility = require("hypergraphx.measures.reducibility", attr="reducibility")

    def _compute() -> Any:
        # 1) 统一成 HyperGraphX 超图（HGX 的边是成员元组，不保留 hypernetx 边名）
        if isinstance(H, hgx.Hypergraph):
            graph = H
        elif isinstance(H, hnx.Hypergraph):
            graph = _to_hgx(H)
        else:
            msg = f"calc_reducibility 需要 hypernetx 或 hypergraphx 超图，收到 {type(H).__name__}"
            raise TypeError(msg)

        # 2) 按是否加权选择实现
        if weighted:
            value: Any = sub_reducibility(graph, optimization="greedy")
            if no_layer:
                value = value[0]
        else:
            value = reducibility(graph, optimization="greedy")
            if no_layer:
                value = value[0]
        return float(value) if no_layer else value

    # 缓存键区分 no_layer / weighted：两者返回不同口径，不能共用条目
    cache_key = f"calc_reducibility:v2:{int(no_layer)}:{int(weighted)}"
    return _cache.cached(H, cache_key, _compute)


def hnxs_global(hnxs: dict[Any, Any]) -> Any:
    """计算多个高阶网络的整体性质，输出成表格。

    把「一堆超图（通常按年份排列）」压成一张 ``pandas.DataFrame``：行是时间键，
    列是指标名，便于直接画趋势图（见 :func:`honet.plot.draw_hnxs_global`）。

    参数:
        hnxs: ``{时间键: 超图对象}``，超图可以是 hypernetx 或 hypergraphx 对象。

    返回:
        ``pandas.DataFrame``，索引为 ``hnxs`` 的键（如年份），列为指标名。
        当前只统计一项指标「高阶网络可约性」（加权口径），与旧实现一致——
        旧代码里其余指标都已被注释掉。

    说明:
        **与旧实现的差异**：旧代码用 ``isinstance(H_hnx, hnx.Hypergraph)`` 判断
        输入，非 hypernetx 对象会被原样转交下游；这里改由
        :func:`honet.metrics.calc_reducibility` 统一处理（既接受 hypernetx 也
        接受 hypergraphx 超图）。单元格取 ``no_layer=True`` 的标量口径，
        与旧实现一致。

    示例:
        >>> import hypernetx as hnx
        >>> H1 = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["a", "b", "c"]})
        >>> H2 = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
        >>> table = hnxs_global({2020: H1, 2021: H2})
        >>> list(table.columns)
        ['高阶网络可约性']
    """
    pd = require("pandas")

    rows = {
        key: {"高阶网络可约性": calc_reducibility(graph, weighted=True)}
        for key, graph in hnxs.items()
    }
    return pd.DataFrame(rows).T
