"""高阶同配系数及其相对配置模型零模型的显著性。

观测值由 :mod:`honet.metrics._assortativity` 计算；零模型用 HyperGraphX 的
配置模型（``configuration_model``，保持度序列与超边规模分布的 MCMC 重连）
抽样，得到 z 分数与双侧经验 p 值。
"""

from __future__ import annotations

from typing import Any

from honet._deps import require
from honet.metrics._assortativity import assortativity_rank_correlation
from honet.metrics._reuse import package_function


def calc_assortativity(
    H: Any,
    n_rand: int = 1000,
    n_steps: int = 1000,
    label: str = "edge",
    seed: int = 42,
) -> tuple[float, float]:
    """计算高阶同配系数与相对配置模型零模型的双侧经验 p 值。

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
