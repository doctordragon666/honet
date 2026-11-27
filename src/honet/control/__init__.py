"""honet.control —— 高阶网络的驱动节点控制。

本子包有两层能力：

**一、拓扑层（驱动节点选择）**——迁移自旧库的 ``honcontrol`` 模块，依赖
HypergraphAnalysisToolbox（导入名 ``HAT``，属于 ``honet[hat]`` 可选组）。
回答「控制哪些结点」：dilation 检测、可达性、最小驱动集。

**二、动力学层（控制律设计）**——移植自 Ma et al., Physica A 653 (2024)
130108 的超图网络可控性框架。回答「怎么控制」：把「结点自身动力学 + 高阶
耦合 + 外部输入」组装成线性系统 ``Ẋ = 𝒜X + ℬU``，给出可控性判据与最小能量
控制律::

    𝒜 = I_N ⊗ A + c/(k-1) · L ⊗ Λ,    ℬ = Δ ⊗ B

这一层只依赖 numpy / scipy，不需要 HAT。

全部符号都是惰性导出的：只有真正访问某个符号时才会 import 对应实现模块，
因此 ``import honet`` 不会因此拉入 HAT / networkx / pandas。

实现分两层的理由（Python 导入语义）：``_`` 前缀的私有模块放真正的算法，
公开模块只做带类型签名的转发。若让公开模块互相 import，
``import honet.control.star_graph`` 会把子模块对象写回包属性，遮蔽本文件的
惰性导出，使 ``honet.control.star_graph`` 从函数变成模块。

约定：拓扑层的驱动节点一律是 ``0..nnodes-1`` 的整数下标（HAT 内部把结点
编号成整数），原始结点名用 ``HG.nodes["Names"]`` 反查；动力学层的结点下标
则是 ``HypergraphControlSystem.L`` 的行序。
"""

from __future__ import annotations

from honet._lazy import lazy_getattr

_EXPORTS = {
    # ---- 拓扑层：驱动节点选择（依赖 HAT）----
    "add_node_fields": "honet.control.add_node_fields:add_node_fields",
    "brute_force_min_driver_sets": (
        "honet.control.brute_force_min_driver_sets:brute_force_min_driver_sets"
    ),
    "calc_drivers": "honet.control.calc_drivers:calc_drivers",
    "compute_and_export_drivers": (
        "honet.control.compute_and_export_drivers:compute_and_export_drivers"
    ),
    "compute_drivers_from_hypernetx": (
        "honet.control.compute_drivers_from_hypernetx:compute_drivers_from_hypernetx"
    ),
    "detect_dilation_and_drivers": (
        "honet.control.detect_dilation_and_drivers:detect_dilation_and_drivers"
    ),
    "detect_dilation_with_given_drivers": (
        "honet.control.detect_dilation_with_given_drivers:detect_dilation_with_given_drivers"
    ),
    "edge_list_to_incidence": "honet.control.edge_list_to_incidence:edge_list_to_incidence",
    "greedy_accessibility": "honet.control.greedy_accessibility:greedy_accessibility",
    "hypergraph_reachable": "honet.control.hypergraph_reachable:hypergraph_reachable",
    "random_uniform_hypergraph": (
        "honet.control.random_uniform_hypergraph:random_uniform_hypergraph"
    ),
    "star_graph": "honet.control.star_graph:star_graph",
    # ---- 动力学层：控制律设计（仅依赖 numpy / scipy）----
    "HypergraphControlSystem": "honet.control._dynamics:HypergraphControlSystem",
    "ControlTrajectory": "honet.control._control_law:ControlTrajectory",
    "hypergraph_laplacian": "honet.control._laplacian:hypergraph_laplacian",
    "kalman_rank": "honet.control._controllability:kalman_rank",
    "is_controllable": "honet.control._controllability:is_controllable",
    "pbh_rank": "honet.control._controllability:pbh_rank",
    "gramian": "honet.control._controllability:gramian",
    "gramian_rank": "honet.control._controllability:gramian_rank",
    "minimum_energy_control": "honet.control._control_law:minimum_energy_control",
    # ---- 稳态与能量分析（参考 nctpy）----
    "matrix_normalization": "honet.control._steady_state:matrix_normalization",
    "is_stable": "honet.control._steady_state:is_stable",
    "infinite_gramian": "honet.control._steady_state:infinite_gramian",
    "gramian_auto": "honet.control._steady_state:gramian_auto",
    # ---- 可控性度量（参考 nctpy / Gu et al. 2015）----
    "average_controllability": "honet.control._metrics:average_controllability",
    "modal_controllability": "honet.control._metrics:modal_controllability",
    "control_energy_ranking": "honet.control._metrics:control_energy_ranking",
    # ---- 带状态约束的最优控制（参考 nctpy）----
    "OptimalControlResult": "honet.control._optimal_control:OptimalControlResult",
    "optimal_control": "honet.control._optimal_control:optimal_control",
    # ---- 前向仿真与批量控制任务（参考 nctpy）----
    "SimulationResult": "honet.control._simulate_system:SimulationResult",
    "simulate_system": "honet.control._simulate_system:simulate_system",
    "simulate_impulse": "honet.control._simulate_system:simulate_impulse",
    "BatchEnergyResult": "honet.control._batch_control:BatchEnergyResult",
    "expand_states": "honet.control._batch_control:expand_states",
    "control_energy_matrix": "honet.control._batch_control:control_energy_matrix",
    # ---- 单纯形牵制控制（周进等 2024，仅依赖 numpy）----
    "SimplexRanking": "honet.control._simplicial_control:SimplexRanking",
    "generalized_laplacian": "honet.control._simplicial_control:generalized_laplacian",
    "fiedler_vector": "honet.control._simplicial_control:fiedler_vector",
    "algebraic_connectivity": "honet.control._simplicial_control:algebraic_connectivity",
    "simplex_delta": "honet.control._simplicial_control:simplex_delta",
    "rank_simplices": "honet.control._simplicial_control:rank_simplices",
    "pinning_matrix": "honet.control._simplicial_control:pinning_matrix",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """按需加载实现模块（PEP 562）。

    拓扑层的符号需要 HAT；动力学层的符号只需要 numpy / scipy。因此先按
    ``hat`` 组探测，失败时回退到 ``analysis`` 组（numpy / scipy 属于分析
    依赖），避免只装了 numpy 却因缺 HAT 而无法使用动力学层。
    """
    try:
        return lazy_getattr(globals(), __name__, _EXPORTS, name, "hat")
    except ImportError:
        return lazy_getattr(globals(), __name__, _EXPORTS, name, "analysis")
