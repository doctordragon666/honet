# 分析子包

本页由 `tools/_make_api.py` 读取各子包的 `__all__` 自动生成，**不要手改**。
新增或删除公开符号后重跑：

```bash
uv run --group docs python tools/_make_api.py
```

| 子包 | 能力 | 导出数 | 需要 |
| --- | --- | ---: | --- |
| `io` | 数据加载与合并 | 5 | `analysis` |
| `convert` | hypernetx / xgi / hypergraphx / NetworkX / HAT 互转 | 8 | `analysis` |
| `build` | 超图构建与随机生成（随机、无标度、聚类、小世界） | 6 | `analysis` |
| `metrics` | 结点 / 超边 / 全局指标 | 16 | `analysis` |
| `community` | 社区发现（Random / Basic / NMF / Recursive / Hierarchy / Kumar） | 15 | `analysis` |
| `contagion` | 传播动力学 SIS / SIR（离散、随机、Gillespie、相图） | 30 | `analysis` |
| `attack` | 攻击鲁棒性（随机 / 度 / 中心性 / 目标） | 6 | `analysis` |
| `control` | 驱动节点控制 + 控制律设计 | 54 | `hat / analysis` |
| `plot` | 绘图（SIS/SIR 曲线、社区、地图、攻击结果） | 13 | `viz` |
| `export` | 导出 LaTeX / CSV / Excel | 5 | `analysis` |

## `honet.io`

数据加载与合并。

honet.io —— 外部数据（超边表、权重表、地图）的读取与合并。

```{eval-rst}
.. autofunction:: honet.io.load_edge_data
```

```{eval-rst}
.. autofunction:: honet.io.load_hypergraph_data
```

```{eval-rst}
.. autofunction:: honet.io.load_map_data
```

```{eval-rst}
.. autofunction:: honet.io.load_weight_data
```

```{eval-rst}
.. autofunction:: honet.io.merge_ew
```

## `honet.convert`

hypernetx / xgi / hypergraphx / NetworkX / HAT 互转。

honet.convert —— 超图格式互转（hypernetx / XGI / HyperGraphX / NetworkX / HAT）。

```{eval-rst}
.. autofunction:: honet.convert.convert_hypergraphx_to_hypernetx
```

```{eval-rst}
.. autofunction:: honet.convert.convert_hypernetx_to_honet
```

```{eval-rst}
.. autofunction:: honet.convert.convert_hypernetx_to_hypergraphx
```

```{eval-rst}
.. autofunction:: honet.convert.convert_hypernetx_to_nxgraph
```

```{eval-rst}
.. autofunction:: honet.convert.convert_hypernetx_to_xgi
```

```{eval-rst}
.. autofunction:: honet.convert.convert_xgi_to_hypernetx
```

```{eval-rst}
.. autofunction:: honet.convert.get_hyperedge_weights
```

```{eval-rst}
.. autofunction:: honet.convert.hypernetx_to_hat_directed
```

## `honet.build`

超图构建与随机生成（随机、无标度、聚类、小世界）。

honet.build —— 超图的构建与随机生成。

```{eval-rst}
.. autofunction:: honet.build.build_hypernet
```

```{eval-rst}
.. autofunction:: honet.build.build_hypernets
```

```{eval-rst}
.. autofunction:: honet.build.clustered_hypergraph
```

```{eval-rst}
.. autofunction:: honet.build.generate_random_hypernet
```

```{eval-rst}
.. autofunction:: honet.build.scale_free_hypergraph
```

```{eval-rst}
.. autofunction:: honet.build.small_world_hypergraph
```

## `honet.metrics`

结点 / 超边 / 全局指标。

honet.metrics —— 超图指标（结点 / 超边 / 全局）。

```{eval-rst}
.. autofunction:: honet.metrics.assortativity_test
```

```{eval-rst}
.. autofunction:: honet.metrics.betweenness_centrality
```

```{eval-rst}
.. autofunction:: honet.metrics.calc_assortativity
```

```{eval-rst}
.. autofunction:: honet.metrics.calc_clustering_coefficient_g
```

```{eval-rst}
.. autofunction:: honet.metrics.calc_entropy
```

```{eval-rst}
.. autofunction:: honet.metrics.calc_information_gain
```

```{eval-rst}
.. autofunction:: honet.metrics.calc_natural_connectivity
```

```{eval-rst}
.. autofunction:: honet.metrics.calc_reducibility
```

```{eval-rst}
.. autofunction:: honet.metrics.clustering_coefficient
```

```{eval-rst}
.. autofunction:: honet.metrics.eigenvector_centrality
```

```{eval-rst}
.. autofunction:: honet.metrics.excess_average_degree
```

```{eval-rst}
.. autofunction:: honet.metrics.get_avg_degree_by_edge_size
```

```{eval-rst}
.. autofunction:: honet.metrics.hnxs_global
```

```{eval-rst}
.. autofunction:: honet.metrics.sub_reducibility
```

```{eval-rst}
.. autofunction:: honet.metrics.temporal_closeness_centrality
```

```{eval-rst}
.. autofunction:: honet.metrics.weighted_degree
```

## `honet.community`

社区发现（Random / Basic / NMF / Recursive / Hierarchy / Kumar）。

honet.community —— 高阶网络社区划分与划分结果比较。

```{eval-rst}
.. autofunction:: honet.community.Basic_Method
```

```{eval-rst}
.. autofunction:: honet.community.Hierarchy_Method
```

```{eval-rst}
.. autoclass:: honet.community.Method
```

```{eval-rst}
.. autofunction:: honet.community.NMF_Method
```

```{eval-rst}
.. autofunction:: honet.community.Random_Method
```

```{eval-rst}
.. autofunction:: honet.community.Recursive_Method
```

```{eval-rst}
.. autofunction:: honet.community.basic_method
```

```{eval-rst}
.. autofunction:: honet.community.check_partition
```

```{eval-rst}
.. autofunction:: honet.community.compare_graphs
```

```{eval-rst}
.. autofunction:: honet.community.compare_methods
```

```{eval-rst}
.. autofunction:: honet.community.extract_community_partition
```

```{eval-rst}
.. autofunction:: honet.community.hierarchy_method
```

```{eval-rst}
.. autofunction:: honet.community.nmf_method
```

```{eval-rst}
.. autofunction:: honet.community.random_method
```

```{eval-rst}
.. autofunction:: honet.community.recursive_method
```

## `honet.contagion`

传播动力学 SIS / SIR（离散、随机、Gillespie、相图）。

高阶网络上的传播动力学：SIS / SIR 模拟、感染率生成与相图。

```{eval-rst}
.. autoclass:: honet.contagion.CompartmentModel
```

```{eval-rst}
.. autoclass:: honet.contagion.ContagionMethod
```

```{eval-rst}
.. autoclass:: honet.contagion.Contagion_Method
```

```{eval-rst}
.. autofunction:: honet.contagion.RandomDiscrete_SIR
```

```{eval-rst}
.. autofunction:: honet.contagion.RandomDiscrete_SIS
```

```{eval-rst}
.. autofunction:: honet.contagion.RandomGillespie_SIR
```

```{eval-rst}
.. autofunction:: honet.contagion.RandomGillespie_SIS
```

```{eval-rst}
.. autofunction:: honet.contagion._parse_strategy
```

```{eval-rst}
.. autofunction:: honet.contagion.available_models
```

```{eval-rst}
.. autofunction:: honet.contagion.classify_point
```

```{eval-rst}
.. autofunction:: honet.contagion.discrete_SIR_control
```

```{eval-rst}
.. autofunction:: honet.contagion.discrete_SIS_control
```

```{eval-rst}
.. autofunction:: honet.contagion.discrete_SIS_control_pulse
```

```{eval-rst}
.. autofunction:: honet.contagion.discrete_sir_control
```

```{eval-rst}
.. autofunction:: honet.contagion.discrete_sis_control
```

```{eval-rst}
.. autofunction:: honet.contagion.discrete_sis_control_pulse
```

```{eval-rst}
.. autofunction:: honet.contagion.generate_tau
```

```{eval-rst}
.. autofunction:: honet.contagion.get_model
```

```{eval-rst}
.. autofunction:: honet.contagion.honSIR
```

```{eval-rst}
.. autofunction:: honet.contagion.honSIS
```

```{eval-rst}
.. autofunction:: honet.contagion.hon_sir
```

```{eval-rst}
.. autofunction:: honet.contagion.hon_sis
```

```{eval-rst}
.. autofunction:: honet.contagion.is_stable
```

```{eval-rst}
.. autofunction:: honet.contagion.random_discrete_sir
```

```{eval-rst}
.. autofunction:: honet.contagion.random_discrete_sis
```

```{eval-rst}
.. autofunction:: honet.contagion.random_gillespie_sir
```

```{eval-rst}
.. autofunction:: honet.contagion.random_gillespie_sis
```

```{eval-rst}
.. autofunction:: honet.contagion.register_model
```

```{eval-rst}
.. autofunction:: honet.contagion.simulate
```

```{eval-rst}
.. autofunction:: honet.contagion.sis_phase
```

## `honet.attack`

攻击鲁棒性（随机 / 度 / 中心性 / 目标）。

honet.attack —— 超图与普通图的攻击鲁棒性分析。

```{eval-rst}
.. autofunction:: honet.attack.basic_attack_sequence
```

```{eval-rst}
.. autofunction:: honet.attack.centrality_attack_sequence
```

```{eval-rst}
.. autofunction:: honet.attack.degree_attack_sequence
```

```{eval-rst}
.. autofunction:: honet.attack.execute_attack
```

```{eval-rst}
.. autofunction:: honet.attack.main_country
```

```{eval-rst}
.. autofunction:: honet.attack.random_attack_sequence
```

## `honet.control`

驱动节点控制 + 控制律设计。

honet.control —— 高阶网络的驱动节点控制。

```{eval-rst}
.. autoclass:: honet.control.BatchEnergyResult
```

```{eval-rst}
.. autoclass:: honet.control.ControlTrajectory
```

```{eval-rst}
.. autoclass:: honet.control.HypergraphControlSystem
```

```{eval-rst}
.. autoclass:: honet.control.LTVControllabilityResult
```

```{eval-rst}
.. autoclass:: honet.control.OptimalControlResult
```

```{eval-rst}
.. autoclass:: honet.control.SimplexRanking
```

```{eval-rst}
.. autoclass:: honet.control.SimulationResult
```

```{eval-rst}
.. autofunction:: honet.control.add_node_fields
```

```{eval-rst}
.. autofunction:: honet.control.algebraic_connectivity
```

```{eval-rst}
.. autofunction:: honet.control.average_controllability
```

```{eval-rst}
.. autofunction:: honet.control.brute_force_min_driver_sets
```

```{eval-rst}
.. autofunction:: honet.control.calc_drivers
```

```{eval-rst}
.. autofunction:: honet.control.compute_and_export_drivers
```

```{eval-rst}
.. autofunction:: honet.control.compute_drivers_from_hypernetx
```

```{eval-rst}
.. autofunction:: honet.control.control_energy_matrix
```

```{eval-rst}
.. autofunction:: honet.control.control_energy_ranking
```

```{eval-rst}
.. autofunction:: honet.control.detect_dilation_and_drivers
```

```{eval-rst}
.. autofunction:: honet.control.detect_dilation_with_given_drivers
```

```{eval-rst}
.. autofunction:: honet.control.edge_list_to_incidence
```

```{eval-rst}
.. autofunction:: honet.control.expand_states
```

```{eval-rst}
.. autofunction:: honet.control.fdr_p_values
```

```{eval-rst}
.. autofunction:: honet.control.fiedler_vector
```

```{eval-rst}
.. autofunction:: honet.control.generalized_laplacian
```

```{eval-rst}
.. autofunction:: honet.control.gramian
```

```{eval-rst}
.. autofunction:: honet.control.gramian_auto
```

```{eval-rst}
.. autofunction:: honet.control.gramian_rank
```

```{eval-rst}
.. autofunction:: honet.control.greedy_accessibility
```

```{eval-rst}
.. autofunction:: honet.control.hypergraph_laplacian
```

```{eval-rst}
.. autofunction:: honet.control.hypergraph_reachable
```

```{eval-rst}
.. autofunction:: honet.control.infinite_gramian
```

```{eval-rst}
.. autofunction:: honet.control.integrate_control_energy
```

```{eval-rst}
.. autofunction:: honet.control.is_controllable
```

```{eval-rst}
.. autofunction:: honet.control.is_stable
```

```{eval-rst}
.. autofunction:: honet.control.kalman_rank
```

```{eval-rst}
.. autofunction:: honet.control.label_to_state
```

```{eval-rst}
.. autofunction:: honet.control.ltv_controllability
```

```{eval-rst}
.. autofunction:: honet.control.ltv_gramian
```

```{eval-rst}
.. autofunction:: honet.control.matrix_normalization
```

```{eval-rst}
.. autofunction:: honet.control.minimum_energy_control
```

```{eval-rst}
.. autofunction:: honet.control.minimum_energy_exact
```

```{eval-rst}
.. autofunction:: honet.control.minimum_energy_fast
```

```{eval-rst}
.. autofunction:: honet.control.modal_controllability
```

```{eval-rst}
.. autofunction:: honet.control.normalize_state
```

```{eval-rst}
.. autofunction:: honet.control.normalize_weights
```

```{eval-rst}
.. autofunction:: honet.control.null_p_value
```

```{eval-rst}
.. autofunction:: honet.control.optimal_control
```

```{eval-rst}
.. autofunction:: honet.control.pbh_rank
```

```{eval-rst}
.. autofunction:: honet.control.pinning_matrix
```

```{eval-rst}
.. autofunction:: honet.control.random_uniform_hypergraph
```

```{eval-rst}
.. autofunction:: honet.control.rank_simplices
```

```{eval-rst}
.. autofunction:: honet.control.simplex_delta
```

```{eval-rst}
.. autofunction:: honet.control.simulate_impulse
```

```{eval-rst}
.. autofunction:: honet.control.simulate_system
```

```{eval-rst}
.. autofunction:: honet.control.star_graph
```

## `honet.plot`

绘图（SIS/SIR 曲线、社区、地图、攻击结果）。

honet.plot —— 高阶网络分析结果的绘图接口（基于 matplotlib）。

```{eval-rst}
.. autofunction:: honet.plot.draw_attack_results
```

```{eval-rst}
.. autofunction:: honet.plot.draw_communities_onmap
```

```{eval-rst}
.. autofunction:: honet.plot.draw_community_partition
```

```{eval-rst}
.. autofunction:: honet.plot.draw_control_partition
```

```{eval-rst}
.. autofunction:: honet.plot.draw_hnxs_global
```

```{eval-rst}
.. autofunction:: honet.plot.draw_sir
```

```{eval-rst}
.. autofunction:: honet.plot.draw_sis
```

```{eval-rst}
.. autofunction:: honet.plot.draw_sis_community
```

```{eval-rst}
.. autofunction:: honet.plot.draw_sis_conmmunity
```

```{eval-rst}
.. autofunction:: honet.plot.draw_sis_init
```

```{eval-rst}
.. autofunction:: honet.plot.draw_sis_init_beta
```

```{eval-rst}
.. autofunction:: honet.plot.draw_sis_phase
```

```{eval-rst}
.. autofunction:: honet.plot.draw_test_assortativity
```

## `honet.export`

导出 LaTeX / CSV / Excel。

honet.export —— 分析结果的导出与读回（CSV / Excel / LaTeX）。

```{eval-rst}
.. autofunction:: honet.export.beautify_global_metrics
```

```{eval-rst}
.. autofunction:: honet.export.beautify_honcommunity
```

```{eval-rst}
.. autofunction:: honet.export.load_global_metrics
```

```{eval-rst}
.. autofunction:: honet.export.load_honcommunity
```

```{eval-rst}
.. autofunction:: honet.export.to_latex
```
