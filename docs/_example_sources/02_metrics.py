# %% [markdown]
# # 02 结构指标：从度到可约性
#
# `honet.metrics` 提供 14 个结构指标。本 notebook 按**分析目的**分组演示，
# 而不是按字母序罗列——因为选指标的关键是知道"它回答什么问题"。
#
# > **重要前提**：`honet.metrics` 全部接收 **hypernetx 超图**，不是
# > `honet.Hypergraph`。两者分工不同：`honet.Hypergraph` 是零依赖容器
# > （构造、查询、投影），分析类函数则建立在 hypernetx 之上。本 notebook
# > 因此用 `hnx.Hypergraph` 作输入。
#
# | 目的 | 指标 |
# | --- | --- |
# | 局部重要性 | `weighted_degree` / `clustering_coefficient` / `betweenness_centrality` / `eigenvector_centrality` |
# | 高阶度分布 | `get_avg_degree_by_edge_size` / `excess_average_degree` |
# | 整体异质性 | `calc_entropy` / `calc_natural_connectivity` / `calc_reducibility` |
# | 批量对比 | `hnxs_global` |

# %%
import hypernetx as hnx
import pandas as pd

import honet

# 用 hypernetx 构造（metrics 的输入类型）。
# 字典的 key 是超边名，value 是成员列表。
H = hnx.Hypergraph(
    {
        "g1": ["a", "b", "c"],  # 小组 1
        "g1_pair": ["a", "b"],  # 小组 1 的紧密二人组
        "bridge": ["c", "d", "e"],  # 桥接三元组
        "g2a": ["d", "e", "f"],  # 小组 2
        "g2b": ["e", "f", "g"],  # 小组 2 的另一个三元组
    }
)
print("类型:", type(H).__module__ + "." + type(H).__name__)
print("规模:", len(H.nodes), "结点 /", len(H.edges), "超边")

# %% [markdown]
# ## 1. 局部重要性：四个指标，四种"重要"
#
# 同一个结点在不同指标下排名可能完全不同。这不是矛盾，而是因为它们
# 回答的问题不同：
#
# - **weighted_degree**：参与了多少"交互强度"——纯数量
# - **clustering_coefficient**：邻居之间是否也彼此相连——局部抱团
# - **betweenness_centrality**：有多少最短路径经过它——桥接作用
# - **eigenvector_centrality**：与"重要的结点"相连的程度——递归重要性

# %%
metrics = {
    "加权度": honet.metrics.weighted_degree(H),
    "聚类系数": honet.metrics.clustering_coefficient(H),
    "介数中心性": honet.metrics.betweenness_centrality(H),
    "特征向量中心性": honet.metrics.eigenvector_centrality(H),
}

frame = pd.DataFrame(metrics).round(3)
frame = frame.sort_values("加权度", ascending=False)
print(frame.to_string())

# %% [markdown]
# 观察上表：加权度最高的是 `e`（参与 3 条边），但**介数中心性最高的却是 `c`**。
# 这不是矛盾——`c` 位于两个小组的接缝上，虽然参与次数不是最多，却是大量
# 最短路径的必经之地。这正是"桥接结点"的典型特征，也是**攻击分析**里
# 最该优先移除的结点。
#
# 注意 `eigenvector_centrality` 会出现**负值**：它基于归一化拉普拉斯的
# Fiedler 向量，符号本身无意义，只有相对大小（以及符号分组）有意义。

# %%
rank_weighted = sorted(frame.index, key=lambda n: -frame.loc[n, "加权度"])
rank_between = sorted(frame.index, key=lambda n: -frame.loc[n, "介数中心性"])
print("按加权度排序:", rank_weighted)
print("按介数排序:  ", rank_between)
print("两种排序是否一致:", rank_weighted == rank_between)
print("\n介数最高的结点:", rank_between[0])

# %% [markdown]
# ## 2. 高阶度分布：区分"参与次数"与"参与规模"
#
# 传统网络只有一种度。超图里可以按超边大小分层看：
#
# - `get_avg_degree_by_edge_size(H)`：每种大小 `m` 下，结点平均度数
# - `excess_average_degree(H)`：邻居的平均超度（"你的邻居有多活跃"）
#
# 后者是**同类相配性**分析的基础：如果活跃结点倾向于连接活跃结点，
# 余平均度会随度上升而上升。

# %%
by_size = honet.metrics.get_avg_degree_by_edge_size(H)
print("每种超边大小下的平均度:")
for size in sorted(by_size):
    print(f"  m={size}: {by_size[size]:.3f}")

excess = honet.metrics.excess_average_degree(H)
print("\n余平均度（邻居的平均超度）:")
for node in sorted(excess):
    print(f"  {node}: {excess[node]:.3f}")

# %% [markdown]
# ## 3. 整体异质性：熵与自然连通度
#
# - **`calc_entropy`**：度分布的香农熵。越大表示度分布越均匀（越"平等"）；
#   越小表示越集中在少数结点（越"集权"）。
# - **`calc_natural_connectivity`**：基于拉普拉斯特征值的鲁棒性指标。
#   越大表示网络越难被摧毁。
# - **`calc_reducibility`**：衡量"能否用更简单的高阶结构近似"。

# %%
entropy = honet.metrics.calc_entropy(H)
natural = honet.metrics.calc_natural_connectivity(H)
reducibility = honet.metrics.calc_reducibility(H)

print(f"度分布熵:       {entropy:.4f}")
print(f"自然连通度:     {natural:.4f}")
print(f"可约性:         {reducibility:.4f}")

# %% [markdown]
# 对比一个**极端集中**的超图（一个中心结点参与所有边），熵应该更低：

# %%
# 星形超图：h 参与全部 3 条边，其余结点各只参与 1 条
star = hnx.Hypergraph(
    {
        "s1": ["h", "a", "b"],
        "s2": ["h", "c", "d"],
        "s3": ["h", "e", "f"],
    }
)
star_entropy = honet.metrics.calc_entropy(star)
print(f"星形超图 度分布熵: {star_entropy:.4f}")
print(f"上面例子 度分布熵: {entropy:.4f}")
print("星形更集中 -> 熵更低:", star_entropy < entropy)

# %% [markdown]
# ## 4. 批量对比：hnxs_global
#
# 做论文实验时经常要比较"多个网络 × 多个指标"。`hnxs_global` 接收
# `{名字: 网络}` 的字典，一次算出全局指标表。

# %%
networks = {
    "示例网络": H,
    "星形": star,
    "三人团": hnx.Hypergraph({"t1": ["a", "b", "c"]}),
    "两两合作": hnx.Hypergraph({"p1": ["a", "b"], "p2": ["b", "c"], "p3": ["a", "c"]}),
}

summary = honet.metrics.hnxs_global(networks)
print("返回类型:", type(summary).__name__)
print(summary)
print("\n注：当前实现只汇总「可约性」一列，不是全部指标。")

# %% [markdown]
# ## 小结
#
# | 你想知道 | 用哪个 |
# | --- | --- |
# | 谁参与最多 | `weighted_degree` |
# | 谁是桥 | `betweenness_centrality` |
# | 谁和重要结点相连 | `eigenvector_centrality` |
# | 邻居抱团吗 | `clustering_coefficient` / `excess_average_degree` |
# | 网络整体平等吗 | `calc_entropy` |
# | 网络抗打击吗 | `calc_natural_connectivity` |
# | 多个网络对比 | `hnxs_global` |
#
# **两条关键提醒**：
#
# 1. **输入类型**：`honet.metrics` 收 hypernetx 超图，不收 `honet.Hypergraph`。
#    先用 `hnx.Hypergraph({...})` 构造，或用 `honet.convert` 系列互转。
# 2. **指标不可互换**：不同指标的排序往往不同（本 notebook 已演示加权度 vs 介数）。
#    报告"重要结点"时必须说明用的是哪个指标，否则结论不可复现。

# %%
print("本例结点重要性排名（加权度）:", rank_weighted)
print("本例结点重要性排名（介数）:  ", rank_between)
