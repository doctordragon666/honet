# %% [markdown]
# # 01 构造与拓扑：Hypergraph 家族
#
# honet 的核心是**高阶交互**——多个主体共同参与一次交互。本 notebook 演示
# 容器层的全部公开类型，以及最常用的结构查询。
#
# 本 notebook 覆盖：
#
# | 主题 | API |
# | --- | --- |
# | 容器类型 | `Hypergraph` / `HypergraphNetwork` / `SimplicialComplexNetwork` |
# | 加权 / 有向 | `HypergraphNetworkW` / `HypergraphNetworkD` |
# | 多层 / 时序 | `MultilayerNetwork` / `TemporalNetwork` |
# | 结构查询 | `incidence_matrix` / `degree_sequence` / `clique_projection` |
#
# **选型口诀**：普通高阶交互用 `Hypergraph`；需要"不向下封闭"的边集用
# `HypergraphNetwork`；要让 `faces()` 自动展开子面就用 `SimplicialComplexNetwork`。

# %%
import honet

# 最小可用构造：只给边，结点自动收集
hg = honet.Hypergraph(edges=[["a", "b", "c"], ["c", "d"], ["d", "e", "f", "g"]])

print("结点顺序:", hg.nodes)
print("规模:", hg.num_nodes, "个结点,", hg.num_edges, "条超边")
print("是否均匀:", hg.is_uniform, "| 各边大小:", hg.edge_sizes)
print("复现:", hg)

# %% [markdown]
# ## 1. 超边是带字段的不可变对象
#
# 每条超边有三个字段：`members`（成员集合）、`label`（可选标签）、
# `weight`（权重，默认 1.0）。成员是 `frozenset`——**顺序无关**，
# 这正是"共同参与一次交互"的语义。

# %%
# 显式给 label 与 weight：适合表达"这篇论文被引用了几次"
hg.add_edge(["a", "c", "e"], label="paper-2024", weight=3.0)

for i, edge in enumerate(hg.edges):
    members = ", ".join(sorted(edge.members))
    print(f"边{i}: {{{members}}}  size={edge.size}  label={edge.label!r}  weight={edge.weight}")

# %% [markdown]
# ## 2. 结构性保护：拒绝重复与非法权重
#
# 静默覆盖数据是最危险的 bug 来源，所以 honet 在构造时就拒绝：
#
# - 成员集合与标签**完全相同**的超边（顺序无关，`["b","c"]` 与 `["c","b"]` 视为同一条）
# - 权重 ≤ 0 的超边
#
# 下面用 `try/except` 演示这两道防线。

# %%
# 防线一：重复超边。
# 注意成员是集合：["b", "c", "a"] 与已有的 ["a", "b", "c"] 是**同一条**，
# 顺序不同不算新边。这正是 frozenset 语义带来的保护。
try:
    hg.add_edge(["b", "c", "a"])
except ValueError as exc:
    print("重复超边被拒绝 ->", exc)

# 防线二：非法权重
try:
    hg.add_edge(["x", "y"], weight=0.0)
except ValueError as exc:
    print("非法权重被拒绝 ->", exc)

print("\n最终图:", hg)

# %% [markdown]
# ## 3. 关联矩阵：一切结构分析的起点
#
# `incidence_matrix(hg)` 返回 `结点数 × 超边数` 的矩阵 `B`，`B[i][j] = 1`
# 表示结点 `i` 属于超边 `j`。用它可以一眼验证结构：
#
# - 每列之和 = 该超边的成员数
# - 每行之和 = 该结点的高阶度

# %%
matrix = honet.incidence_matrix(hg)
print(f"关联矩阵形状: {len(matrix)} × {len(matrix[0])}")

nodes = list(hg.nodes)
for j, edge in enumerate(hg.edges):
    column_sum = sum(row[j] for row in matrix)
    assert column_sum == edge.size, "列和应等于超边成员数"
print("列和全部等于超边大小 ✓")

for i, node in enumerate(nodes):
    row_sum = sum(matrix[i])
    assert row_sum == hg.degree(node), "行和应等于高阶度"
print("行和全部等于高阶度 ✓")

# %% [markdown]
# ## 4. 度序列与团投影
#
# 两个最常用的结构摘要：
#
# - `degree_sequence(hg)`：每个结点参与几条超边（高阶度），降序排列
# - `clique_projection(hg)`：把超图展开成成对网络，每条 `k` 元超边产生
#   `C(k,2) = k(k-1)/2` 条成对边

# %%
print("度序列（降序）:", honet.degree_sequence(hg))

projection = honet.clique_projection(hg)
pairs = {(x, y) for x, ys in projection.items() for y in ys if x < y}

# 每条 k 元超边贡献 C(k,2) 条成对边，但**不同超边之间会重叠**，
# 所以投影后的去重边数 ≤ ΣC(k,2)。下面把两个数都算出来看差多少。
raw_pairs = sum(edge.size * (edge.size - 1) // 2 for edge in hg.edges)
print(f"ΣC(k,2) = {raw_pairs}（含跨边重复）")
print(f"去重后投影边数 = {len(pairs)}")
print(f"重叠抵消了 {raw_pairs - len(pairs)} 条")

# %% [markdown]
# ## 5. 投影的信息损失
#
# 这是高阶分析里最容易踩的坑：**两个不同的超图可以有完全相同的团投影**。
#
# 下面左边是"三人同场"，右边是"三次两两合作"——投影一模一样，但前者
# 记录了集体事件，后者没有。若后续分析只依赖投影，两者不可区分。

# %%
triple = honet.Hypergraph(edges=[["a", "b", "c"]])
pairwise = honet.Hypergraph(edges=[["a", "b"], ["b", "c"], ["a", "c"]])

proj_triple = honet.clique_projection(triple)
proj_pairwise = honet.clique_projection(pairwise)


# 统一成无向边集合再比较
def as_edges(proj):
    return {(x, y) for x, ys in proj.items() for y in ys if x < y}


same = as_edges(proj_triple) == as_edges(proj_pairwise)
print("投影是否相同:", same)
print("  三人同场  ->", sorted(as_edges(proj_triple)))
print("  三次两两  ->", sorted(as_edges(proj_pairwise)))
print("但超边数不同:", triple.num_edges, "vs", pairwise.num_edges)

# %% [markdown]
# ## 6. 其他容器类型速览
#
# 上面用的是最通用的 `Hypergraph`。honet 还提供针对特定语义的容器：

# %%
# (a) HypergraphNetwork：不向下封闭的边集（与 Hypergraph 是不同类型）
hnet = honet.HypergraphNetwork(edges=[["a", "b", "c"], ["b", "c"]])
print("HypergraphNetwork:", hnet)

# (b) SimplicialComplexNetwork：只加极大面，子面由 faces() 自动展开
sc = honet.SimplicialComplexNetwork()
sc.add_facet(["a", "b", "c"])
print("单纯复形 faces():", sorted(map(lambda f: tuple(sorted(f)), sc.faces())))

# (c) 加权变体：无权变体给非 1.0 权重会抛 TypeError
hgw = honet.HypergraphNetworkW(edges=[["a", "b", "c"]])
hgw.add_edge(["a", "b"], weight=2.5)
print("加权变体:", hgw)

# %% [markdown]
# ## 小结
#
# - **构造**：`Hypergraph(edges=...)` 最简；`add_edge(members, label=..., weight=...)` 补充字段
# - **语义**：成员是 `frozenset`（顺序无关），重复与非法权重在构造时即拒绝
# - **结构**：`incidence_matrix` 是矩阵化起点，`degree_sequence` / `clique_projection` 是两个常用摘要
# - **陷阱**：投影会丢失高阶信息，做高阶分析时不要一上来就投影
#
# 下一步：`02` 讲全部结构指标，`03` 讲随机超图生成器。
