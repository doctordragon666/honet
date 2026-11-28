# %% [markdown]
# # 03 随机超图生成与格式互转
#
# 做实验时经常需要"有已知结构的合成网络"来验证算法，也需要在
# hypernetx / XGI / NetworkX 之间搬数据。本 notebook 演示这两组 API。
#
# > ## ⚠️ 先记住这条：honet 里有**两种**超图对象
# >
# > | 类型 | 谁返回 | 谁接受 |
# > | --- | --- | --- |
# > | `honet.Hypergraph` | `honet.Hypergraph(...)` 手工构造 | `incidence_matrix` / `degree_sequence` / `clique_projection` |
# > | `hypernetx.Hypergraph` | `honet.build.*` 四个生成器 | `honet.metrics.*` / `honet.community.*` |
# >
# > 两者**不能互换**。`honet.build` 生成的是 hypernetx 对象，所以它
# > **不能**直接喂给 `honet.incidence_matrix`（后者需要 `.num_nodes`，
# > hypernetx 没有这个属性）。本 notebook 会演示怎么跨过这道边界。
#
# | 组 | API |
# | --- | --- |
# | 随机生成 | `generate_random_hypernet` / `scale_free_hypergraph` / `small_world_hypergraph` / `clustered_hypergraph` |
# | 从表格构建 | `build_hypernet` |
# | 格式互转 | `convert_hypernetx_to_xgi` / `convert_hypernetx_to_nxgraph` / `convert_xgi_to_hypernetx` |
# | 元数据 | `get_hyperedge_weights` |

# %%
import hypernetx as hnx
import pandas as pd

import honet

SEED = 7
N_NODES, N_EDGES = 30, 60

# %% [markdown]
# ## 1. 参数语义：`m` 是超边数，不是结点数
#
# 以 `scale_free_hypergraph(n, k, m, gamma)` 为例：
#
# - `n`：**结点池**大小（编号 `0..n-1`）
# - `k`：每条超边参与结点数的上界（tail 取 `k` 个，head 再取 `1..k-1` 个）
# - `m`：**超边总数**
#
# 所以 `n=20, m=2` 只会生成 2 条边，被覆盖的结点远少于 20 个——这不是
# bug，而是参数含义。下面用足够的 `m` 让结点池被覆盖。

# %%
random_hg = honet.build.generate_random_hypernet(
    n_nodes=N_NODES, n_edges=N_EDGES, max_edge_size=4, seed=SEED
)
print("随机超图:      类型 =", type(random_hg).__module__ + "." + type(random_hg).__name__)
print("  规模 (结点, 超边):", random_hg.shape)
# 注意：hypernetx 2.4.3 的 __repr__ 返回 None，直接 print(H) 会得到
# "None <class ...>"，所以这里一律用 .shape / .nodes / .edges 读规模。

# %% [markdown]
# ## 2. 三个结构特化生成器
#
# - `scale_free_hypergraph`：优先连接 → **幂律度分布**（少数枢纽）
# - `small_world_hypergraph`：规则环 + 重连 → **高聚类 + 短路径**
# - `clustered_hypergraph`：显式社区 → 用来测试社区发现能否找回已知簇

# %%
scale_free = honet.build.scale_free_hypergraph(n=N_NODES, k=3, m=N_EDGES, gamma=2.5, seed=SEED)
print("无标度超图:    规模 =", scale_free.shape)

small_world = honet.build.small_world_hypergraph(
    n=N_NODES, k=3, m=N_EDGES, rewire_prob=0.1, seed=SEED
)
print("小世界超图:    规模 =", small_world.shape)

clustered = honet.build.clustered_hypergraph(n=N_NODES, k=3, m=N_EDGES, num_clusters=3, seed=SEED)
print("带社区超图:    规模 =", clustered.shape)

# %% [markdown]
# ## 3. 跨过类型边界：hypernetx → honet.Hypergraph
#
# 这是本 notebook 最实用的一段。生成器给了 hypernetx 对象，但
# `incidence_matrix` 只认 `honet.Hypergraph`。**没有官方转换器**，
# 需要手动提取成员重建：
#
# ```python
# edges = [sorted(H.edges[e]) for e in H.edges]
# hg = honet.Hypergraph(edges=edges)
# ```


# %%
def hnx_to_honet(H):
    """把 hypernetx 超图转成 honet.Hypergraph。

    需要处理一处**语义差异**：hypernetx 允许两条成员完全相同的超边（只是
    uid 不同），而 ``honet.Hypergraph`` 会把它们视为重复并拒绝。生成器
    产生的随机网络经常出现这种情况，因此这里按成员集合去重。
    """
    seen = set()
    edges = []
    for uid in H.edges:
        members = frozenset(H.edges[uid])
        if members in seen:
            continue
        seen.add(members)
        edges.append(sorted(members))
    return honet.Hypergraph(edges=edges)


random_hg_honet = hnx_to_honet(random_hg)
print("转换后:", random_hg_honet)
print(
    f"原始 {len(random_hg.edges)} 条 -> 去重后 {random_hg_honet.num_edges} 条"
    f"（去掉 {len(random_hg.edges) - random_hg_honet.num_edges} 条重复）"
)

matrix = honet.incidence_matrix(random_hg_honet)
print("关联矩阵形状:", f"{len(matrix)} × {len(matrix[0])}")

# %% [markdown]
# 转换后 `degree_sequence` / `clique_projection` 也都能用了：

# %%
print("度序列（前 10）:", honet.degree_sequence(random_hg_honet)[:10])

# %% [markdown]
# ## 4. 可复现性验证
#
# 同样的 `seed` 必须给出同样的网络。用关联矩阵比较比逐个对象比较更可靠。

# %%
again = honet.build.generate_random_hypernet(
    n_nodes=N_NODES, n_edges=N_EDGES, max_edge_size=4, seed=SEED
)
print("同 seed 两次结果相同:", honet.incidence_matrix(hnx_to_honet(again)) == matrix)

different = honet.build.generate_random_hypernet(
    n_nodes=N_NODES, n_edges=N_EDGES, max_edge_size=4, seed=SEED + 1
)
print("不同 seed 结果不同:", honet.incidence_matrix(hnx_to_honet(different)) != matrix)

# %% [markdown]
# ## 5. 度分布差异：无标度 vs 随机
#
# 生成器的价值在于**度分布形状不同**。下面直接比较三者的度序列：

# %%
# 注意：三者结点数不同（无标度网络只覆盖到 18 个结点），不能直接并排成表。
# 这里按「排名位置」对齐——第 k 名的度值，而不是第 k 个结点的度值。
seq_random = sorted(honet.degree_sequence(random_hg_honet), reverse=True)
seq_scale = sorted(honet.degree_sequence(hnx_to_honet(scale_free)), reverse=True)
seq_world = sorted(honet.degree_sequence(hnx_to_honet(small_world)), reverse=True)

depth = max(len(seq_random), len(seq_scale), len(seq_world))


def pad(values, size):
    """用 0 补齐到统一长度（0 = 该排名位置没有结点）。"""
    return values + [0] * (size - len(values))


comparison = pd.DataFrame(
    {
        "随机": pad(seq_random, depth),
        "无标度": pad(seq_scale, depth),
        "小世界": pad(seq_world, depth),
    }
)
print("度序列（按排名位置对齐，降序，前 10 位）:")
print(comparison.head(10).to_string(index=False))

print(
    "\n结点数: 随机 =",
    len(seq_random),
    "| 无标度 =",
    len(seq_scale),
    "| 小世界 =",
    len(seq_world),
)
print("最高度: 随机 =", seq_random[0], "| 无标度 =", seq_scale[0], "| 小世界 =", seq_world[0])
print("无标度的最高度更大（枢纽效应）:", seq_scale[0] >= seq_random[0])

# %% [markdown]
# ## 6. 从表格构建：build_hypernet
#
# `build_hypernet` 是**领域专用**构建器（来自贸易协定分析），列名有硬性要求：
#
# - `agreement`：超边标识（一行 = 某协议包含某国家）
# - `country`：成员
# - `depth`：超边深度（会同时写进边的 `weight` 与 `depth` 属性）
#
# 不是任意长表都能直接喂进去——通用场景请自己用 `honet.Hypergraph(edges=...)`。

# %%
records = pd.DataFrame(
    {
        # 两个协议，各含若干国家
        "agreement": ["A1", "A1", "A1", "A2", "A2", "A2"],
        "country": ["CN", "JP", "KR", "CN", "DE", "FR"],
        "depth": [3, 3, 3, 2, 2, 2],
        "year": [2020, 2020, 2020, 2020, 2020, 2020],
    }
)
print(records.to_string(index=False))

built = honet.build.build_hypernet(records)
print("\n从表格构建:    类型 =", type(built).__module__ + "." + type(built).__name__)
print("  规模:", built.shape)
print("  结点:", sorted(built.nodes))
print("  超边:", list(built.edges))

# %% [markdown]
# 给了 `year` 列时可以按年份筛选，此时深度会挂到超边权重上：

# %%
built_2020 = honet.build.build_hypernet(records, i=2020)
print("按 2020 年筛选:  规模 =", built_2020.shape)
for uid in built_2020.edges:
    edge = built_2020.edges[uid]
    print(f"  {uid}: weight={edge.weight} depth={getattr(edge, 'depth', '<无>')}")

# %% [markdown]
# ## 7. 格式互转：XGI / NetworkX
#
# `convert` 模块在 hypernetx 与下游生态之间桥接。

# %%
H = hnx.Hypergraph(
    {
        "e1": ["a", "b", "c"],
        "e2": ["c", "d"],
        "e3": ["d", "e", "f"],
    }
)
print("源:", type(H).__name__)

# (a) -> XGI（保留超边映射，可回转）
xgi_graph, edge_map = honet.convert.convert_hypernetx_to_xgi(H)
print("\nXGI:", type(xgi_graph).__module__ + "." + type(xgi_graph).__name__)
print("  边映射条目数:", len(edge_map))

# (b) -> NetworkX（团投影，**会丢失高阶信息**）
nx_graph = honet.convert.convert_hypernetx_to_nxgraph(H)
print("\nNetworkX:", type(nx_graph).__name__)
print("  结点数:", nx_graph.number_of_nodes(), "边数:", nx_graph.number_of_edges())

# (c) XGI -> hypernetx（回程）
back = honet.convert.convert_xgi_to_hypernetx(xgi_graph)
print("\n回转:", type(back).__name__, "| 规模:", back.shape)

# %% [markdown]
# ## 8. 元数据读取

# %%
weights = honet.convert.get_hyperedge_weights(H)
print("超边权重（无权重时默认 1.0）:", weights)

# %% [markdown]
# ## 9. 跨类型边界：hypernetx → honet.Hypergraph
#
# 前面反复提到「两类超图对象不能混用」。`honet.build.*` 产出的是 hypernetx
# 对象，而 `honet.incidence_matrix` / `degree_sequence` / `clique_projection`
# 只收 `honet.Hypergraph`。手写转换有两个坑：
#
# - **丢边名与权重**：`Hypergraph.__init__` 的 `edges` 参数只接受成员列表；
# - **重复成员集冲突**：hypernetx 允许 `{"e1": ["a","b"], "e2": ["b","a"]}`，
#   而 `honet.Hypergraph` 的判重键是「成员集合 + 标签」，标签为空时第二条会报错。
#
# `convert_hypernetx_to_honet` 把这些细节收敛到一处。

# %%
g = honet.convert.convert_hypernetx_to_honet(H)
print("转换结果类型:", type(g).__name__)
print("规模:", g.num_nodes, "结点 ×", g.num_edges, "超边")
print("边名与权重:", [(e.label, e.weight) for e in g.edges])

# 转换后即可直接使用拓扑层 API
print("度序列:", honet.degree_sequence(g))

# %% [markdown]
# > **注意「权重」的两个口径**。上面转换出的 `e1` 权重是 1.0，而第 8 节
# > `get_hyperedge_weights` 给出的是 3.0——两者不是同一个东西：
# >
# > | 口径 | 存在哪 | 含义 | 是否迁移 |
# > | --- | --- | --- | --- |
# > | **边级权重** | `H.edges.properties["weight"]` | 整条边一个标量，默认 1.0 | 迁移到 `Hyperedge.weight` |
# > | **关联权重** | 长表 `(edge, node, weight)` | 每个结点在该边里的份量，列和=超边权重 | 不迁移（容器无对应字段） |
# >
# > 需要保留关联权重时，直接读 `H.edges.properties` 或调 `get_hyperedge_weights`。

# %%
# 重复成员集：默认合并为一条（权重相加），不合并则靠不同边名保留
import hypernetx as hnx

H_dup = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "a"]})
print("原始边数:", len(list(H_dup.edges)))
print("合并后边数:", honet.convert.convert_hypernetx_to_honet(H_dup).num_edges)
print(
    "不合并边数:",
    honet.convert.convert_hypernetx_to_honet(H_dup, merge_duplicates=False).num_edges,
)

# %% [markdown]
# ## 小结
#
# | 需求 | 用哪个 |
# | --- | --- |
# | 通用随机基线 | `generate_random_hypernet` |
# | 测试枢纽效应 | `scale_free_hypergraph` |
# | 测试聚类/传播速度 | `small_world_hypergraph` |
# | 测试社区发现 | `clustered_hypergraph` |
# | 从长表构建 | `build_hypernet` |
# | 交给 XGI 生态 | `convert_hypernetx_to_xgi` |
# | 交给 NetworkX 算法 | `convert_hypernetx_to_nxgraph`（**注意丢高阶信息**） |
#
# **三条关键提醒**：
#
# 1. **类型边界**：`honet.build.*` 返回 hypernetx 对象，而
#    `honet.incidence_matrix` / `degree_sequence` / `clique_projection`
#    只收 `honet.Hypergraph`。跨边界用 `convert_hypernetx_to_honet`（见第 9 节），
#    不要手写转换——手写会丢边名、丢权重，且遇到重复成员集会直接报错。
# 2. `m` 是**超边数**不是结点数；`n` 是结点池，未被抽中就不出现。
# 3. 所有生成器都接受 `seed`，做实验必须固定，否则结果不可复现。
