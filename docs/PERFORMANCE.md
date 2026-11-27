# 性能设计

本文记录 honet 在「减少重复计算」上的设计：为什么要有拓扑索引、缓存怎么校验、
哪些指标走哪条路径。

## 1. 两个核心模块

| 模块 | 职责 |
| --- | --- |
| `honet._topology` | 一次性把 hypernetx 对象抽取成 `Topology` 快照（成员、隶属、度、规模） |
| `honet._cache` | 按「结构特征 + 内容指纹」校验的缓存，挂在对象生命周期上 |

两者配合：**同一超图上的多次分析共享同一份结构，结构没变就不重算**。

## 2. 为什么要索引

hypernetx 的访问接口有隐藏成本：

| 操作 | 500 条边耗时 |
| --- | ---: |
| `H.edges.incidence_dict` | ~0.00 ms |
| `sum(len(H.edges[e]) for e in H.edges)` | ~3.2 ms |

`H.edges[e]` 每次访问都会重建成员列表。多个指标各自遍历一次，就重复付一次代价。
`Topology` 只走一次 `incidence_dict`，之后所有指标共享。

实测（300 结点 / 800 超边，连续算 6 个指标）：

```
无共享（每次重建）：  163.82 ms
共享索引：             84.56 ms
加速：1.9×
```

## 3. 缓存怎么判断「数据没变」

两级校验，代价从低到高：

1. **结构特征**（每次调用都做，~0.02ms）
   `(超边数, 结点数, 成员总数)` 三个整数。任何增删边 / 增删结点都会改变其一。
2. **内容指纹**（仅在第一级变化时才算）
   SHA-256，成员排序后拼接，因此与插入顺序、对象身份无关。用于区分
   「删一条又加一条、数量巧合相同」的情况。

> 第一级已覆盖绝大多数「结构没变」的情形，因此常见路径根本不计算指纹。

### 对象身份与生命周期

`Hypergraph` 未实现 `__hash__`，不能作为 `WeakKeyDictionary` 的键。因此缓存按
`id(H)` 索引，并注册 `weakref.finalize` 在对象回收时删除条目——不会泄漏。

### 可变性

超图是可变的。每次取缓存前都重新校验结构特征，**结构一变立刻丢弃旧条目**，
不会返回过期结果。这一点有专门的回归测试
（`tests/test_cache.py::TestCacheHit::test_structure_change_invalidates`）。

## 4. 各指标的计算路径

| 指标 | 路径 | 说明 |
| --- | --- | --- |
| `weighted_degree` | 索引 | 无权时直接取度值，不再构造关联矩阵 |
| `clustering_coefficient` | 索引 | 复杂度从 O(结点×超边) 降到 O(Σ 成员数²) |
| `excess_average_degree` | 索引 | 邻居表按需填充并复用 |
| `calc_entropy` | 索引 | 直接用度序列，不调 `dist_stats` |
| `get_avg_degree_by_edge_size` | 索引 | 按超边规模分桶，不构造稠密矩阵 |
| `betweenness_centrality` | 索引 + 缓存 | 二部图只建一次；支持 `samples` 近似 |
| `calc_natural_connectivity` | 缓存 | 特征值分解最贵，结果复用 |
| `calc_clustering_coefficient_g` | 复用局部 | 直接对局部结果取均值 |
| `calc_reducibility` | 缓存 | 跨格式转换结果也缓存 |

**全局指标尽量由局部指标导出**，例如全局聚类系数就是局部聚类系数的均值，
不重复实现一套逻辑。

## 5. 实测收益

200 结点 / 500 超边 / 每条 4 成员：

| 函数 | 冷启动 | 命中缓存 | 加速 |
| --- | ---: | ---: | ---: |
| `weighted_degree` | 5.94 ms | 0.03 ms | 174× |
| `clustering_coefficient` | 43.83 ms | 0.05 ms | 822× |
| `excess_average_degree` | 3.28 ms | 0.05 ms | 61× |
| `betweenness_centrality` | 1219.40 ms | 0.06 ms | 21060× |
| `calc_clustering_coefficient_g` | 51.85 ms | 0.13 ms | 404× |
| `calc_entropy` | 1.82 ms | 0.03 ms | 55× |
| `calc_natural_connectivity` | 4.65 ms | 0.04 ms | 122× |

复现：

```bash
uv run python docs/bench_metrics.py
uv run python docs/_verify_shared_index.py
```

## 6. 什么时候不该依赖缓存

- **超图被就地修改后**：缓存会自动失效，但如果你持有的是旧结果变量，那仍是旧值。
- **想对比两种结构**：请分别调用，缓存按对象区分，不会混淆。
- **内存敏感场景**：单对象最多 64 条缓存条目，可用 `honet._cache.clear(H)` 手动清理。
