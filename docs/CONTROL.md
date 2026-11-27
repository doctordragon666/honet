# 控制框架

`honet.control` 有两层能力，分别回答控制论的两个不同问题：

| 层 | 问题 | 依赖 | 主要入口 |
| --- | --- | --- | --- |
| 拓扑层 | **控制哪些结点** | HAT（`honet[hat]`） | `calc_drivers` / `detect_dilation_and_drivers` |
| 动力学层 | **怎么控制** | numpy / scipy | `HypergraphControlSystem` / `minimum_energy_control` |
| 单纯形层 | **牵制哪个单纯形** | numpy | `rank_simplices` / `pinning_matrix` |

拓扑层回答「控制哪些结点」，但不含动力学、耦合与控制律。后两层补齐这部分。

---

## 一、动力学层：超图网络可控性

移植自 Ma, Bao & Ma, *Physica A* **653** (2024) 130108,
[doi:10.1016/j.physa.2024.130108](https://doi.org/10.1016/j.physa.2024.130108)。

### 系统模型

论文式 (4) 把「结点自身动力学 + 高阶耦合 + 外部输入」组装成线性系统：

```
Ẋ(t) = 𝒜X(t) + ℬU(t)
𝒜 = I_N ⊗ A + c/(k-1) · L ⊗ Λ
ℬ = Δ ⊗ B
```

其中 `A` 是结点自身动力学、`Λ` 是内耦合矩阵、`L` 是 m-阶广义 Laplacian、
`Δ = diag(δ)` 标记受控结点、`B` 是输入矩阵。

```python
import numpy as np
from honet.control import HypergraphControlSystem, hypergraph_laplacian, kalman_rank

# 论文 Example 2.1 的超边（结点用 1 基编号）
edges = {
    "e1": ([1, 2, 3], [(1, 2), (2, 3)]),
    "e2": ([2, 3, 4], [(2, 3)]),
    "e3": ([3, 4, 5], [(3, 5), (4, 5)]),
}
mats = hypergraph_laplacian(edges, num_nodes=5)
L = mats[2] + mats[3]

system = HypergraphControlSystem(
    A=np.eye(2),
    L=L,
    Lambda=np.eye(2),
    B=np.array([[0.0, 1.0], [1.0, 0.0]]),
    controlled_nodes=(1, 2, 4),   # 论文的结点 2, 3, 5（0 基）
    coupling=1.0,
    order=3,
)
Ac, Bc = system.to_blocks()
print(kalman_rank(Ac, Bc))   # 10，与论文一致
```

### 广义 Laplacian 的求和口径

`L⁽ᵐ⁾` 的构造有一个容易踩的坑。对一条 k-元超边的有向对 `(i, j)`：

- **m=2 阶**：项落在 `(i, j)`，每个「第三点」都累加一次；
- **m≥3 阶**：项落在 `(i, 第三点)`，**不是** `(i, j)`。

写成伪代码就是参考实现里的两行：

```
T2(i, j) += fac          # 对每个 other
T3(i, other) += fac      # 对每个 other
```

`fac = 1/(k-1)`，对角线最后统一用「负行和」补齐。实现若把 m≥3 的项也写到
`(i, j)`，得到的 `L⁽³⁾` 会与 `L⁽²⁾` 完全相同——测试 `test_l3_matches_paper`
专门锁住这一点（对照论文原文第 3 页给出的矩阵，误差必须为 0）。

### 三个可控性判据

| 函数 | 判据 | 说明 |
| --- | --- | --- |
| `kalman_rank(A, B)` | Kalman 秩 | `rank[B, AB, ..., A^{d-1}B]` |
| `pbh_rank(A, B)` | PBH 秩 | `min_λ rank[λI-A, B]` |
| `gramian_rank(A, B, T)` | Gramian 秩 | 有限时间可控性 Gramian |

三者互相独立，测试里交叉验证必须一致。

### 最小能量控制律

```python
from honet.control import minimum_energy_control

result = minimum_energy_control(
    Ac, Bc,
    X0=np.arange(1, 11, dtype=float),
    X_target=np.zeros(10),
    T=2.0,
)
print(result.final_error)   # 3.9e-04
print(result.energy)        # 控制能量 ∫‖U‖²dt
```

控制律为：

```
U(t) = ℬᵀ e^{𝒜ᵀ(T-t)} W(T)⁻¹ (X(T) - e^{𝒜T}X(0))
W(T) = ∫₀ᵀ e^{𝒜s}ℬℬᵀe^{𝒜ᵀs} ds
```

**推进器精度**：`_propagate` 用分段线性插值 + 增广矩阵指数做**解析精确**的
段内积分，因此终点误差随采样点数按 `1/n²` 二阶收敛（默认 1000 点）。
若改成零阶保持（只用左端点值），会退化为 `1/n` 一阶——同样的 4000 点下
误差从 `3.5e-05` 劣化到 `2.9e-01`。

### 论文四例的复现结果

| 例子 | 类型 | Kalman 秩 | 论文值 |
| --- | --- | ---: | ---: |
| Ex 2.1 | 有向 3-均匀超图 | 10 | 10 |
| Ex 2.2 | 有向 3-均匀超图 | 9 | 9 (< 12) |
| Ex 3.1 | 2-阶单纯复形 | 8 | 8 |
| Ex 3.2 | 2-阶单纯复形 | 6 | 6 (< 8) |

Ex 2.1 与 Ex 3.1 的最小能量驱动误差分别为 `3.9e-04` 与 `2.9e-05`，均 < 1e-3。

论文的核心论点（Remark 3）也已复现：**高阶交互能让不可控的成对网络变可控**。
Ex 2.1 对应的成对网络秩只有 6，而超图版本是 10
（`test_pairwise_uncontrollable_but_hypergraph_controllable`）。

---

## 二、单纯形层：牵制哪个单纯形

移植自周进, 李博, 陆君安, 史定华, *中国科学: 信息科学* **54**(3): 708–718 (2024),
[doi:10.1360/SSI-2023-0142](https://doi.org/10.1360/SSI-2023-0142)。

### 选择指标 Δ^s

```
Δ^s = Σ_{d=1..D} Σ_{(s,t) ∈ 该 d-单纯形} (v_s - v_t)²
    = 2 · Σ_{无序对 (s,t)} (v_s - v_t)²
```

`v` 是广义 Laplacian `L = c₁L⁽¹⁾ + c₂L⁽²⁾` 的 Fiedler 向量。`Δ^s` 越大，
牵制该单纯形对 `λ₂` 的提升越多，全网同步越快。

**求和口径是最容易错的地方**：`d=1` 部分（3 条 1-单纯形）与 `d=2` 部分
（2-单纯形自身）各含同样的 3 个无序对，因此必须乘 2。只写一重求和得到的
恰好是论文值的一半——**排序仍对、数值全错**，这种错误很难被排序测试发现。

### 用法

```python
import numpy as np
from honet.control import (
    generalized_laplacian, fiedler_vector, rank_simplices, pinning_matrix,
)

L = generalized_laplacian(edges, triangles, num_nodes=N, c1=1.0, c2=1.0)
v = fiedler_vector(L)
ranked = rank_simplices(triangles, v)

print(ranked[0].simplex, ranked[0].delta, ranked[0].rank)  # 最该牵制的

# 牵制：在选中单纯形的 1-单纯形上加权重
E = pinning_matrix([ranked[0].simplex], N, weight=1.0)
L_pinned = L + E
```

### 定理 3 的系数关系

一阶近似 `Δλ₂ ≈ ε·Δ^s` 对应的是 1-单纯形与 2-单纯形权重**同时** `+ε`；
若只改 1-单纯形（论文图 2 的设定，`β` 保持 1），则为 `Δλ₂ ≈ ε·Δ^s/2`。

`pinning_matrix` 的 `include_simplex` 参数区分这两种情形：

```python
pinning_matrix([s], N, weight=eps, include_simplex=True)   # Δλ₂ ≈ ε·Δ^s
pinning_matrix([s], N, weight=eps, include_simplex=False)  # Δλ₂ ≈ ε·Δ^s/2
```

实测（脑网络，`(6,14,15)` 三角形，`Δ^s = 3.991993`）：

| ε | 预测 `ε·Δ^s` | 实测 `Δλ₂` |
| ---: | ---: | ---: |
| 1e-3 | 0.003992 | 0.003992 |
| 1e-2 | 0.039920 | 0.039917 |

### 脑网络复现结果

用论文的 `bn-cat-mixed-species_brain_1.edges`（65 结点）：

| 指标 | 论文 | 本实现 |
| --- | ---: | ---: |
| 结点数 | 65 | 65 |
| 1-单纯形 | 730 | 730 |
| 2-单纯形 | 3613 | 3613 |
| λ₂ | 9.0417925 | 9.04792548（相对差 6.8e-4） |
| λ₂ 重数 | 单根 | 1 |

表 1 的 12 个 2-单纯形**排名 12/12 完全一致**，`Δ^s` 最大相对误差 3e-3
（来自论文只印 6 位小数的量化，如 `0.000596` 的量化步长本身就有 1.7e-3）。

`λ₂` 的 6.8e-4 相对差无法从现有材料判断来源（可能是论文所用文件的早期版本
或编号细节不同），**未做任何数值修饰**。

---

## 三、已知边界

1. **λ₂ 必须是单根**。若 `λ₂` 有重数（如完全图 K4 的重数为 3），Fiedler 向量
   不唯一，`Δ^s` 的排序依赖选取，定理 3 的一阶近似会退化（实测 `Δλ₂ ≈ 0`）。
   论文用 `min_{j≤r}` 处理重根情形；本实现要求调用方先用
   `algebraic_connectivity` 检查重数。`test_degenerate_spectrum_breaks_first_order`
   把这个边界固化为反例测试。

2. **动力学层是线性时不变的**。`𝒜`、`ℬ` 不随时间变化，不支持时变参数或
   非线性结点动力学（如 Lorenz）。周进论文的仿真用 Lorenz，属于非线性
   同步问题，与本框架的线性可控性分析是两套数学工具。

3. **拓扑层与动力学层的结点下标口径不同**。拓扑层的驱动节点是
   `0..nnodes-1` 的整数（HAT 内部编号），动力学层是 `L` 的行序。两者都从 0
   开始，但语义不同，混用时需自行维护映射。

4. **`Gillespie` 式的连续时间控制未实现**。当前控制律是开环最小能量解，
   不含状态反馈（LQR / 观测器）。需要闭环时可用
   `scipy.linalg.solve_continuous_are` 自行求解 Riccati 方程。

5. **EoN 式的 `rng` 参数尚未贯通到控制层**。控制层的随机性仅出现在
   初值抽样（调用方自行提供），因此当前不需要。
