# 控制框架

`honet.control` 有两层能力，分别回答控制论的两个不同问题：

| 层 | 问题 | 依赖 | 主要入口 |
| --- | --- | --- | --- |
| 拓扑层 | **控制哪些结点** | HAT（`honet[hat]`） | `calc_drivers` / `detect_dilation_and_drivers` |
| 动力学层 | **怎么控制** | numpy / scipy | `HypergraphControlSystem` / `minimum_energy_control` |
| 单纯形层 | **牵制哪个单纯形** | numpy | `rank_simplices` / `pinning_matrix` |
| 稳态层 | **哪个结点是好的控制入口** | numpy / scipy | `average_controllability` / `optimal_control` |

拓扑层回答「控制哪些结点」，但不含动力学、耦合与控制律。后三层补齐这部分。

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

## 三、稳态层：网络控制入口与带约束最优控制

参考 [nctpy](https://github.com/BassettLab/nctpy)（Gu et al., *Nature
Communications* 6:8414, 2015 的实现库）。这一层回答「**哪个结点是好的控制
入口**」以及「**在状态轨迹约束下怎么控制**」。

### 矩阵归一化与稳定性

```python
from honet.control import matrix_normalization, is_stable

A_norm = matrix_normalization(A, "continuous")   # A/(c+λmax) - I
print(is_stable(A_norm, "continuous"))           # True
```

连续时间归一化保证稳定（谱实部为负），离散时间保证谱半径 < 1。这是后续所有
稳态量的前提——不稳定系统的无限时域 Gramian 不存在。

### 无限时域 Gramian

有限时域 Gramian 需要数值积分，无限时域则退化为解 Lyapunov 方程：

```
连续：A W + W Aᵀ + BBᵀ = 0      ⟺  W = ∫₀^∞ e^{As}BBᵀe^{Aᵀs}ds
离散：W - A W Aᵀ = BBᵀ          ⟺  W = Σₖ AᵏBBᵀ(Aᵀ)ᵏ
```

```python
from honet.control import infinite_gramian, gramian_auto

W = infinite_gramian(A_norm, np.eye(N), "continuous")
W = gramian_auto(A_norm, np.eye(N), None, "continuous")   # T=None 自动选无限
```

不稳定系统会抛 `ValueError`，而不是静默返回 `nan`（nctpy 是打印警告后返回 `nan`）。

### 可控性度量

| 指标 | 含义 | 用途 |
| --- | --- | --- |
| `average_controllability` | 该结点作为唯一入口时的平均能量 | 选「能量上划算」的入口 |
| `modal_controllability` | 驱动难控模态（λ 接近 ±1）的能力 | 选「能撬动慢模态」的入口 |
| `control_energy_ranking` | 按平均可控性排序 | 直接给出候选名单 |

```python
from honet.control import average_controllability, control_energy_ranking

ac = average_controllability(A_norm, "continuous")   # 每个结点的平均可控性
ranking = control_energy_ranking(A_norm)             # [(结点, 值), ...] 降序
```

解析验证：对角系统 `A = diag(a)` 有闭式解 `ac_i = -1/(2a_i)`、
`φ_i = 1 - a_i²`，测试直接对照这两个式子。

### 带状态轨迹约束的最优控制

与最小能量控制的区别：这里允许惩罚**状态轨迹偏离参考状态**，目标函数是

```
J = ∫₀ᵀ [ ρ‖u(t)‖² + ‖S(x(t) - x_r)‖² ] dt
```

求解用 Hamiltonian 状态-协态法。连续时间构造联合矩阵：

```
d/dt [x; λ] = M [x; λ] + c
M = [[A,        -BBᵀ/(2ρ)],
     [-2S,      -Aᵀ       ]]
c = [0; 2S x_r]
```

两端边值（`x(0)` 与 `x(T)` 都已知）先解出 `λ(0)`，再正向积分。

```python
from honet.control import optimal_control

result = optimal_control(
    A_norm, np.eye(N), x0, xf, T=1.0,
    rho=1.0, S=np.eye(N), xr="zero", system="continuous",
)
print(result.final_error, result.energy)
```

`rho` 越小，控制能量权重越低；`S` 越大，状态越贴合参考。`S=0` 时退化为
最小能量控制（与 `minimum_energy_control` 数值一致）。

离散时间走另一条路径：把所有时刻的约束拼成一个大型稀疏线性系统一次求解。

**与 nctpy 的数值对照**（`rho`/`S` 三组参数）：状态误差 ≤7.8e-16、
输入误差 ≤2.9e-15，达到机器精度。

### 前向仿真与批量控制任务

反解（`minimum_energy_control` / `optimal_control`）与正算
（`simulate_system`）方向相反，两者合起来构成开环控制的闭环自检：

```python
from honet.control import minimum_energy_control, simulate_system

solved = minimum_energy_control(A, B, x0, xf, T=2.0)
forward = simulate_system(A, B, x0, solved.inputs, dt=2.0 / 1000)
assert np.linalg.norm(forward.final_state - xf) < 1e-3   # 反解正确
```

`simulate_impulse` 是便捷封装，用于模拟「持续刺激某几个结点」：

```python
from honet.control import simulate_impulse

result = simulate_impulse(A, B, np.zeros(N), [1.0, 0, 0, 0], steps=8000)
# 稳态应趋近 -A⁻¹Bu
```

脑网络控制文献里的常见分析是**所有状态对之间的转移**（如 65 个脑区分成若干
功能系统，考察系统 A → B 的能量）。`control_energy_matrix` 一次算完：

```python
from honet.control import control_energy_matrix

result = control_energy_matrix(A_norm, np.eye(N), labels, T=1.0)
print(result.energies)            # (n_states, n_states)
print(result.hardest())           # 最难驱动的一对
print(result.as_dict())           # {(源, 目标): 能量}
```

`labels` 是长度 `N` 的标签序列（如 `["A","A","B","B","C","C"]`），
`expand_states` 可单独用于把标签展开成初态/终态矩阵。

**闭环一致性已验证**：反解输入经前向仿真后的终点误差 1.5e-04；批量能量与
逐条反解的相对差 2.4e-07。

### 能量积分、归一化与统计检验

**能量积分**用 Simpson 法则（稀疏控制集下 `u` 变化剧烈，梯形法误差偏大）：

```python
from honet.control import integrate_control_energy

energy = integrate_control_energy(U, dt=1e-3)   # 每个通道的 ∫u²dt
```

**最小能量的两种算法**：

| 函数 | 方法 | 精度 |
| --- | --- | --- |
| `minimum_energy_exact` | 精确 Gramian（块指数法） | 机器精度 |
| `minimum_energy_fast` | Simpson 型 Gramian 数值积分 | 相对误差 ~1e-4 |

`minimum_energy_fast` 在 `steps` 增大时收敛到精确解，测试验证了这一点。

**归一化**：

```python
from honet.control import normalize_state, normalize_weights

v = normalize_state(np.array([3.0, 4.0]))        # [0.6, 0.8]
w = normalize_weights(np.array([10, 20, 30, 40]))  # [1, 4/3, 5/3, 2]
```

`normalize_weights` 的 `add_constant` 是**整体 +1**（结果落在 `[1, 2]`），
不是缩放到 `(0, 1]`——后者会改变相对量级。这与 nctpy 一致（逐位验证误差 0）。

**显著性检验**：控制能量本身没有绝对标准，必须和零分布比较。

```python
from honet.control import null_p_value, fdr_p_values

p = null_p_value(observed, null_distribution)      # 加一平滑的经验 p 值
adjusted, rejected = fdr_p_values(p_values, alpha=0.05)   # BH 步进法
```

---

## 四、已知边界

1. **λ₂ 必须是单根**。若 `λ₂` 有重数（如完全图 K4 的重数为 3），Fiedler 向量
   不唯一，`Δ^s` 的排序依赖选取，定理 3 的一阶近似会退化（实测 `Δλ₂ ≈ 0`）。
   论文用 `min_{j≤r}` 处理重根情形；本实现要求调用方先用
   `algebraic_connectivity` 检查重数。`test_degenerate_spectrum_breaks_first_order`
   把这个边界固化为反例测试。

2. **动力学层默认线性时不变（LTI）**。`𝒜`、`ℬ` 为常数时不支持非线性结点
   动力学（如 Lorenz）——周进论文的仿真用 Lorenz，属于非线性同步问题，
   与本框架的线性可控性分析是两套数学工具。
   **线性时变（LTV）已支持**：`simulate_system` / `simulate_impulse` 的 `A`、`B`
   可传时间函数，另有 `ltv_gramian` / `ltv_controllability` 做时变可控性判定
   （见下方「时变系统」）。

3. **拓扑层与动力学层的结点下标口径不同**。拓扑层的驱动节点是
   `0..nnodes-1` 的整数（HAT 内部编号），动力学层是 `L` 的行序。两者都从 0
   开始，但语义不同，混用时需自行维护映射。

4. **`Gillespie` 式的连续时间控制未实现**。当前控制律是开环最小能量解，
   不含状态反馈（LQR / 观测器）。需要闭环时可用
   `scipy.linalg.solve_continuous_are` 自行求解 Riccati 方程。

5. **EoN 式的 `rng` 参数尚未贯通到控制层**。控制层的随机性仅出现在
   初值抽样（调用方自行提供），因此当前不需要。

6. **nctpy 的部分路径在当前 scipy 上不可用**。nctpy 的 `gramian` 用字符串比较
   版本号（`sp.__version__ < '1.6.0'`），在 scipy 1.18 上误判为旧版，去调用已被
   移除的 `sp.integrate.simps`，导致 `ave_control(..., system='continuous')` 与
   `gramian(..., T=有限)` 抛 `AttributeError`。本实现不受影响（用
   `scipy.linalg.expm` 与 Lyapunov 求解），连续平均可控性已用解析解验证。
   这是 nctpy 的 bug，不是移植错误。

7. **`optimal_control` 的 `S=0` 情形依赖 Gramian 伪逆**。状态-协态法在 `S=0`
   时协态方程与状态解耦，理论上 `E12` 会变奇异；实测本实现对给定算例仍精确
   （`E12` 未真正奇异）。若遇到 `E12` 奇异，会抛明确的 `ValueError` 提示增大
   `T` 或检查 `B`，而不是静默返回错误结果。

## 时变系统（LTV）

控制框架默认是线性时不变（LTI）的：`𝒜`、`ℬ` 为常数矩阵。但真实系统常常时变——
脑刺激实验中连接强度随疲劳变化、通信网络随负载改变、机械系统刚度随温度漂移。

### 前向仿真

`simulate_system` / `simulate_impulse` 的 `A`、`B` 可以传**接收时间返回矩阵**的函数：

```python
import numpy as np
from honet.control import simulate_system

# 耦合强度随时间增强
A = lambda t: np.array([[-1.0, 0.5 + 0.1 * t], [0.0, -2.0]])
B = np.eye(2)
result = simulate_system(A, B, [1.0, 1.0], np.zeros((100, 2)), dt=1e-2)
```

每段用该段起点时刻的 `A(t)` / `B(t)` 做**冻结系数**近似：`dt` 足够小时误差为
`O(dt²)`，与零阶保持的输入假设同阶。这是 LTV 数值仿真的标准做法，也是 `dt`
需要取小的原因。传常数矩阵时走原路径，不做任何额外求值。

### 可控性判定

**常数系统的 Kalman 秩判据对 LTV 不适用**：`[B, AB, A²B, …]` 只在 `A` 为常数时
才有意义。LTV 系统的可控性由**可控性 Gramian** 的秩决定：

```
W_c(t0, tf) = ∫ Φ(tf, τ) B(τ) B(τ)ᵀ Φ(tf, τ)ᵀ dτ
```

`W_c` 满秩即系统在 `[t0, tf]` 上完全可控。

```python
from honet.control import ltv_controllability

result = ltv_controllability(A, B, t0=0.0, t_final=5.0, steps=512)
result.controllable      # 是否完全可控
result.rank              # Gramian 的数值秩
result.condition_number  # 最大/最小奇异值之比；秩亏时为 inf
```

`ltv_gramian` 单独返回 Gramian，`ltv_controllability` 在其上做秩判定并返回
`LTVControllabilityResult`。Gramian 用中点法数值积分，状态转移矩阵按时间网格
递推 `Φ(t_{k+1}, t0) = e^{A(t_k)Δt} Φ(t_k, t0)`，与仿真的冻结系数口径一致。

**LTV 相对 LTI 的关键优势**：单个固定输入通道只能控制一个方向，但若 `B` 随时间
切换驱动不同通道，Gramian 仍可满秩。例如

```python
B_tv = lambda t: (np.array([[1.0], [0.0]]) if int(t) % 2 == 0
                  else np.array([[0.0], [1.0]]))
ltv_controllability(np.diag([-1.0, -2.0]), B_tv, t_final=10.0, steps=1001).rank  # 2
```

而固定 `B = [[1],[0]]` 的秩恒为 1。这一差异由测试固定。

### 边界

- 常数系统的 Kalman 秩判据与 LTV Gramian 的秩在 4 组用例上一致（测试保证），
  因此本模块也可用于常数系统的交叉验证。
- 时变参数在运行中途改变形状会抛 `ValueError`（报错含具体时刻与期望形状）。
- 反解类接口（`minimum_energy_control` / `optimal_control`）仍按 LTI 推导，
  不接受时变 `A` / `B`；LTV 的最优控制需要求解时变 Riccati 方程，尚未实现。
