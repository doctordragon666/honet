# 传播模型的扩展性

本文说明 `honet.contagion` 的仓室模型设计：如何新增模型、为什么不需要改引擎，
以及当前已知的边界。

## 1. 核心思路：用规格替代布尔分叉

旧实现用 `model: str` 加大量 `is_sir = model == "SIR"` 判断来区分模型：

| 文件 | `is_sir` 出现次数 |
| --- | ---: |
| `_discrete.py` | 9 |
| `_random_models.py` | 7 |

每加一个模型（SIRS、SIER……）都要在所有分叉点补一个分支，改一处漏一处。
`hon_sis` 与 `hon_sir` 两个入口更有 95% 的重复代码。

现在模型被抽象成**声明式规格**（`honet.contagion._model.CompartmentModel`）：

```python
from honet.contagion import CompartmentModel, register_model, simulate

SIER = CompartmentModel(
    name="SIER",
    states=("S", "E", "I", "R"),          # 有哪些仓室
    infected=("I",),                       # 哪些仓室有传染性
    on_infection={"S": "E", "E": "I"},     # 被感染时迁到哪
    on_recovery={"I": "R"},                # 恢复时迁到哪
)
```

引擎只读规格，不再判断模型名。**新增模型 = 注册一份规格**。

## 2. 新增一个模型要写多少代码

以内置的 SIER 为例，全部代码就是上面那 6 行声明。不需要碰
`_discrete.py` / `_random_models.py` / `_simulate.py`。

自定义模型同理（测试里用五仓室 `SIQD` 验证过）：

```python
from honet.contagion import CompartmentModel, register_model, simulate

register_model(CompartmentModel(
    name="SIQD",
    states=("S", "I", "Q", "D"),
    infected=("I",),
    on_infection={"S": "I"},
    on_recovery={"I": "Q", "Q": "D", "D": "S"},
))

result = simulate(H, gamma=0.3, tau={2: 0.5}, model="SIQD", tmax=10)
```

离散与随机两条路径都能直接跑，无需改任何引擎代码
（`tests/test_contagion_models.py::TestNewModelsRunWithoutEngineChanges`）。

## 3. 内置模型

| 模型 | 仓室 | 转移 | 类型 |
| --- | --- | --- | --- |
| `SIS` | S, I | I→S | 循环（可反复感染） |
| `SIR` | S, I, R | I→R | 吸收（R 永久免疫） |
| `SIRS` | S, I, R | I→R, R→S | 循环（免疫失效） |
| `SIER` | S, E, I, R | S→E, E→I, I→R | 吸收（含潜伏期） |
| `SEIR` | S, E, I, R | 同 SIER | 吸收 |
| `SI` | S, I | 无恢复 | 吸收 |

`is_sir_like` 用**可达性分析**判断是否存在吸收态，而不是朴素地看「恢复目标
能否被感染」。区别体现在 SIRS：它的 R 本身不可被感染，但经 R→S 能回到易感，
所以整体仍是循环型。

## 4. 复用性：显式随机源

旧实现把随机源写死成全局状态（`random.random()` / `np.random.*`），因此：

- 无法在同一进程里并行跑多个模拟；
- 嵌套调用会互相污染随机序列；
- 单测必须靠 `seed_everything` 改全局状态才能复现。

现在引擎接受 `rng` 参数（借鉴 EoN 的 `fast_SIR(..., rng=...)` 设计）：

```python
from honet.contagion._common import make_rng
from honet.contagion import simulate

py_rng, np_rng = make_rng(42)
result = simulate(H, 0.3, {2: 0.5}, model="SIS", tmax=10, rng=py_rng)
```

为 `None` 时退回全局状态，旧行为与旧结果完全不变。

## 5. 关于借鉴 EoN

EoN（Epidemics on Networks）是成对网络上的经典实现，公开 API 有 113 个。
核实后的结论：

**已借鉴**
- 显式 `rng` 参数化（本仓库已实现）
- `return_full_data` 返回完整迁移事件（本仓库已有同等能力）

**已具备同等能力**
- SIS/SIR 的 Gillespie 与离散时间实现
- 初始条件参数族（`initial_infecteds` / `initial_recovereds` / `rho`）

**不引入**
- EoN 面向**成对网络**（`networkx.Graph`），honet 面向**超图**，超边内的
  传播语义不同，其平均场 / 对近似公式不能直接搬用
- 引入 EoN 会破坏 `dependencies = []` 的零依赖承诺（探测用的 EoN 已卸载）

## 6. 已知边界

1. **Gillespie 方法只支持 SIS / SIR**。hypernetx 只内置这两个的连续时间实现；
   其它模型请用 `Discrete` 或 `RandomGillespie`，传错会得到明确的 `ValueError`
   而不是静默错误结果。
2. **确定性模型仍是静态参数**。`RandomDiscrete` / `RandomGillespie` 的 `tau` 与
   `gamma` 支持传时间函数（见下方「时变参数」），但 `hon_sis` / `hon_sir` 等
   确定性（平均场）实现仍只接受常数参数；需要时变请用随机模型，或按时间段
   分段调用。
3. **`E` 仓室当前按「可被感染」处理**。SIER 里 S→E、E→I 都走感染力判定，
   这与部分文献把潜伏期设为固定时长的做法不同；如需固定时长，应扩展规格增加
   「停留时长」字段。
4. **`initial_recovereds` 在 SIS 下的语义特殊**：它表示「受控结点」而非初始
   恢复者（SIS 没有 R 仓室）。这是为兼容旧实现而保留的行为，已在
   `_discrete.py` 的模块 docstring 中注明。

## 7. 时变参数

`random_discrete_sis` / `random_discrete_sir` / `random_gillespie_sis` /
`random_gillespie_sir` 的 `tau` 与 `gamma` 除了常数，还可以传**接收时间返回取值**
的函数，模拟过程每步用当前时刻求值：

```python
from honet import random_discrete_sis

# 季节性传播：感染力随时间正弦起伏
def seasonal(t):
    return {2: 0.4 * (1 + 0.8 * math.sin(2 * math.pi * t / 30))}

# 政策干预：t=20 之后恢复率翻倍
def policy(t):
    return 0.1 if t < 20 else 0.25

t, S, I = random_discrete_sis(H, seasonal, policy, rho=0.1, tmax=120, seed=0)
```

语义要点：

- **离散时间**（`random_discrete_*`）每步在 `t = tmin, tmin+dt, …` 处求值，
  `tmax=10, dt=1` 共求值 11 次；
- **连续时间**（`random_gillespie_*`）事件时刻不固定，每轮事件按当前时刻求值；
- 传常数时走快路径，不做任何额外求值，结果与旧版逐位一致；
- `tau` 也可以直接传数值，等价于 `{2: 该值}`（只有成对交互）。
- 确定性模型（`hon_sis` / `hon_sir`）仍只接受常数参数。
