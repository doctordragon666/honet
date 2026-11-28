# 网络形态

除轻量的 `Hypergraph` 容器外，`honet` 还提供一套**形态明确**的类层次：
抽象基类 `HigherOrderNetwork` 定义公共接口（`nodes` / `edges` / `num_nodes` /
`num_edges` / `degree` / `incidence_matrix`），不可直接实例化；四个形态基类各自实现，
再分别派生 `W`（加权）、`D`（有向）、`WD`（加权有向）三个变体，共 **16 个具体类**。

```
HigherOrderNetwork (ABC，不可实例化)          honet/base/network.py
├── SimplicialComplexNetwork          单纯复形：按极大单形/面存储，子面自动向下封闭
│   ├── SimplicialComplexNetworkW     加权          honet/simplicial/network.py
│   ├── SimplicialComplexNetworkD     有向（单形顶点有序）
│   └── SimplicialComplexNetworkWD    加权有向
├── HypergraphNetwork                 超图：按超边存储，不要求向下封闭
│   ├── HypergraphNetworkW                         honet/hypergraph/network.py
│   ├── HypergraphNetworkD
│   └── HypergraphNetworkWD
├── MultilayerNetwork                 多层：按“层名 → 超边”存储，结点跨层共享
│   ├── MultilayerNetworkW                         honet/multilayer/network.py
│   ├── MultilayerNetworkD
│   └── MultilayerNetworkWD
└── TemporalNetwork                   时序：按 (时间戳, 事件) 存储，始终按时间有序
    ├── TemporalNetworkW                           honet/temporal/network.py
    ├── TemporalNetworkD
    └── TemporalNetworkWD
```

## 选型表

| 需求 | 类 |
| --- | --- |
| 通用超图容器（超边可带 `label` / `weight`） | `Hypergraph` |
| 超图网络（不向下封闭） | `HypergraphNetwork` |
| 单纯复形（`faces()` 自动展开子面） | `SimplicialComplexNetwork` |
| 结点跨层共享的分层结构 | `MultilayerNetwork` |
| 事件流（同一组结点可在不同时刻重复） | `TemporalNetwork` |
| 加权 / 有向 | 同名 `W` / `D` / `WD` 变体 |

```{important}
`Hypergraph` 与 `HypergraphNetwork` 是**不同类型**，按上表选型，不要互相替换。
`Hypergraph` 面向「一张高阶网络的容器」，`HypergraphNetwork` 是形态类层次的一员，
接收 hypernetx 风格的分析函数时通常需要后者。
```

## 加权与有向

加权与有向能力由 `WeightedMixin` / `DirectedMixin` 承载（`is_weighted` / `is_directed`
标记、权重默认 `1.0` 且必须为正、有向成员保留顺序并拒绝自环与重复结点），
但每个变体都是**显式声明的真实类**，没有运行时动态造类。

```python
import honet

# 加权有向超图：成员顺序即方向，权重表达强度
net = honet.HypergraphNetworkWD(edges=[["a", "b", "c"]])
net.add_edge(["c", "d"], weight=2.5)
print(net)                     # <HypergraphNetworkWD nodes=4 ... weighted=True directed=True>
print(net.weights)             # (1.0, 2.5)

# 多层网络：同一批结点在不同层有不同关系
ml = honet.MultilayerNetwork(layers=["work"], nodes=["a", "b", "c"])
ml.add_edge(["a", "b"], layer="work")
ml.add_edge(["a", "c"], layer="family")            # 新层自动注册
print(ml.degree("a"), ml.layer_degree("work", "a"))  # 2 1

# 时序网络：同一组结点在不同时间反复出现
tn = honet.TemporalNetwork(events=[(3.0, ["a", "b"]), (1.0, ["a"])])
print(tn.times, tn.active_nodes(1.0))              # (1.0, 3.0) frozenset({'a'})

# 单纯复形：一个三元面自动拥有全部子面
sc = honet.SimplicialComplexNetwork(facets=[["a", "b", "c"]])
print(sc.num_facets, sc.num_faces, sc.dimension)   # 1 7 2
```

## 必须确认的语义

- 无权变体（不带 `W`）给非 1.0 权重会抛 `TypeError`，别猜，改用 `W` 变体。
- 无向形态的成员是 `frozenset`，有向变体是**有序** `tuple`，且拒绝重复成员与自环。
- 单纯复形只加**极大面**，子面由 `faces()` 展开，重复添加会抛重复错误。
- 查询（`nodes` / `edges` / `neighbors` 等）返回不可变类型；修改只能通过
  `add_*` / `remove_*`。

构造顺序：先建对象，再 `add_edge`（单纯复形是 `add_facet`，多层是
`add_edge(..., layer=...)`，时序是 `add_event(..., time=...)`）。
