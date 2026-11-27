#!/usr/bin/env python
"""综合实例：从高阶协作数据到"该干预谁"的完整分析流程。

这是一个**端到端**的例子，串起 honet 的主要能力，回答一个具体问题：

    一个 13 人的科研协作网络里，如果只能投入有限资源做一次干预
    （控制一个结点、或牵制一个三人合作组），应该选谁？

分析分六步，每步都说明「为什么这么做」而不只是「怎么做」：

1. **构造**：从长表建超图，检查结构
2. **拓扑画像**：谁是枢纽、谁是桥
3. **高阶结构**：高阶度分布、熵、可约性
4. **传播风险**：谁感染了最危险
5. **控制入口**：谁能最有效地改变系统
6. **结论**：综合多个指标给出建议

设计原则
--------
* **每步都可独立验证**：打印中间量，不靠"黑箱出结果"。
* **区分"看起来重要"与"实际可控"**：这是本例子最想传达的一点——
  中心性高不等于适合干预（枢纽结点可能已经饱和）。
* **固定随机种子**：所有随机过程都可复现。

运行::

    python examples/06_case_study_collaboration.py
"""

from __future__ import annotations

import numpy as np

import honet

# ---------------------------------------------------------------------------
# 全局设置
# ---------------------------------------------------------------------------

SEED = 2024
np.set_printoptions(precision=4, suppress=True)

# 图表中文字体（若没有 matplotlib 或不需要图，可忽略）
try:
    import matplotlib

    matplotlib.use("Agg")  # 无界面环境下也能出图
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    HAS_PLOT = True
except ImportError:  # pragma: no cover - 绘图是可选的
    HAS_PLOT = False


def section(title: str) -> None:
    """打印分节标题，便于阅读长输出。"""
    print()
    print("=" * 74)
    print(f"  {title}")
    print("=" * 74)


# ---------------------------------------------------------------------------
# 数据：13 人实验室、4 个研究组、跨组合作
# ---------------------------------------------------------------------------
#
# 为什么用硬编码数据而不是随机生成？
#   因为例子的价值在于「结论可被人工检验」。随机数据每次不同，
#   读者无法判断输出是否合理。这里的结构是刻意设计的：
#   - 4 个组各 3 人（组内紧密合作）
#   - 2 条跨组桥接（模拟跨学科项目）
#   - 1 个"孤岛"式小组，只通过 1 个人连到主体

#: 每条记录 = (成员列表, 项目名, 合作强度)
RECORDS: list[tuple[list[str], str, float]] = [
    # --- 组 A：数据科学 ---
    (["ann", "ben", "cy"], "A1", 3.0),
    (["ann", "ben"], "A2", 2.0),
    (["ann", "cy", "dee"], "A3", 2.5),
    # --- 组 B：实验物理 ---
    (["eve", "fay", "gus"], "B1", 3.0),
    (["eve", "gus"], "B2", 1.5),
    (["fay", "gus", "hal"], "B3", 2.0),
    # --- 组 C：理论 ---
    (["ivy", "jon", "kim"], "C1", 2.5),
    (["ivy", "jon"], "C2", 1.0),
    # --- 跨组桥接（关键！）---
    (["cy", "dee", "eve"], "X1", 2.0),  # A ↔ B
    (["gus", "hal", "ivy"], "X2", 1.5),  # B ↔ C
    # --- 孤岛小组：只通过 ivy 连到主体 ---
    (["ivy", "lee", "max"], "D1", 1.0),
]

print(f"数据: {len(RECORDS)} 个项目, 涉及 {len({n for r in RECORDS for n in r[0]})} 人")


# ---------------------------------------------------------------------------
# 第 1 步：构造超图
# ---------------------------------------------------------------------------
section("第 1 步 · 构造与结构检查")

# honet.Hypergraph 用于结构分析（incidence_matrix / degree_sequence 等）。
# 成员是 frozenset，所以顺序无关；label 用项目名，weight 用合作强度。
lab = honet.Hypergraph()
for members, label, weight in RECORDS:
    lab.add_edge(members, label=label, weight=weight)

print(f"规模: {lab.num_nodes} 个结点 / {lab.num_edges} 条超边")
print(f"是否均匀超图: {lab.is_uniform} | 各边大小: {lab.edge_sizes}")
print(f"结点: {lab.nodes}")

# 关联矩阵是最基础的矩阵化表示：行=结点、列=超边
incidence = honet.incidence_matrix(lab)
print(f"\n关联矩阵形状: {len(incidence)} × {len(incidence[0])}")

# 自检：列和应等于每条边的成员数，行和应等于该结点的高阶度。
# 这类断言比"看起来对"可靠得多，建议在真实项目里保留。
for j, edge in enumerate(lab.edges):
    assert sum(row[j] for row in incidence) == edge.size
for i, node in enumerate(lab.nodes):
    assert sum(incidence[i]) == lab.degree(node)
print("结构自检通过（列和 = 边大小，行和 = 高阶度）")


# ---------------------------------------------------------------------------
# 第 2 步：拓扑画像
# ---------------------------------------------------------------------------
section("第 2 步 · 拓扑画像：枢纽 vs 桥")

# 高阶度：参与几个项目
degrees = {node: lab.degree(node) for node in lab.nodes}

# 加权度：把合作强度也计入（参与次数 × 强度）
weighted = {node: 0.0 for node in lab.nodes}
for edge in lab.edges:
    for node in edge.members:
        weighted[node] += edge.weight

# 手算一个"桥接分数"：该结点的邻居之间是否互不相连。
# 高 = 它是不同组之间的唯一通道（结构洞）。
projection = honet.clique_projection(lab)
bridge_score = {}
for node in lab.nodes:
    neighbors = projection[node]
    if len(neighbors) < 2:
        bridge_score[node] = 0.0
        continue
    # 邻居对中有多少「本身不是邻居」
    pairs = [(a, b) for i, a in enumerate(sorted(neighbors)) for b in sorted(neighbors)[i + 1 :]]
    disconnected = sum(1 for a, b in pairs if b not in projection[a])
    bridge_score[node] = disconnected / len(pairs)

print(f"{'成员':<6}{'高阶度':>7}{'加权度':>8}{'邻居数':>7}{'桥接分数':>10}")
print("-" * 40)
for node in sorted(lab.nodes, key=lambda n: -weighted[n]):
    print(
        f"{node:<6}{degrees[node]:>7}{weighted[node]:>8.1f}"
        f"{len(projection[node]):>7}{bridge_score[node]:>10.3f}"
    )

# 关键观察：加权度最高的未必是桥。这两个指标回答不同问题：
#   - 加权度：谁是"最活跃"的（资源多）
#   - 桥接分数：谁是"不可替代"的（位置关键）
hub = max(weighted, key=weighted.__getitem__)
bridge = max(bridge_score, key=bridge_score.__getitem__)
print(f"\n最活跃（加权度最高）: {hub} ({weighted[hub]:.1f})")
print(f"最不可替代（桥接最高）: {bridge} ({bridge_score[bridge]:.3f})")
print(f"两者是否同一人: {hub == bridge}")


# ---------------------------------------------------------------------------
# 第 3 步：高阶结构指标
# ---------------------------------------------------------------------------
section("第 3 步 · 高阶结构：分布与异质性")

# 注意：honet.metrics 收 hypernetx 超图，不收 honet.Hypergraph。
# 需要转换——这里手动重建（honet 暂无官方 hypernetx -> Hypergraph 转换器）。
import hypernetx as hnx  # noqa: E402 - 可选依赖，放在用到处导入更清晰

H = hnx.Hypergraph({label: members for members, label, _weight in RECORDS})
print(f"转换为 hypernetx 超图: {H.shape}")

# 每种超边大小下的平均度：看"小组合作"和"大组合作"哪个更普遍
by_size = honet.metrics.get_avg_degree_by_edge_size(H)
print("\n各超边大小下的平均度:")
for size in sorted(by_size):
    print(f"  m={size}: {by_size[size]:.3f}")

# 度分布熵：越高越"平等"
entropy = honet.metrics.calc_entropy(H)
print(f"\n度分布熵: {entropy:.4f}")
print("  （越高表示合作机会越分散，越低表示越集中在少数人）")

# 可约性：能否用更简单的结构近似。接近 1 说明高阶结构信息量低
reducibility = honet.metrics.calc_reducibility(H)
print(f"可约性: {reducibility:.4f}")
print("  （越接近 1 越说明可以用成对网络近似；越低越说明高阶结构不可忽略）")

# 自然连通度：抗打击能力
natural = honet.metrics.calc_natural_connectivity(H)
print(f"自然连通度: {natural:.4f}")


# ---------------------------------------------------------------------------
# 第 4 步：传播风险
# ---------------------------------------------------------------------------
section("第 4 步 · 传播风险：谁感染了最危险")

# 为什么用 SIS 而不是 SIR？
#   科研协作的"信息传播"是可以反复发生的（一个想法会被反复讨论），
#   不像疾病那样"康复后免疫"。SIS 更贴合这个场景。
#
# tau 是 {边大小: 感染率}：3 人会议比 2 人讨论更容易传播信息。
TAU = {2: 0.15, 3: 0.35}
GAMMA = 0.25  # "遗忘率"：讨论过的想法多久被淡忘

# 对每个人单独做"从你开始"的模拟，看最终波及多少人。
# 为什么跑多次？因为传播是随机的，单次结果不可靠。
#
# 关键：必须用 initial_infecteds 显式指定起点。若用 rho（初始感染比例），
# 每次模拟会**随机**挑人，所有人算出来都一样，这一步就失去意义了。
N_TRIALS = 15
risk = {}
for node in sorted(lab.nodes):
    finals = []
    for trial in range(N_TRIALS):
        _t, _s, infected = honet.contagion.random_discrete_sis(
            H,
            tau=TAU,
            gamma=GAMMA,
            initial_infecteds={node},  # 明确从这一个人开始（此时 rho 自动失效）
            tmax=80,
            dt=1.0,
            seed=SEED + trial,
        )
        finals.append(infected[-1])
    risk[node] = sum(finals) / len(finals)

print(f"从每个人出发的最终波及人数（{N_TRIALS} 次平均）:")
for node in sorted(risk, key=lambda n: -risk[n]):
    bar = "#" * int(risk[node] * 4)
    print(f"  {node:<6}{risk[node]:>6.2f}  {bar}")

spreader = max(risk, key=risk.__getitem__)
print(f"\n最强传播源: {spreader}（平均波及 {risk[spreader]:.2f} 人）")


# ---------------------------------------------------------------------------
# 第 5 步：控制入口分析
# ---------------------------------------------------------------------------
section("第 5 步 · 控制入口：谁能最有效改变系统")

# 把超图结构转成动力学：用 2 阶 Laplacian 作耦合矩阵。
# 为什么用 Laplacian 而不是邻接矩阵？
#   因为 Laplacian 描述"差异驱动的扩散"，是网络动力学的标准选择。
laplacian = honet.control.hypergraph_laplacian(
    {label: members for members, label, _weight in RECORDS},
    num_nodes=lab.num_nodes,
    order=2,
)

# 单结点内部动力学取 -1（自然衰减），这是最小假设：
# 不做控制时，每个人的"状态"会慢慢回到基线。
node_index = {node: i for i, node in enumerate(lab.nodes)}
n_nodes = lab.num_nodes

system = honet.control.HypergraphControlSystem(
    A=np.array([[-1.0]]),
    L=laplacian,
    Lambda=np.array([[1.0]]),
    B=np.array([[1.0]]),
    coupling=1.0,
    order=2,
)
A_sys = system.state_matrix()

# 可控性：单点控制 vs 全点控制
#   这是本例子最想强调的一点——**能不能控制**和**控制效果好不好**是两件事。
print("可控性对比:")
B_single = np.zeros((n_nodes, 1))
B_single[0, 0] = 1.0
rank_single = honet.control.kalman_rank(A_sys, B_single)
rank_full = honet.control.kalman_rank(A_sys, np.eye(n_nodes))
print(f"  只控制 1 个结点 -> Kalman 秩 = {rank_single} / {n_nodes}")
print(f"  全部结点受控   -> Kalman 秩 = {rank_full} / {n_nodes}")

# 平均可控性：把每个结点分别作为唯一入口，看驱动整个系统需要多少能量。
# 数值越大 = 越省力 = 越好的控制入口。
# 注意：需要先归一化保证稳定，否则无限时域 Gramian 不存在。
A_stable = honet.control.matrix_normalization(A_sys - np.eye(n_nodes) * 0.0, "continuous")
controllability = honet.control.average_controllability(A_stable, "continuous")

print("\n每个结点作为唯一控制入口的平均可控性:")
ranking = sorted(range(n_nodes), key=lambda i: -controllability[i])
node_list = list(lab.nodes)
for rank, idx in enumerate(ranking, 1):
    print(f"  {rank:2d}. {node_list[idx]:<6} {controllability[idx]:>8.4f}")

best_control = node_list[ranking[0]]
print(f"\n最佳控制入口: {best_control}")

# ⚠️ 重要观察：留意上表的**并列**。
#   平均可控性只反映"能量代价"，不反映"位置重要性"。在这张图里，
#   若干叶子结点（只连一个组的边缘成员）算出了几乎相同的值——因为
#   它们在 Gramian 里结构等价。这意味着：
#     * 平均可控性可以用来排除"明显难控"的结点，
#     * 但**不能**单独用来在等价结点之间做选择；
#     * 必须配合结构指标（桥接分数、传播风险）一起看。
#
# 用相对容差 1e-4 分组（绝对容差 1e-9 太严，会漏掉浮点级差异）。
tolerance = 1e-4 * abs(controllability[ranking[0]])
groups: list[list[str]] = []
for i in ranking:
    if groups and abs(controllability[i] - controllability[ranking[0]]) <= tolerance:
        groups[-1].append(node_list[i])
    else:
        if groups:
            break
        groups.append([node_list[i]])
tied = groups[0] if groups else [best_control]
print(f"\n可控性最高的一组（容差 {tolerance:.2e}）: {tied}")
if len(tied) > 1:
    print("  -> 这些结点在平均可控性下不可区分，需要其它指标打破平局")
else:
    print("  -> 最高值唯一")


# ---------------------------------------------------------------------------
# 第 6 步：综合结论
# ---------------------------------------------------------------------------
section("第 6 步 · 综合结论")

# 把四个视角的排名放在一起。为什么需要多个视角？
#   因为不同指标回答不同问题，单一指标会误导。
#   例如：传播最强的人未必是最佳控制入口（他可能已经在饱和区）。
print(f"{'成员':<6}{'加权度':>9}{'桥接':>9}{'传播风险':>10}{'可控性':>10}")
print("-" * 46)
for node in sorted(lab.nodes, key=lambda n: -controllability[node_index[n]]):
    print(
        f"{node:<6}{weighted[node]:>9.1f}{bridge_score[node]:>9.3f}"
        f"{risk[node]:>10.2f}{controllability[node_index[node]]:>10.4f}"
    )

print()
print("关键发现:")
print(f"  1. 最活跃的是 {hub}，但他不一定是干预重点")
print(f"  2. 位置最不可替代的是 {bridge}（桥接分数 {bridge_score[bridge]:.3f}）")
print(f"  3. 传播风险最高的是 {spreader}（平均波及 {risk[spreader]:.2f} 人）")
print(f"  4. 最佳控制入口是 {best_control}")

# 综合建议：如果只能选一个人做干预，优先考虑"桥 + 高传播"的人。
# 理由：桥决定了信息能否跨组流动；高传播决定了单点干预的杠杆率。
combined = {node: bridge_score[node] * risk[node] / max(risk.values()) for node in lab.nodes}
priority = max(combined, key=combined.__getitem__)

print(f"\n综合优先级（桥接 × 传播风险）最高: {priority}")
print(f"  -> 如果只有一次干预机会，建议选 {priority}")
print()
print("注意：这是启发式组合，不是严格最优解。真实决策还需考虑成本、")
print("      人员意愿等本模型之外的因素。")


# ---------------------------------------------------------------------------
# 可选：把关键结果画出来
# ---------------------------------------------------------------------------
if HAS_PLOT:
    section("附图 · 四个指标对比")

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    names = list(lab.nodes)

    specs = [
        ("加权度", [weighted[n] for n in names], "tab:blue"),
        ("桥接分数", [bridge_score[n] for n in names], "tab:orange"),
        ("传播风险", [risk[n] for n in names], "tab:red"),
        ("可控性", [controllability[node_index[n]] for n in names], "tab:green"),
    ]

    for ax, (title, values, color) in zip(axes, specs, strict=True):
        ax.bar(names, values, color=color, alpha=0.75)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=90)
        ax.grid(alpha=0.25, axis="y")

    fig.suptitle("13 人协作网络：四个视角的结点重要性", fontsize=12)
    plt.tight_layout()

    output = "examples/06_case_study_result.png"
    fig.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存: {output}")

print()
print("=" * 74)
print("  分析完成")
print("=" * 74)
