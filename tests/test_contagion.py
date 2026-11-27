"""honet.contagion 子包的测试。

全部用例都在 3~4 个结点、2~3 条超边的小图上真实调用公开函数，并给随机
过程显式传 ``seed``，保证可复现且跑得快。
"""

from __future__ import annotations

import hypernetx as hnx
import numpy as np
import pytest

from honet import contagion
from honet.contagion import (
    Contagion_Method,
    classify_point,
    discrete_sir_control,
    discrete_sis_control,
    discrete_sis_control_pulse,
    generate_tau,
    hon_sir,
    hon_sis,
    is_stable,
    random_discrete_sir,
    random_discrete_sis,
    random_gillespie_sir,
    random_gillespie_sis,
    sis_phase,
)

#: 4 个结点、3 条超边（含一条三元超边），足够区分 SIS / SIR 与不同策略
TAU = {2: 0.5, 3: 0.5}


@pytest.fixture
def H() -> hnx.Hypergraph:
    """构造测试用的小超图：a-b、b-c-d、c-d。"""
    return hnx.Hypergraph(
        {
            "e1": ["a", "b"],
            "e2": ["b", "c", "d"],
            "e3": ["c", "d"],
        }
    )


def test_public_api_is_lazy_and_complete() -> None:
    """子包应导出全部新名字与旧名别名，且别名指向同一个对象。"""
    assert "hon_sis" in contagion.__all__
    assert "honSIS" in contagion.__all__
    # 旧名别名与新名是同一个函数对象
    assert contagion.honSIS is contagion.hon_sis
    assert contagion.honSIR is contagion.hon_sir
    assert contagion.ContagionMethod is contagion.Contagion_Method
    assert contagion.discrete_SIS_control is contagion.discrete_sis_control
    assert contagion.RandomGillespie_SIR is contagion.random_gillespie_sir


def test_contagion_method_enum() -> None:
    """枚举应有 4 个成员，且取值互不相同。"""
    members = list(Contagion_Method)
    assert len(members) == 4
    assert {m.name for m in members} == {
        "Gillespie",
        "RandomGillespie",
        "Discrete",
        "RandomDiscrete",
    }
    assert len({m.value for m in members}) == 4


def test_generate_tau_all_types(H: hnx.Hypergraph) -> None:
    """六种生成方式都应覆盖图中出现过的全部超边大小。"""
    assert generate_tau(H, [0.3], type="fixed") == {2: 0.3, 3: 0.3}
    assert generate_tau(H, [0.2, 0.4], type="fixed_compare") == {2: 0.2, 3: 0.4}
    sized = generate_tau(H, [0.1], type="size_based")
    assert sized[2] == pytest.approx(0.2)
    assert sized[3] == pytest.approx(0.3)
    compared_sized = generate_tau(H, [0.2, 0.1], type="size_based_compare")
    assert compared_sized[2] == pytest.approx(0.2)
    assert compared_sized[3] == pytest.approx(0.3)
    # sum_based：各不同大小的 tau 之和等于 beta[0]
    summed = generate_tau(H, [0.5], type="sum_based")
    assert set(summed) == {2, 3}
    assert summed[2] == pytest.approx(0.5 * 2 / 5)
    assert summed[3] == pytest.approx(0.5 * 3 / 5)
    assert sum(summed.values()) == pytest.approx(0.5)
    compared = generate_tau(H, [0.9, 0.5], type="sum_based_compare")
    assert compared[2] == 0.9


def test_generate_tau_rejects_bad_input(H: hnx.Hypergraph) -> None:
    """空 beta 与未知 type 都应报错。"""
    with pytest.raises(ValueError, match="beta"):
        generate_tau(H, [])
    with pytest.raises(ValueError, match="type"):
        generate_tau(H, [0.1], type="nope")


def test_is_stable_responds_to_infection_rate(H: hnx.Hypergraph) -> None:
    """感染率足够低时稳定，足够高时不稳定。"""
    assert is_stable(H, {2: 0.01, 3: 0.01}, gamma=0.9) is True
    assert is_stable(H, {2: 5.0, 3: 5.0}, gamma=0.01) is False


def test_is_stable_without_size_two_edges() -> None:
    """图中没有大小为 2 的超边时按旧实现口径直接返回 True。"""
    H = hnx.Hypergraph({"e1": ["a", "b", "c"]})
    assert is_stable(H, {3: 9.0}, gamma=0.1) is True


def test_classify_point_and_custom_function(H: hnx.Hypergraph) -> None:
    """相图单点分类应与 is_stable 的结果一致，并支持自定义 tau 函数。"""
    low = classify_point(0.01, 0.01, 0.9, H, "fixed_compare")
    high = classify_point(5.0, 5.0, 0.01, H, "fixed_compare")
    assert low is True
    assert high is False

    calls: list[dict] = []

    def custom(graph, beta, type):  # noqa: A002 - 与旧签名保持一致
        calls.append({"beta": beta, "type": type})
        return {2: 0.01, 3: 0.01}

    assert classify_point(0.1, 0.2, 0.9, H, "fixed_compare", function=custom) is True
    assert calls == [{"beta": [0.1, 0.2], "type": "fixed_compare"}]


def test_sis_phase_matrix_shape_and_values(H: hnx.Hypergraph) -> None:
    """相图矩阵形状为 (len(beta_higher), len(beta2))，元素只取 0/1。"""
    phase = sis_phase(H, 0.9, [0.01, 0.05], [0.01, 0.5, 5.0], kind="fixed_compare")
    assert phase.shape == (3, 2)
    assert phase.dtype == np.dtype(int)
    assert set(np.unique(phase)) <= {0, 1}


def test_sis_phase_empty_input(H: hnx.Hypergraph) -> None:
    """空参数序列返回形状正确的空矩阵，不抛异常。"""
    assert sis_phase(H, 0.5, [], []).shape == (0, 0)


def test_discrete_sis_control_shapes_and_ranges(H: hnx.Hypergraph) -> None:
    """离散 SIS 返回三个等长数组，易感者与感染者之和守恒且取值非负。"""
    t, n_sus, n_inf = discrete_sis_control(H, TAU, 0.3, tmax=8, rho=0.25, seed=1)
    assert len(t) == len(n_sus) == len(n_inf)
    assert len(t) >= 2
    assert t[0] == 0
    assert np.all(n_sus >= 0) and np.all(n_inf >= 0)
    assert np.all(n_sus + n_inf == len(H.nodes))
    # 初始感染者数量与 rho 抽样一致
    assert n_inf[0] == 1


def test_discrete_sis_control_full_data(H: hnx.Hypergraph) -> None:
    """return_full_data=True 返回以时刻为键的迁移事件字典。"""
    events = discrete_sis_control(H, TAU, 0.3, tmax=3, rho=0.5, seed=2, return_full_data=True)
    assert isinstance(events, dict)
    assert 0 in events
    assert isinstance(events[0], list)


def test_discrete_sis_control_pulse(H: hnx.Hypergraph) -> None:
    """脉冲控制同样满足人数守恒，且与普通控制给出不同轨迹。"""
    t, n_sus, n_inf = discrete_sis_control_pulse(H, TAU, 0.3, tmax=8, rho=0.25, seed=1)
    assert len(t) == len(n_sus) == len(n_inf)
    assert np.all(n_sus + n_inf == len(H.nodes))
    plain = discrete_sis_control(H, TAU, 0.3, tmax=8, rho=0.25, seed=1)
    assert not np.array_equal(n_sus, plain[1])


def test_discrete_sis_control_target_strategy(H: hnx.Hypergraph) -> None:
    """strategy='target' 走目标控制分支，受控结点被强制恢复。"""
    t, n_sus, n_inf = discrete_sis_control(
        H,
        TAU,
        0.3,
        "target",
        tmax=6,
        rho=0.25,
        initial_recovereds=["a"],
        seed=3,
    )
    assert len(t) == len(n_sus) == len(n_inf)
    assert np.all(n_sus + n_inf == len(H.nodes))


def test_discrete_sir_control_shapes_and_ranges(H: hnx.Hypergraph) -> None:
    """离散 SIR 返回四个等长数组，三类人数之和守恒。"""
    t, n_sus, n_inf, n_rec = discrete_sir_control(H, TAU, 0.3, tmax=8, rho=0.25, seed=1)
    assert len(t) == len(n_sus) == len(n_inf) == len(n_rec)
    assert np.all(n_sus + n_inf + n_rec == len(H.nodes))
    # SIR 的康复人数单调不减（不会从 R 回到其他状态）
    assert np.all(np.diff(n_rec) >= 0)


def test_discrete_sir_control_recovereds_are_controlled(H: hnx.Hypergraph) -> None:
    """受控结点初始就计入康复人数。"""
    t, n_sus, n_inf, n_rec = discrete_sir_control(
        H,
        TAU,
        0.3,
        tmax=6,
        rho=0.25,
        initial_recovereds=["a"],
        seed=4,
    )
    assert n_rec[0] == 1
    assert np.all(n_sus + n_inf + n_rec == len(H.nodes))


@pytest.mark.parametrize(
    ("model", "expected_columns"),
    [(random_discrete_sis, 3), (random_discrete_sir, 4)],
)
def test_random_discrete_models(model, expected_columns: int, H: hnx.Hypergraph) -> None:
    """随机离散模型的返回值列数与取值范围合理。"""
    result = model(H, TAU, 0.3, rho=0.25, tmax=5, seed=1)
    assert len(result) == expected_columns
    t, n_sus, n_inf = result[0], result[1], result[2]
    assert len(t) == len(n_sus) == len(n_inf)
    assert t[0] == 0
    assert all(0 <= value <= len(H.nodes) for value in n_sus)
    assert all(0 <= value <= len(H.nodes) for value in n_inf)
    if expected_columns == 4:
        n_rec = result[3]
        assert all(s + i + r == len(H.nodes) for s, i, r in zip(n_sus, n_inf, n_rec, strict=True))


@pytest.mark.parametrize(
    ("model", "expected_columns"),
    [(random_gillespie_sis, 3), (random_gillespie_sir, 4)],
)
def test_random_gillespie_models(model, expected_columns: int, H: hnx.Hypergraph) -> None:
    """随机 Gillespie 模型的返回值列数与取值范围合理。"""
    result = model(H, TAU, 0.3, rho=0.25, tmax=5, seed=1)
    assert len(result) == expected_columns
    t, n_sus, n_inf = result[0], result[1], result[2]
    assert len(t) == len(n_sus) == len(n_inf)
    assert t[0] == 0
    assert all(0 <= value <= len(H.nodes) for value in n_sus)
    assert all(0 <= value <= len(H.nodes) for value in n_inf)
    # 连续时间模型的时间戳应单调不减
    assert all(later >= earlier for earlier, later in zip(t, t[1:], strict=False))


@pytest.mark.parametrize(
    "model",
    [random_discrete_sis, random_discrete_sir, random_gillespie_sis, random_gillespie_sir],
)
def test_random_models_are_reproducible(model, H: hnx.Hypergraph) -> None:
    """同一 seed 下两次调用结果完全相同。"""
    first = model(H, TAU, 0.3, rho=0.5, tmax=4, seed=11)
    second = model(H, TAU, 0.3, rho=0.5, tmax=4, seed=11)
    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        assert left == right


@pytest.mark.parametrize("method", list(Contagion_Method))
def test_hon_sis_all_methods(method: Contagion_Method, H: hnx.Hypergraph) -> None:
    """SIS 入口的四种方法都能跑通并返回三个序列。"""
    result = hon_sis(H, 0.3, TAU, method=method, tmax=5, seed=1)
    assert len(result) == 3
    t, n_sus, n_inf = result
    assert len(t) == len(n_sus) == len(n_inf)
    assert len(t) >= 1


@pytest.mark.parametrize("method", list(Contagion_Method))
def test_hon_sir_all_methods(method: Contagion_Method, H: hnx.Hypergraph) -> None:
    """SIR 入口的四种方法都能跑通并返回四个序列。"""
    result = hon_sir(H, 0.3, TAU, method=method, tmax=5, seed=1)
    assert len(result) == 4
    t, n_sus, n_inf, n_rec = result
    assert len(t) == len(n_sus) == len(n_inf) == len(n_rec)
    assert len(t) >= 1


def test_hon_sis_with_control_nodes(H: hnx.Hypergraph) -> None:
    """给 control_nodes 时走控制分支。"""
    t, n_sus, n_inf = hon_sis(H, 0.3, TAU, tmax=6, seed=1, control_nodes=["a"])
    assert len(t) == len(n_sus) == len(n_inf)
    assert np.all(n_sus + n_inf == len(H.nodes))


def test_hon_sir_with_recovereds(H: hnx.Hypergraph) -> None:
    """给 initial_recovereds 时走 SIR 控制分支。"""
    t, n_sus, n_inf, n_rec = hon_sir(H, 0.3, TAU, tmax=6, seed=1, initial_recovereds=["a"])
    assert len(t) == len(n_sus) == len(n_inf) == len(n_rec)
    assert np.all(n_sus + n_inf + n_rec == len(H.nodes))


def test_hon_sis_with_explicit_initial_infection(H: hnx.Hypergraph) -> None:
    """显式给定初始感染者时，初始感染人数等于给定数量。"""
    t, n_sus, n_inf = hon_sis(H, 0.3, TAU, tmax=4, seed=1, initial_infection=["a", "b"])
    assert n_inf[0] == 2
    assert n_sus[0] == len(H.nodes) - 2


def test_hon_sis_rejects_unknown_method(H: hnx.Hypergraph) -> None:
    """未知方法应抛 ValueError。"""
    with pytest.raises(ValueError, match="传播方法"):
        hon_sis(H, 0.3, TAU, method="nope", tmax=3)


def test_hon_sir_rejects_unknown_method(H: hnx.Hypergraph) -> None:
    """未知方法应抛 ValueError。"""
    with pytest.raises(ValueError, match="传播方法"):
        hon_sir(H, 0.3, TAU, method="nope", tmax=3)


def test_rho_and_initial_infecteds_conflict(H: hnx.Hypergraph) -> None:
    """rho 与 initial_infecteds 同时给出应报错。"""
    with pytest.raises(ValueError, match="不能同时指定"):
        discrete_sis_control(H, TAU, 0.3, rho=0.5, initial_infecteds=["a"], tmax=3, seed=1)


def test_strategy_names_are_supported(H: hnx.Hypergraph) -> None:
    """所有策略名都能被解析并跑通（用会真正解析策略的 pulse 版本）。"""
    strategies = ("individual", "collective", "threshold", "default", "majority_vote")
    for strategy in strategies:
        t, n_sus, n_inf = discrete_sis_control_pulse(
            H, TAU, 0.3, strategy, tmax=4, rho=0.25, seed=1
        )
        assert len(t) == len(n_sus) == len(n_inf)


def test_unknown_strategy_raises(H: hnx.Hypergraph) -> None:
    """未知策略名应抛 ValueError。

    注：``discrete_sis_control`` 在非 target 分支会无视传入策略、一律用
    ``collective_contagion``（与旧实现一致），因此这里用会真正解析策略的
    ``discrete_sis_control_pulse`` 来验证报错路径。
    """
    with pytest.raises(ValueError, match="传播策略"):
        discrete_sis_control_pulse(H, TAU, 0.3, "nope", tmax=3, rho=0.25, seed=1)


def test_discrete_sis_control_ignores_strategy_like_old_impl(H: hnx.Hypergraph) -> None:
    """旧实现的 ``discrete_SIS_control`` 非 target 分支固定用 collective 传播。"""
    a = discrete_sis_control(H, TAU, 0.3, "individual", tmax=5, rho=0.25, seed=5)
    b = discrete_sis_control(H, TAU, 0.3, "collective", tmax=5, rho=0.25, seed=5)
    assert np.array_equal(a[2], b[2])


def test_direct_submodule_import_does_not_shadow_export() -> None:
    """直接 import 同名实现模块后，包属性仍是可调用函数。

    本子包受「文件名 = 函数名」约定约束，import ``honet.contagion.hon_sis``
    会把包属性写成模块对象；子包用自定义模块类型拦截，把它还原成函数。
    """
    import subprocess
    import sys

    script = """
from honet import contagion

# 先直接 import 子模块路径，再访问包属性
import honet.contagion.hon_sis  # noqa: F401

assert callable(contagion.hon_sis), type(contagion.hon_sis)

# 全部导出在任意访问顺序下都应是可调用对象
for name in contagion.__all__:
    value = getattr(contagion, name)
    assert callable(value), f"{name} 不是可调用对象：{type(value)}"

# 别名与正名仍是同一对象
assert contagion.honSIS is contagion.hon_sis
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_import_honet_does_not_pull_dependencies() -> None:
    """``import honet`` 与访问 contagion 的全部导出都不得拉第三方依赖。

    本子包的实现模块只在模块级 import ``honet._deps`` / ``honet._hnx`` 这类
    零依赖基础设施，numpy / hypernetx 一律在函数体内通过 ``require()`` 获取，
    因此「把 ``__all__`` 里的名字全部取一遍」也不应加载任何第三方包；真正
    调用模拟函数时才加载。
    """
    import subprocess
    import sys

    script = """
import sys
import honet

THIRD_PARTY = ("hypernetx", "numpy", "pandas", "scipy", "matplotlib")


def loaded():
    return sorted({m.split(".")[0] for m in sys.modules} & set(THIRD_PARTY))


assert not loaded(), f"import honet 拉了第三方依赖：{loaded()}"

from honet import contagion

for name in contagion.__all__:
    getattr(contagion, name)

assert not loaded(), f"访问导出符号就拉了第三方依赖：{loaded()}"

# 真正调用模拟函数之后才允许加载 numpy / hypernetx
import hypernetx as hnx

H = hnx.Hypergraph({"e1": ["a", "b"], "e2": ["b", "c"]})
contagion.hon_sis(H, 0.3, {2: 0.5}, tmax=2, seed=1)
assert "numpy" in sys.modules
assert "hypernetx" in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
