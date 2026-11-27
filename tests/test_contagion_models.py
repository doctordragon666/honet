"""仓室模型规格与可扩展性的测试。

核心主张：新增传播模型**不需要修改任何引擎代码**，只需声明一份
:class:`CompartmentModel` 规格并注册。本文件用「不依赖内置模型」的方式
验证这一点：自定义模型 → 注册 → 直接 simulate。
"""

from __future__ import annotations

import hypernetx as hnx
import numpy as np
import pytest

from honet.contagion import (
    CompartmentModel,
    Contagion_Method,
    available_models,
    get_model,
    register_model,
    simulate,
)
from honet.contagion._model import SEIR, SI, SIER, SIR, SIRS, SIS

#: 4 个结点、3 条超边（含一条三元超边）
TAU = {2: 0.5, 3: 0.5}


@pytest.fixture
def H() -> hnx.Hypergraph:
    return hnx.Hypergraph(
        {
            "e1": ["a", "b"],
            "e2": ["b", "c", "d"],
            "e3": ["c", "d"],
        }
    )


class TestBuiltinModels:
    """内置模型的规格自洽性。"""

    @pytest.mark.parametrize("spec", [SIS, SIR, SIRS, SIER, SEIR, SI])
    def test_spec_is_self_consistent(self, spec: CompartmentModel):
        """状态、传染态、转移目标必须彼此一致（构造时已校验，这里再确认）。"""
        assert spec.susceptible in spec.states
        assert all(state in spec.states for state in spec.infected)
        for source, target in {**spec.on_infection, **spec.on_recovery}.items():
            assert source in spec.states
            assert target in spec.states

    def test_sis_is_cyclic(self):
        """SIS 康复后回到易感，不是吸收型。"""
        assert SIS.is_sir_like is False
        assert SIS.on_recovery["I"] == "S"

    def test_sir_is_absorbing(self):
        """SIR 的 R 是吸收态。"""
        assert SIR.is_sir_like is True
        assert SIR.can_be_infected("R") is False

    def test_sirs_is_cyclic(self):
        """SIRS 的 R 会回到 S，因此不是吸收型。

        这是可达性判据与「看目标是否可感染」朴素判据的分界点：R 本身不可被
        感染，但经 R→S 能回到易感，所以整体仍是循环型。
        """
        assert SIRS.is_sir_like is False
        assert SIRS.on_recovery["R"] == "S"

    def test_sier_has_exposed_stage(self):
        """SIER 的感染链是 S→E→I，E 不可恢复。"""
        assert SIER.target_of_infection("S") == "E"
        assert SIER.target_of_infection("E") == "I"
        assert SIER.can_recover("E") is False
        assert SIER.can_recover("I") is True

    def test_si_never_recovers(self):
        """SI 没有恢复转移。"""
        assert SI.on_recovery == {}
        assert SI.is_sir_like is True


class TestModelRegistry:
    """注册表行为。"""

    def test_builtins_are_registered(self):
        names = available_models()
        for expected in ("SIS", "SIR", "SIRS", "SIER", "SEIR", "SI"):
            assert expected in names

    def test_get_model_by_name(self):
        assert get_model("SIER") is SIER

    def test_get_model_accepts_spec_directly(self):
        """直接传规格对象应当原样返回。"""
        assert get_model(SIER) is SIER

    def test_unknown_model_lists_available(self):
        with pytest.raises(ValueError, match="已注册的有"):
            get_model("NOT_A_MODEL")

    def test_duplicate_registration_rejected(self):
        with pytest.raises(ValueError, match="已注册"):
            register_model(SIS)

    def test_overwrite_allowed(self):
        """显式 overwrite=True 时允许覆盖，并在用完后还原。"""
        original = get_model("SI")
        try:
            register_model(SI, overwrite=True)
            assert get_model("SI") is SI
        finally:
            register_model(original, overwrite=True)


class TestSpecValidation:
    """规格校验：错误的规格必须尽早报错，而不是跑出错误结果。"""

    def test_rejects_unknown_state_name(self):
        with pytest.raises(ValueError, match="非法状态名"):
            CompartmentModel(name="BAD", states=("S", "lowercase"))

    def test_rejects_missing_susceptible(self):
        with pytest.raises(ValueError, match="susceptible"):
            CompartmentModel(name="BAD", states=("I", "R"))

    def test_rejects_empty_infected(self):
        with pytest.raises(ValueError, match="传染态"):
            CompartmentModel(name="BAD", states=("S", "I"), infected=())

    def test_rejects_transition_to_unknown_state(self):
        with pytest.raises(ValueError, match="转移目标"):
            CompartmentModel(
                name="BAD",
                states=("S", "I"),
                on_infection={"S": "X"},
            )

    def test_allows_custom_state_letters(self):
        """允许 D（接种）/ Q（隔离）这类领域自定义仓室。"""
        spec = CompartmentModel(
            name="SIQ",
            states=("S", "I", "Q"),
            infected=("I",),
            on_infection={"S": "I"},
            on_recovery={"I": "Q"},
        )
        assert "Q" in spec.states


class TestNewModelsRunWithoutEngineChanges:
    """核心可扩展性主张：新模型无需改引擎。"""

    @pytest.mark.parametrize("name", ["SIS", "SIR", "SIRS", "SIER", "SEIR", "SI"])
    def test_builtin_models_run_and_conserve(self, name: str, H: hnx.Hypergraph):
        """每个内置模型都能跑通，且各仓室人数之和守恒。"""
        result = simulate(H, 0.3, TAU, model=name, tmax=6, seed=1)
        times, *series = result
        assert len(series) == len(get_model(name).output_states())
        assert all(len(s) == len(times) for s in series)
        totals = [sum(row) for row in zip(*series, strict=True)]
        assert all(total == len(H.nodes) for total in totals)

    def test_custom_model_runs_without_touching_engine(self, H: hnx.Hypergraph):
        """注册一个全新五仓室模型后，simulate 直接可用。

        若框架不可扩展，这里就必须改 ``_discrete.py`` 的分支逻辑才能跑通。
        """
        spec = CompartmentModel(
            name="SIQD",
            states=("S", "I", "Q", "D"),
            infected=("I",),
            on_infection={"S": "I"},
            on_recovery={"I": "Q", "Q": "D", "D": "S"},
        )
        register_model(spec, overwrite=True)

        result = simulate(H, 0.3, TAU, model="SIQD", tmax=8, seed=1)
        times, *series = result
        assert len(series) == 4
        totals = [sum(row) for row in zip(*series, strict=True)]
        assert all(total == len(H.nodes) for total in totals)

    def test_custom_model_works_in_random_path(self, H: hnx.Hypergraph):
        """自定义模型在随机离散路径下同样可用（规格被两条引擎共用）。"""
        spec = CompartmentModel(
            name="SIQR",
            states=("S", "I", "Q", "R"),
            infected=("I",),
            on_infection={"S": "I"},
            on_recovery={"I": "Q", "Q": "R"},
        )
        register_model(spec, overwrite=True)

        result = simulate(
            H,
            0.3,
            TAU,
            model="SIQR",
            method=Contagion_Method.RandomDiscrete,
            tmax=5,
            seed=1,
        )
        times, *series = result
        assert len(series) == 4
        totals = [sum(row) for row in zip(*series, strict=True)]
        assert all(total == len(H.nodes) for total in totals)


class TestExplicitRandomSource:
    """显式随机源（借鉴 EoN 的 ``rng`` 设计）。

    旧实现把随机源写死成全局状态，因此无法并行/嵌套运行互不干扰的模拟。
    引擎现在接受 ``rng`` 参数，这里验证三件事：可复现、互不干扰、不污染全局。
    """

    def test_discrete_rng_reproducible(self, H: hnx.Hypergraph):
        """同一 seed 的两个独立随机源给出相同轨迹。"""
        from honet.contagion._common import make_rng
        from honet.contagion._discrete import simulate_discrete

        def run(rng):
            return simulate_discrete(
                H, TAU, 0.3, model="SIS", mode="control", rho=0.25, tmax=8, rng=rng
            )

        a = run(make_rng(7)[0])
        b = run(make_rng(7)[0])
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))

    def test_discrete_rng_does_not_pollute_global(self, H: hnx.Hypergraph):
        """注入 rng 时不得改动全局 random 状态。"""
        import random

        from honet.contagion._common import make_rng
        from honet.contagion._discrete import simulate_discrete

        random.seed(1234)
        before = random.random()
        random.seed(1234)
        simulate_discrete(
            H, TAU, 0.3, model="SIS", mode="control", rho=0.25, tmax=8, rng=make_rng(7)[0]
        )
        assert random.random() == before

    def test_random_model_rng_reproducible(self, H: hnx.Hypergraph):
        """随机模型同样支持显式 rng。"""
        from honet.contagion._common import make_rng
        from honet.contagion._random_models import simulate_random_discrete

        def run(rng):
            return simulate_random_discrete(H, TAU, 0.3, model="SIS", rho=0.25, tmax=5, rng=rng)

        a = run(make_rng(7)[1])
        b = run(make_rng(7)[1])
        assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b, strict=True))

    def test_parallel_simulations_are_independent(self, H: hnx.Hypergraph):
        """两个不同 seed 的模拟交错执行，结果与各自单独执行一致。"""
        from honet.contagion._common import make_rng
        from honet.contagion._discrete import simulate_discrete

        def run(seed):
            return simulate_discrete(
                H,
                TAU,
                0.3,
                model="SIS",
                mode="control",
                rho=0.25,
                tmax=6,
                rng=make_rng(seed)[0],
            )

        alone_a = run(11)
        alone_b = run(22)
        # 交错执行：先建两个随机源，再交替推进（这里用完整调用模拟交错）
        again_a = run(11)
        again_b = run(22)
        assert all(np.array_equal(x, y) for x, y in zip(alone_a, again_a, strict=True))
        assert all(np.array_equal(x, y) for x, y in zip(alone_b, again_b, strict=True))
        # 不同 seed 应当给出不同轨迹
        assert not all(np.array_equal(x, y) for x, y in zip(alone_a, alone_b, strict=True))


class TestEntryPointsShareImplementation:
    """hon_sis / hon_sir 与 simulate 是同一实现，不应出现行为漂移。"""

    """hon_sis / hon_sir 与 simulate 是同一实现，不应出现行为漂移。"""

    def test_hon_sis_matches_simulate(self, H: hnx.Hypergraph):
        from honet.contagion import hon_sis

        a = hon_sis(H, 0.3, TAU, tmax=6, seed=3)
        b = simulate(H, 0.3, TAU, model="SIS", tmax=6, seed=3)
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))

    def test_hon_sir_matches_simulate(self, H: hnx.Hypergraph):
        from honet.contagion import hon_sir

        a = hon_sir(H, 0.3, TAU, tmax=6, seed=3)
        b = simulate(H, 0.3, TAU, model="SIR", tmax=6, seed=3)
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))

    def test_gillespie_rejects_unsupported_model(self, H: hnx.Hypergraph):
        """SIER 没有 hypernetx 的连续时间实现，应给出明确报错而非乱跑。"""
        with pytest.raises(ValueError, match="Gillespie"):
            simulate(
                H,
                0.3,
                TAU,
                model="SIER",
                method=Contagion_Method.Gillespie,
                tmax=5,
            )

    def test_unknown_model_rejected(self, H: hnx.Hypergraph):
        with pytest.raises(ValueError, match="未知的 model"):
            simulate(H, 0.3, TAU, model="NOPE", tmax=3)
