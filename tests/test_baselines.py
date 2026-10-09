import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import baselines as B
import problem as P


def _without_timing(result):
    return {k: v for k, v in result.items() if k != "wall_time_s"}


def test_output_schema():
    inst = P.generate_instance(6, seed=1)
    for result in (B.karmarkar_karp(inst), B.greedy_lpt(inst),
                   B.simulated_annealing(inst, seed=0, n_iters=200)):
        assert set(result) == {"bitstring", "cost", "violations", "wall_time_s"}
        assert isinstance(result["bitstring"], str) and len(result["bitstring"]) == 6
        assert set(result["bitstring"]) <= {"0", "1"}
        assert isinstance(result["cost"], float)
        assert isinstance(result["violations"], int)
        assert result["wall_time_s"] >= 0


def test_kk_and_lpt_cost_matches_recompute():
    inst = {"id": "t", "n": 3, "weights": [1, 1, 2], "edges": [], "lambda": 1.0, "seed": 0}
    bf = P.brute_force(inst)
    for result in (B.karmarkar_karp(inst), B.greedy_lpt(inst)):
        z = P.bitstring_to_spins(result["bitstring"])
        assert np.isclose(result["cost"], P.cost(inst, z))
        assert result["bitstring"] in bf["opt_states"]


def test_lpt_matches_hand_traced_assignment():
    # order by weight desc: 4,3,3,2 -> sides alternate 0,1,1,0 -> {4,2} vs {3,3}, balanced.
    inst = {"id": "t2", "n": 4, "weights": [4, 3, 3, 2], "edges": [], "lambda": 1.0, "seed": 0}
    result = B.greedy_lpt(inst)
    assert result["bitstring"] == "0110"
    assert np.isclose(result["cost"], 0.0)


def test_kk_and_lpt_ignore_conflict_graph():
    weights = [4, 3, 3, 2]
    inst_a = {"id": "a", "n": 4, "weights": weights, "edges": [], "lambda": 1.0, "seed": 0}
    inst_b = {"id": "b", "n": 4, "weights": weights, "edges": [[0, 1], [2, 3]], "lambda": 9.0, "seed": 0}
    assert B.karmarkar_karp(inst_a)["bitstring"] == B.karmarkar_karp(inst_b)["bitstring"]
    assert B.greedy_lpt(inst_a)["bitstring"] == B.greedy_lpt(inst_b)["bitstring"]


def test_kk_and_lpt_are_deterministic():
    inst = P.generate_instance(9, seed=2)
    assert _without_timing(B.karmarkar_karp(inst)) == _without_timing(B.karmarkar_karp(inst))
    assert _without_timing(B.greedy_lpt(inst)) == _without_timing(B.greedy_lpt(inst))


def test_sa_is_deterministic_given_seed():
    inst = P.generate_instance(8, seed=3)
    a = _without_timing(B.simulated_annealing(inst, seed=42, n_iters=500))
    b = _without_timing(B.simulated_annealing(inst, seed=42, n_iters=500))
    assert a == b


def test_sa_finds_known_optimum_on_small_instance():
    inst = P.generate_instance(8, seed=3)
    bf = P.brute_force(inst)
    for seed in range(5):
        result = B.simulated_annealing(inst, seed=seed, n_iters=5000)
        assert np.isclose(result["cost"], bf["c_min"], atol=1e-6)
        assert result["bitstring"] in bf["opt_states"]


def test_sa_respects_conflict_penalty():
    # LPT's balance-optimal split collocates jobs 0 and 2, which conflict here
    # under a heavy lambda; SA (using the full cost) finds the true optimum.
    inst = {"id": "t", "n": 4, "weights": [1, 1, 2, 2], "edges": [[2, 0]], "lambda": 5.0, "seed": 0}
    bf = P.brute_force(inst)
    lpt = B.greedy_lpt(inst)
    sa = B.simulated_annealing(inst, seed=1, n_iters=3000)
    assert lpt["violations"] > 0
    assert sa["violations"] == 0
    assert np.isclose(sa["cost"], bf["c_min"])
    assert sa["cost"] < lpt["cost"]


def test_sa_beats_random_baseline():
    inst = P.generate_instance(10, seed=5)
    rng = np.random.default_rng(99)
    best_random = min(
        float(P.cost(inst, rng.choice(np.array([1, -1], dtype=np.int8), size=10)))
        for _ in range(200)
    )
    sa = B.simulated_annealing(inst, seed=7, n_iters=3000)
    assert sa["cost"] <= best_random


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
