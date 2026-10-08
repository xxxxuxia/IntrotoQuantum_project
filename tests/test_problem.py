import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import problem as P


def test_generator_is_reproducible_and_matches_schema():
    a = P.generate_instance(10, seed=7)
    b = P.generate_instance(10, seed=7)
    assert a == b
    assert set(a) == {"id", "n", "weights", "edges", "lambda", "seed"}
    assert len(a["weights"]) == 10
    assert all(isinstance(w, int) and w >= 1 for w in a["weights"])
    assert all(0 <= i < j < 10 for i, j in a["edges"])
    assert P.generate_instance(10, seed=8) != a


def test_save_load_roundtrip():
    import tempfile
    inst = P.generate_instance(6, seed=1)
    with tempfile.TemporaryDirectory() as d:
        assert P.load_instance(P.save_instance(inst, d)) == inst


def test_tiny_hand_checked_instance():
    inst = {"id": "t", "n": 3, "weights": [1, 1, 2], "edges": [], "lambda": 1.0, "seed": 0}
    bf = P.brute_force(inst)
    assert np.isclose(bf["c_min"], 0.0)   # {0,1} vs {2}: 0.5 + 0.5 - 1 = 0
    assert np.isclose(bf["c_max"], 4.0)   # all on one node: (0.5+0.5+1)^2
    assert sorted(bf["opt_states"]) == ["001", "110"]


def test_brute_force_matches_naive_loop():
    for seed in range(5):
        inst = P.generate_instance(7, seed=seed)
        naive = [P.cost(inst, np.array(z)) for z in itertools.product([1, -1], repeat=7)]
        bf = P.brute_force(inst, chunk_bits=4)
        assert np.isclose(bf["c_min"], min(naive))
        assert np.isclose(bf["c_max"], max(naive))
        for s in bf["opt_states"]:
            assert np.isclose(P.cost(inst, P.bitstring_to_spins(s)), bf["c_min"])


def test_optima_come_in_mirror_pairs():
    for seed in range(5):
        bf = P.brute_force(P.generate_instance(8, seed=seed))
        states = set(bf["opt_states"])
        flip = str.maketrans("01", "10")
        assert all(s.translate(flip) in states for s in states)


def test_mean_cost_over_all_states_is_sum_w_squared():
    # E0 check: at gamma = beta = 0 QAOA samples uniformly, so <C> = sum(w_i^2)
    inst = P.generate_instance(8, seed=3)
    bf = P.brute_force(inst, return_costs=True)
    assert np.isclose(bf["costs"].mean(), (P.normalized_weights(inst) ** 2).sum())


def test_violations():
    inst = {"id": "t", "n": 3, "weights": [1, 1, 1], "edges": [[0, 1], [1, 2]], "lambda": 1.0, "seed": 0}
    assert P.violations(inst, P.bitstring_to_spins("000")) == 2
    assert P.violations(inst, P.bitstring_to_spins("010")) == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
