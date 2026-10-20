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
    assert set(a) == {"id", "n", "seed", "weights_raw", "weights", "edges",
                      "planted_sides", "lambda", "scale", "params"}
    assert len(a["weights_raw"]) == len(a["weights"]) == 10
    assert all(isinstance(w, int) and w >= 1 for w in a["weights_raw"])
    assert max(a["weights"]) == 1.0
    assert np.allclose(a["weights"], np.array(a["weights_raw"]) / max(a["weights_raw"]))
    assert all(0 <= i < j < 10 for i, j in a["edges"])
    assert P.generate_instance(10, seed=8) != a


def test_save_load_roundtrip():
    import tempfile
    inst = P.generate_instance(6, seed=1)
    with tempfile.TemporaryDirectory() as d:
        assert P.load_instance(P.save_instance(inst, d)) == inst


def test_tiny_hand_checked_instance():
    inst = {"id": "t", "n": 3, "weights_raw": [1, 1, 2], "weights": [0.5, 0.5, 1.0], "edges": [], "lambda": 1.0, "seed": 0}
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


def test_planted_assignment_has_zero_violations():
    for seed in range(20):
        inst = P.generate_instance(12, seed=seed)
        sides = inst["planted_sides"]
        assert all(sides[i] != sides[j] for i, j in inst["edges"])
        planted = "".join(str(b) for b in sides)
        assert P.violations(inst, P.bitstring_to_spins(planted)) == 0


def test_hamiltonian_reproduces_cost_on_every_state():
    for seed in range(5):
        inst = P.generate_instance(8, seed=seed, lam=2.0)
        ham = P.build_hamiltonian(inst)
        z = np.array(list(itertools.product([1, -1], repeat=8)))
        assert np.allclose(ham["scale"] * P.ising_energy(ham, z), P.cost(inst, z))


def test_hamiltonian_is_normalised():
    for seed in range(5):
        inst = P.generate_instance(10, seed=seed)
        ham = P.build_hamiltonian(inst)
        assert np.isclose(max(abs(J) for _, _, J in ham["terms"]), 1.0)
        assert all(i < j for i, j, _ in ham["terms"])
        assert np.isclose(ham["scale"], inst["scale"])


def test_hamiltonian_hand_checked():
    # w = [0.5, 0.5, 1], edge (0,1), lambda = 1:
    # J01 = 2*.25 + 1 = 1.5, J02 = J12 = 2*.5 = 1  ->  scale 1.5, offset 1.5/1.5 = 1
    inst = {"n": 3, "weights": [0.5, 0.5, 1.0], "edges": [[0, 1]], "lambda": 1.0}
    ham = P.build_hamiltonian(inst)
    assert np.isclose(ham["scale"], 1.5)
    assert np.isclose(ham["offset"], 1.0)
    assert [(i, j) for i, j, _ in ham["terms"]] == [(0, 1), (0, 2), (1, 2)]
    assert np.allclose([J for *_, J in ham["terms"]], [1.0, 2 / 3, 2 / 3])


def test_planted_split_is_balanced():
    for n in (4, 5, 8, 11):
        for seed in range(10):
            sides = P.generate_instance(n, seed=seed)["planted_sides"]
            assert sum(sides) == n // 2


def test_bitstring_index_roundtrip():
    n = 6
    for k in range(1 << n):
        assert P.bitstring_to_index(P.index_to_bitstring(k, n)) == k
    assert P.bitstring_to_index("100") == 1   # qubit 0 is the lowest bit
    assert P.bitstring_to_index("001") == 4


def test_brute_force_index_matches_bitstring_cost():
    # brute_force scores index k; the reported bitstring must give the same cost
    inst = P.generate_instance(6, seed=2)
    bf = P.brute_force(inst, return_costs=True)
    for k in (0, 5, 17, 63):
        z = P.bitstring_to_spins(P.index_to_bitstring(k, 6))
        assert np.isclose(bf["costs"][k], P.cost(inst, z))


def test_c_max_exact_matches_brute_force():
    for seed in range(5):
        inst = P.generate_instance(9, seed=seed, lam=2.0)
        assert np.isclose(P.c_max_exact(inst), P.brute_force(inst)["c_max"])


def test_approx_ratio_endpoints():
    assert P.approx_ratio(1.0, 1.0, 5.0) == 1.0
    assert P.approx_ratio(5.0, 1.0, 5.0) == 0.0
    assert P.approx_ratio(2.0, 1.0, 5.0) == 0.75


def test_makespan_raw_units():
    inst = {"weights_raw": [3, 5, 2]}
    assert P.makespan(inst, [1, -1, 1]) == 5      # A = 3+2, B = 5
    assert P.makespan(inst, [1, 1, 1]) == 10      # everything on A


def test_tts99():
    assert P.tts99(2.0, 0.995) == 2.0
    assert P.tts99(2.0, 0.0) == np.inf
    assert np.isclose(P.tts99(1.0, 0.5), np.log(0.01) / np.log(0.5))   # ~6.64 runs


def test_sample_metrics_on_fake_counts():
    inst = P.generate_instance(8, seed=4)
    bf = P.brute_force(inst)
    opt, worst = bf["opt_states"][0], bf["max_state"]
    m = P.sample_metrics(inst, {opt: 30, worst: 70}, bf)
    assert np.isclose(m["p_opt"], 0.3)
    assert m["best_state"] == opt and np.isclose(m["best_cost"], bf["c_min"])
    assert np.isclose(m["best_approx_ratio"], 1.0)
    assert np.isclose(m["approx_ratio"], 0.3)            # linear in mean cost
    assert m["violations"] == bf["opt_violations"]
    assert m["makespan"] == P.makespan(inst, P.bitstring_to_spins(opt))


def test_violations():
    inst = {"id": "t", "n": 3, "weights_raw": [1, 1, 1], "weights": [1.0, 1.0, 1.0], "edges": [[0, 1], [1, 2]], "lambda": 1.0, "seed": 0}
    assert P.violations(inst, P.bitstring_to_spins("000")) == 2
    assert P.violations(inst, P.bitstring_to_spins("010")) == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
