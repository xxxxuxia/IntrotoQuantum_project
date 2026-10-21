"""E0: validate the QAOA kernel against the classical problem definition (P1 + P2)."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cudaq")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import problem as P
import qaoa as Q


def _all_spins(n):
    return np.array([P.bitstring_to_spins(P.index_to_bitstring(k, n)) for k in range(1 << n)])


def test_zero_angles_give_uniform_state_and_mean_cost():
    inst = P.generate_instance(6, seed=0)
    w = P.normalized_weights(inst)
    assert np.isclose(Q.expected_cost(inst, [0.0, 0.0]), (w ** 2).sum())
    assert np.allclose(Q.probabilities(inst, [0.0, 0.0]), 1 / 64)


def test_bit_order_qubit_zero_first():
    # A kernel-independent check of the convention problem.py relies on.
    import cudaq

    @cudaq.kernel
    def flip_first(n: int):
        q = cudaq.qvector(n)
        x(q[0])

    counts = cudaq.sample(flip_first, 4, shots_count=10)
    assert dict(counts.items()) == {"1000": 10}
    assert P.bitstring_to_index("1000") == 1


@pytest.mark.parametrize("params", [[0.4, 0.3], [0.2, 0.7, 0.5, 0.1]])
def test_observe_matches_exact_probabilities(params):
    # Catches wrong Hamiltonian signs, a missing factor of 2, and reversed qubits.
    inst = P.generate_instance(6, seed=1)
    probs = Q.probabilities(inst, params)
    assert np.isclose(Q.expected_cost(inst, params), probs @ P.cost(inst, _all_spins(6)))


def test_sampled_costs_match_observe():
    inst = P.generate_instance(6, seed=2)
    params = [0.4, 0.3]
    counts = Q.sample(inst, params, shots=20_000, seed=0)
    z = np.array([P.bitstring_to_spins(s) for s in counts])
    freq = np.array(list(counts.values()))
    sampled = (freq * P.cost(inst, z)).sum() / freq.sum()
    assert np.isclose(sampled, Q.expected_cost(inst, params), rtol=0.03)


def test_mirror_states_have_equal_probability():
    inst = P.generate_instance(6, seed=3)
    probs = Q.probabilities(inst, [0.5, 0.2, 0.3, 0.6])
    mirror = [P.bitstring_to_index(P.spins_to_bitstring(-z)) for z in _all_spins(6)]
    assert np.allclose(probs, probs[mirror])


def test_interp_init():
    assert np.allclose(Q.interp_init([0.4, 0.2]), [0.4, 0.4, 0.2, 0.2])
    x = Q.interp_init([0.1, 0.3, 0.6, 0.4])
    assert len(x) == 6
    assert np.allclose(x[:3], [0.1, 0.2, 0.3])


def test_p3_at_n6_concentrates_on_optimum():
    inst = P.generate_instance(6, seed=0)
    ref = P.brute_force(inst)
    random_p_opt = len(ref["opt_states"]) / 2 ** 6
    init = None
    for p in (1, 2, 3):
        params, row = Q.run(inst, p, ref, init_params=init, shots=10_000, seed=0)
        init = Q.interp_init(params)
    assert row["approx_ratio"] > 0.9
    assert row["p_opt"] > 5 * random_p_opt


def test_beats_random_sampling_at_n10():
    # W9 checkpoint (1): optimised p = 2 is clearly better than the uniform state.
    inst = P.generate_instance(10, seed=0)
    ref = P.brute_force(inst)
    random_r = P.approx_ratio((P.normalized_weights(inst) ** 2).sum(), ref["c_min"], ref["c_max"])
    params, row = Q.run(inst, 1, ref)
    _, row = Q.run(inst, 2, ref, init_params=Q.interp_init(params))
    assert row["approx_ratio"] > random_r + 0.05
    assert row["p_opt"] > 2 * len(ref["opt_states"]) / 2 ** 10
