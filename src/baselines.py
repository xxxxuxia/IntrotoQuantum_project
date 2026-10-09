"""Classical baselines for the job-placement problem: Karmarkar-Karp, greedy LPT,
simulated annealing.

KK and greedy LPT are pure load-balancing heuristics: they only look at job
weights and ignore the conflict graph entirely, which is standard practice for
these algorithms. Simulated annealing optimizes the full cost function
(balance + conflict penalty) via local search, so it's the one baseline that
can actually respect anti-affinity constraints.

Every function returns a dict with the same shape so results are directly
comparable: {"bitstring", "cost", "violations", "wall_time_s"}.
"""

import heapq
import itertools
import time

import numpy as np

import problem as P


def _result(inst, z, t0):
    return {
        "bitstring": P.spins_to_bitstring(z),
        "cost": float(P.cost(inst, z)),
        "violations": int(P.violations(inst, z)),
        "wall_time_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------- Karmarkar-Karp

def karmarkar_karp(inst):
    """KK differencing heuristic for 2-way number partitioning (balance only)."""
    t0 = time.perf_counter()
    n = inst["n"]
    w = P.normalized_weights(inst)

    tiebreak = itertools.count()
    heap = []
    for i in range(n):
        signs = np.zeros(n, dtype=np.int8)
        signs[i] = 1
        heapq.heappush(heap, (-w[i], next(tiebreak), signs))

    while len(heap) > 1:
        neg_a, _, signs_a = heapq.heappop(heap)
        neg_b, _, signs_b = heapq.heappop(heap)
        merged = signs_a - signs_b
        heapq.heappush(heap, (neg_a - neg_b, next(tiebreak), merged))

    _, _, z = heap[0]
    return _result(inst, z, t0)


# ---------------------------------------------------------------- greedy LPT

def greedy_lpt(inst):
    """Greedy Longest-Processing-Time heuristic (balance only)."""
    t0 = time.perf_counter()
    n = inst["n"]
    w = P.normalized_weights(inst)
    order = np.argsort(-w)

    side_sum = [0.0, 0.0]
    z = np.empty(n, dtype=np.int8)
    for j in order:
        side = 0 if side_sum[0] <= side_sum[1] else 1
        z[j] = 1 if side == 0 else -1
        side_sum[side] += w[j]

    return _result(inst, z, t0)


# ---------------------------------------------------------------- simulated annealing

def simulated_annealing(inst, seed, n_iters=20000, T0=1.0, T_min=1e-3, z_init=None):
    """Single-spin-flip SA on the full cost C(z) (balance + conflict penalty)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = inst["n"]

    if z_init is not None:
        z = np.array(z_init, dtype=np.int8)
    else:
        z = rng.choice(np.array([1, -1], dtype=np.int8), size=n)

    cur_cost = float(P.cost(inst, z))
    best_z, best_cost = z.copy(), cur_cost
    cooling_rate = (T_min / T0) ** (1.0 / n_iters)
    T = T0

    for _ in range(n_iters):
        i = rng.integers(n)
        z_new = z.copy()
        z_new[i] *= -1
        new_cost = float(P.cost(inst, z_new))
        delta = new_cost - cur_cost
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            z, cur_cost = z_new, new_cost
            if cur_cost < best_cost:
                best_z, best_cost = z.copy(), cur_cost
        T = max(T * cooling_rate, T_min)

    return _result(inst, best_z, t0)
