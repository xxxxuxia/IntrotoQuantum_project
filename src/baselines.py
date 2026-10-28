"""Classical baselines for the job-placement problem: Karmarkar-Karp, greedy LPT,
simulated annealing, CP-SAT.

KK and greedy LPT are pure load-balancing heuristics: they only look at job
weights and ignore the conflict graph entirely, which is standard practice for
these algorithms. Simulated annealing optimizes the full cost function
(balance + conflict penalty) via local search, so it's the one baseline that
can actually respect anti-affinity constraints. CP-SAT is an exact solver: it
can find (and often prove) the true optimum well past n=24, where
problem.brute_force stops being feasible.

Every function returns a dict with the same shape so results are directly
comparable: {"bitstring", "cost", "violations", "wall_time_s"}. cp_sat_solve
additionally reports "status" and "optimal".
"""

import heapq
import itertools
import time

import numpy as np
from ortools.sat.python import cp_model

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


# ---------------------------------------------------------------- CP-SAT

def cp_sat_solve(inst, time_limit_s=10, num_workers=8):
    """Exact solve via CP-SAT, usable far past problem.brute_force's n<=24 limit.

    CP-SAT only works with integers, so the model is built on the raw integer
    weights (not the normalized ones): D = sum(w_raw_i * z_i) is an integer,
    D^2 is linearized via AddMultiplicationEquality, and dividing by max(w_raw)^2
    at the end recovers the same normalized cost problem.cost() reports. Each
    conflict edge similarly gets one linearized product term (z_i*z_j via the
    boolean-AND identity) so the full C(z) -- balance and penalty together --
    is the actual objective being minimized, same as simulated_annealing.
    """
    t0 = time.perf_counter()
    n = inst["n"]
    w_raw = inst["weights_raw"]
    max_w = max(w_raw)
    lam = inst["lambda"]

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{i}") for i in range(n)]  # x_i=1 -> z_i=-1

    sum_w = sum(w_raw)
    D = model.NewIntVar(-sum_w, sum_w, "D")
    model.Add(D == sum(w_raw[i] * (1 - 2 * x[i]) for i in range(n)))
    D_sq = model.NewIntVar(0, sum_w * sum_w, "D_sq")
    model.AddMultiplicationEquality(D_sq, [D, D])

    obj_terms = [(1.0 / (max_w ** 2)) * D_sq]
    for i, j in inst["edges"]:
        p_ij = model.NewBoolVar(f"p{i}_{j}")
        model.AddMultiplicationEquality(p_ij, [x[i], x[j]])
        obj_terms.append(lam * (1 - 2 * x[i] - 2 * x[j] + 4 * p_ij))
    model.Minimize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = num_workers
    status = solver.Solve(model)

    bitstring = "".join(str(solver.Value(xi)) for xi in x)
    z = P.bitstring_to_spins(bitstring)
    result = _result(inst, z, t0)
    result["status"] = solver.StatusName(status)
    result["optimal"] = status == cp_model.OPTIMAL
    return result


def reference_optimum(inst, cp_sat_time_limit_s=10):
    """Ground truth (c_min, c_max, opt_states) for approx_ratio/p_opt, valid for any n.

    Uses problem.brute_force (exact, exhaustive) up to n=24 as the team agreed;
    above that, c_max has a closed form (problem.c_max_exact) regardless of n,
    and c_min falls back to CP-SAT, which should be able to prove optimality
    well past n=24 for instances of this size/structure (verified up to n=30
    during development). opt_states is every optimal bitstring for n<=24 (from
    brute_force), or the one CP-SAT found plus its spin-flip mirror above that
    -- cost is invariant under flipping every spin, so the mirror is optimal too.
    """
    if inst["n"] <= P.BRUTE_FORCE_MAX_N:
        bf = P.brute_force(inst)
        return {"c_min": bf["c_min"], "c_max": bf["c_max"], "opt_states": bf["opt_states"],
                "method": "brute_force", "optimal": True}
    r = cp_sat_solve(inst, time_limit_s=cp_sat_time_limit_s)
    mirror = "".join("1" if b == "0" else "0" for b in r["bitstring"])
    return {"c_min": r["cost"], "c_max": P.c_max_exact(inst), "opt_states": [r["bitstring"], mirror],
            "method": "cp_sat", "optimal": r["optimal"]}
