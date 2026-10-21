"""QAOA for the scheduling Hamiltonian in CUDA-Q (P1).

Usage (P2 / P3):
    params, energy, n_evals = optimize(inst, p=3)            # COBYLA on <H>
    counts = sample(inst, params, shots=10_000)              # {bitstring: count}
    m      = problem.sample_metrics(inst, counts, ref)
    params4 = interp_init(params)                            # warm start for p + 1

Circuit (one qubit per job, bit 0 -> z = +1 -> node A, qubit 0 first):
    |+>^n  ->  prod_{l=1..p} [ exp(-i gamma_l H)  exp(-i beta_l sum_i X_i) ]

    exp(-i gamma J Z_i Z_j) = CX(i,j) . RZ(2 gamma J) on j . CX(i,j)
    exp(-i beta X_i)        = RX(2 beta) on i

Parameter layout everywhere: params = [gamma_1 .. gamma_p, beta_1 .. beta_p].
Energies are in scaled units (H = C / scale, max |J_ij| = 1, offset included),
so <C> = scale * <H>; expected_cost() does that conversion.
"""

import time

import cudaq
import numpy as np
from scipy.optimize import minimize

import problem as P


# ---------------------------------------------------------------- kernel

@cudaq.kernel
def qaoa_kernel(n: int, p: int, ii: list[int], jj: list[int], jv: list[float],
                params: list[float]):
    q = cudaq.qvector(n)
    h(q)
    for layer in range(p):
        gamma = params[layer]
        beta = params[p + layer]
        for k in range(len(ii)):
            x.ctrl(q[ii[k]], q[jj[k]])
            rz(2.0 * gamma * jv[k], q[jj[k]])
            x.ctrl(q[ii[k]], q[jj[k]])
        for i in range(n):
            rx(2.0 * beta, q[i])


def problem_data(inst):
    """Everything the kernel and observe need, built once per instance."""
    ham = P.build_hamiltonian(inst)
    ii = [i for i, _, _ in ham["terms"]]
    jj = [j for _, j, _ in ham["terms"]]
    jv = [J for _, _, J in ham["terms"]]
    op = 0.0 * cudaq.spin.i(0)
    for i, j, J in ham["terms"]:
        op += J * cudaq.spin.z(i) * cudaq.spin.z(j)
    return {"n": inst["n"], "ii": ii, "jj": jj, "jv": jv, "op": op,
            "offset": ham["offset"], "scale": ham["scale"]}


def _args(data, params):
    params = [float(x) for x in params]
    return data["n"], len(params) // 2, data["ii"], data["jj"], data["jv"], params


# ---------------------------------------------------------------- energy and sampling

def energy(data, params):
    """<H> in scaled units, offset included."""
    zz = cudaq.observe(qaoa_kernel, data["op"], *_args(data, params)).expectation()
    return data["offset"] + zz


def expected_cost(inst, params, data=None):
    """<C> in cost units, comparable with brute_force()'s c_min / c_max."""
    data = data or problem_data(inst)
    return data["scale"] * energy(data, params)


def sample(inst, params, shots=10_000, seed=None, data=None):
    """Measure the QAOA state; returns a {bitstring: count} dict, qubit 0 first."""
    data = data or problem_data(inst)
    if seed is not None:
        cudaq.set_random_seed(int(seed))
    result = cudaq.sample(qaoa_kernel, *_args(data, params), shots_count=shots)
    return {bits: count for bits, count in result.items()}


def probabilities(inst, params, data=None):
    """Exact probability of every bitstring (index order of problem.index_to_bitstring).

    Statevector only, so keep n small; used by the E0 tests.
    """
    data = data or problem_data(inst)
    state = cudaq.get_state(qaoa_kernel, *_args(data, params))
    n = inst["n"]
    return np.array([abs(state.amplitude([int(b) for b in P.index_to_bitstring(k, n)])) ** 2
                     for k in range(1 << n)])


# ---------------------------------------------------------------- initial angles

def ramp_init(p, dt=0.75):
    """Linear ramp, a discretised annealing schedule: gamma rises, beta falls."""
    t = (np.arange(1, p + 1) - 0.5) / p
    return np.concatenate([t * dt, (1 - t) * dt])


def random_init(p, rng):
    """Uniform random angles: gamma in [0, pi], beta in [0, pi/2]."""
    return np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)])


def _interp(x):
    """INTERP (Zhou et al. 2020) for one angle vector of length p -> p + 1."""
    p = len(x)
    padded = np.concatenate([[0.0], x, [0.0]])
    i = np.arange(1, p + 2)
    return (i - 1) / p * padded[i - 1] + (p - i + 1) / p * padded[i]


def interp_init(params):
    """Warm start for depth p + 1 from optimised depth-p params."""
    params = np.asarray(params, dtype=np.float64)
    p = len(params) // 2
    return np.concatenate([_interp(params[:p]), _interp(params[p:])])


# ---------------------------------------------------------------- optimisation

def optimize(inst, p, init_params=None, method="COBYLA", maxiter=500, data=None):
    """Minimise <H> over the 2p angles; returns (params, energy, n_evals).

    energy is <H> in scaled units; multiply by data["scale"] for <C>.
    """
    data = data or problem_data(inst)
    x0 = ramp_init(p) if init_params is None else np.asarray(init_params, dtype=np.float64)
    if len(x0) != 2 * p:
        raise ValueError(f"init_params has length {len(x0)}, expected 2p = {2 * p}")

    n_evals = 0

    def f(x):
        nonlocal n_evals
        n_evals += 1
        return energy(data, x)

    options = {"maxiter": maxiter}
    if method == "COBYLA":
        options["rhobeg"] = 0.2
    res = minimize(f, x0, method=method, options=options)
    return res.x, float(res.fun), n_evals


def run(inst, p, ref, init_params=None, shots=10_000, seed=0, method="COBYLA", data=None):
    """Optimise, then sample; returns params and one results-CSV row (bench.CSV_FIELDS)."""
    data = data or problem_data(inst)
    t0 = time.perf_counter()
    params, e, n_evals = optimize(inst, p, init_params, method=method, data=data)
    counts = sample(inst, params, shots=shots, seed=seed, data=data)
    wall = time.perf_counter() - t0

    m = P.sample_metrics(inst, counts, ref)
    mean_c = data["scale"] * e
    row = {
        "instance_id": inst["id"], "n": inst["n"], "p": p, "optimizer": method,
        "energy": e,
        "approx_ratio": float(P.approx_ratio(mean_c, ref["c_min"], ref["c_max"])),
        "p_opt": m["p_opt"], "best_cost": m["best_cost"],
        "violations": m["violations"], "makespan": m["makespan"],
        "n_evals": n_evals, "wall_time_s": wall, "seed": seed,
    }
    return params, row
