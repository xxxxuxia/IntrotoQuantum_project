"""Problem definition: instances, Hamiltonian, cost, brute force and metrics.

Usage (P1 / P3):
    inst = load_instance("instances/n10_s0.json")
    ham  = build_hamiltonian(inst)              # terms for the QAOA cost layer
    ref  = brute_force(inst)                    # c_min, c_max, opt_states
    m    = sample_metrics(inst, counts, ref)    # counts from cudaq.sample
    r    = approx_ratio(ham["scale"] * energy, ref["c_min"], ref["c_max"])

Formulation (fixed by the team):
    C(z) = (sum_i w_i z_i)^2 + lambda * sum_{(i,j) in E} z_i z_j,   z_i in {+1, -1}

z_i = +1 puts job i on node A, z_i = -1 puts it on node B.
Bit convention (same as Pauli-Z eigenstates): bit 0 -> z = +1, bit 1 -> z = -1.
Bitstrings are written qubit 0 first, so bitstring[i] is job i.
"""

import argparse
import json
from pathlib import Path

import numpy as np

EDGE_PROB = 0.2
LOGNORMAL_MU = 2.0
LOGNORMAL_SIGMA = 1.0
DEFAULT_LAMBDA = 0.5  # from experiments/lambda_sweep.py: smallest lambda with >= 90 % zero-violation optima
BRUTE_FORCE_MAX_N = 24


# ---------------------------------------------------------------- instances

def generate_instance(n, seed, lam=DEFAULT_LAMBDA, edge_prob=EDGE_PROB,
                      mu=LOGNORMAL_MU, sigma=LOGNORMAL_SIGMA):
    """One random instance: lognormal integer runtimes + planted-bipartition conflict graph.

    Jobs are split into two hidden sides of size floor(n/2) and ceil(n/2) at
    random positions. Conflict edges are only drawn between jobs on opposite
    sides, each with probability edge_prob, so the planted assignment (and its
    mirror) always has zero violations. The even split maximises the number of
    possible conflict pairs; the runtimes on each side are still unbalanced.
    """
    rng = np.random.default_rng(seed)
    weights_raw = np.maximum(1, np.rint(rng.lognormal(mu, sigma, size=n))).astype(int)
    sides = rng.permutation(np.arange(n) % 2)
    edges = [[i, j] for i in range(n) for j in range(i + 1, n)
             if sides[i] != sides[j] and rng.random() < edge_prob]
    inst = {
        "id": f"n{n}_s{seed}",
        "n": n,
        "seed": seed,
        "weights_raw": weights_raw.tolist(),
        "weights": (weights_raw / weights_raw.max()).tolist(),
        "edges": edges,
        "planted_sides": sides.tolist(),
        "lambda": lam,
    }
    inst["scale"] = hamiltonian_scale(inst)
    inst["params"] = {"edge_prob": edge_prob, "mu": mu, "sigma": sigma}
    return inst


def generate_instances(n, count, base_seed=0, lam=DEFAULT_LAMBDA):
    return [generate_instance(n, base_seed + k, lam=lam) for k in range(count)]


def save_instance(inst, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{inst['id']}.json"
    path.write_text(json.dumps(inst))
    return path


def load_instance(path):
    return json.loads(Path(path).read_text())


def normalized_weights(inst):
    """Weights divided by max(w_raw). Use these for the cost function AND the Hamiltonian."""
    return np.asarray(inst["weights"], dtype=np.float64)


# ---------------------------------------------------------------- conversions

def bitstring_to_spins(bitstring):
    return 1 - 2 * np.array([int(b) for b in bitstring], dtype=np.int8)


def spins_to_bitstring(z):
    return "".join("0" if s == 1 else "1" for s in z)


def index_to_bitstring(k, n):
    """Integer k -> bitstring with bit i of k at position i (qubit 0 first)."""
    return "".join(str((k >> i) & 1) for i in range(n))


def bitstring_to_index(bitstring):
    """Inverse of index_to_bitstring: bitstring[i] is bit i of the index.

    Every CUDA-Q bitstring goes through here (and bitstring_to_spins), so if
    E0 shows CUDA-Q prints qubits in the other order, it is fixed in one place.
    """
    return sum(int(b) << i for i, b in enumerate(bitstring))


# ---------------------------------------------------------------- cost

def _edge_array(inst):
    return np.asarray(inst["edges"], dtype=np.int64).reshape(-1, 2)


def cost(inst, z):
    """C(z) for one spin vector (shape n) or a batch (shape m x n). Uses normalized weights."""
    z = np.asarray(z)
    w = normalized_weights(inst)
    e = _edge_array(inst)
    balance = (z @ w) ** 2
    conflict = (z[..., e[:, 0]] * z[..., e[:, 1]]).sum(axis=-1)
    return balance + inst["lambda"] * conflict


def violations(inst, z):
    """Number of conflict edges whose two jobs sit on the same node."""
    z = np.asarray(z)
    e = _edge_array(inst)
    return (z[..., e[:, 0]] == z[..., e[:, 1]]).sum(axis=-1)


# ---------------------------------------------------------------- Hamiltonian

def _couplings(inst):
    """Unscaled J_ij = 2 w_i w_j + lambda * [(i,j) in E], upper triangle (i < j) only."""
    w = normalized_weights(inst)
    J = np.triu(2 * np.outer(w, w), k=1)
    for i, j in inst["edges"]:
        J[i, j] += inst["lambda"]
    return J


def hamiltonian_scale(inst):
    """max |J_ij|; dividing by it puts the largest coupling at exactly 1."""
    J = _couplings(inst)
    return float(np.abs(J).max()) if inst["n"] > 1 else 1.0


def build_hamiltonian(inst):
    """Scaled Ising form of the cost: H(z) = C(z) / scale = offset + sum_{i<j} J_ij z_i z_j.

    terms is a list of (i, j, J_ij) with max |J_ij| = 1; offset = sum(w^2) / scale.
    The offset never changes which schedule is best, but is needed to turn
    energies back into costs: C = scale * H.
    """
    scale = hamiltonian_scale(inst)
    J = _couplings(inst) / scale
    rows, cols = np.nonzero(J)
    return {
        "offset": float((normalized_weights(inst) ** 2).sum() / scale),
        "terms": [(int(i), int(j), float(J[i, j])) for i, j in zip(rows, cols)],
        "scale": scale,
    }


def ising_energy(ham, z):
    """Classically evaluate H(z) for one spin vector or a batch (m x n)."""
    z = np.asarray(z)
    energy = np.full(z.shape[:-1], ham["offset"])
    for i, j, Jij in ham["terms"]:
        energy = energy + Jij * z[..., i] * z[..., j]
    return energy


# ---------------------------------------------------------------- brute force

def brute_force(inst, chunk_bits=20, return_costs=False, tol=1e-9):
    """Exact solve by scoring all 2^n assignments, in chunks to bound memory.

    Returns c_min, c_max, and every optimal bitstring. Optima come in mirror
    pairs (flipping all spins keeps the cost), and integer weights can give
    several distinct optimal pairs, so P(opt) should sum over all of them.
    """
    n = inst["n"]
    if n > BRUTE_FORCE_MAX_N:
        raise ValueError(f"brute force limited to n <= {BRUTE_FORCE_MAX_N}, got {n}")

    total = 1 << n
    chunk = min(total, 1 << chunk_bits)
    shifts = np.arange(n, dtype=np.int64)
    all_costs = np.empty(total) if return_costs else None

    c_min, c_max = np.inf, -np.inf
    opt_idx = np.empty(0, dtype=np.int64)
    max_idx = 0

    for start in range(0, total, chunk):
        idx = np.arange(start, min(start + chunk, total), dtype=np.int64)
        bits = (idx[:, None] >> shifts) & 1
        z = (1 - 2 * bits).astype(np.int8)
        c = cost(inst, z)
        if return_costs:
            all_costs[start:start + len(idx)] = c

        lo = c.min()
        if lo < c_min - tol:
            c_min = lo
            opt_idx = idx[c <= lo + tol]
        elif lo <= c_min + tol:
            opt_idx = np.concatenate([opt_idx, idx[c <= c_min + tol]])

        hi_pos = int(c.argmax())
        if c[hi_pos] > c_max:
            c_max, max_idx = c[hi_pos], int(idx[hi_pos])

    opt_states = [index_to_bitstring(int(k), n) for k in opt_idx]
    result = {
        "c_min": float(c_min),
        "c_max": float(c_max),
        "opt_states": opt_states,
        "opt_violations": int(violations(inst, bitstring_to_spins(opt_states[0]))),
        "max_state": index_to_bitstring(max_idx, n),
    }
    if return_costs:
        result["costs"] = all_costs
    return result


def c_max_exact(inst):
    """Worst cost, valid for any n: all jobs on one node maximises both terms at once."""
    return float(normalized_weights(inst).sum() ** 2 + inst["lambda"] * len(inst["edges"]))


# ---------------------------------------------------------------- metrics
#
# ref is a dict with c_min, c_max and opt_states: brute_force() for n <= 24,
# CP-SAT best-known (with c_max_exact) for n > 24.
# counts is a CUDA-Q style {bitstring: count} dict.

def approx_ratio(c, c_min, c_max):
    """r = (C_max - C) / (C_max - C_min): 1 at the optimum, 0 at the worst schedule.

    Use it on <C> (from observe, converted with C = scale * <H>) and on the best sample.
    """
    return (c_max - c) / (c_max - c_min)


def p_opt(counts, opt_states):
    """Total sampled probability of all optimal bitstrings (mirror pairs included)."""
    shots = sum(counts.values())
    return sum(counts.get(s, 0) for s in opt_states) / shots


def makespan(inst, z):
    """Finish time of the busier node in raw units: (W_raw + |D_raw|) / 2."""
    z = np.asarray(z)
    w = np.asarray(inst["weights_raw"], dtype=np.float64)
    return (w.sum() + np.abs(z @ w)) / 2


def tts99(t_run, p, target=0.99):
    """Time to reach the optimum at least once with probability `target`."""
    if p >= target:
        return t_run
    if p <= 0:
        return np.inf
    return t_run * np.log(1 - target) / np.log(1 - p)


def sample_metrics(inst, counts, ref):
    """All per-run metrics from one sampling run; keys match the results CSV."""
    states = list(counts)
    freq = np.array([counts[s] for s in states], dtype=np.float64)
    z = np.array([bitstring_to_spins(s) for s in states])
    c = cost(inst, z)
    mean_cost = float((freq * c).sum() / freq.sum())
    best = int(c.argmin())
    return {
        "mean_cost": mean_cost,
        "approx_ratio": float(approx_ratio(mean_cost, ref["c_min"], ref["c_max"])),
        "best_approx_ratio": float(approx_ratio(c[best], ref["c_min"], ref["c_max"])),
        "p_opt": p_opt(counts, ref["opt_states"]),
        "best_state": states[best],
        "best_cost": float(c[best]),
        "violations": int(violations(inst, z[best])),
        "makespan": float(makespan(inst, z[best])),
    }


# ---------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description="Generate instances and solve them by brute force.")
    parser.add_argument("--n", type=int, nargs="+", default=[8])
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--lam", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--out", default="instances")
    parser.add_argument("--solve", action="store_true", help="also brute-force each instance")
    args = parser.parse_args()

    for n in args.n:
        for inst in generate_instances(n, args.count, args.base_seed, args.lam):
            path = save_instance(inst, args.out)
            line = f"{path}  weights_raw={inst['weights_raw']}  edges={len(inst['edges'])}"
            if args.solve:
                bf = brute_force(inst)
                line += (f"  C_min={bf['c_min']:.4f}  C_max={bf['c_max']:.4f}"
                         f"  #opt={len(bf['opt_states'])}  viol={bf['opt_violations']}")
            print(line)


if __name__ == "__main__":
    main()
