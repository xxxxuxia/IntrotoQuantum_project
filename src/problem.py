"""Problem definition: instance generator, cost function and brute-force solver.

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
DEFAULT_LAMBDA = 1.0  # placeholder until the W8 lambda sweep fixes it
BRUTE_FORCE_MAX_N = 24


# ---------------------------------------------------------------- instances

def generate_instance(n, seed, lam=DEFAULT_LAMBDA, edge_prob=EDGE_PROB,
                      mu=LOGNORMAL_MU, sigma=LOGNORMAL_SIGMA):
    """One random instance: lognormal integer runtimes + Erdos-Renyi conflict graph."""
    rng = np.random.default_rng(seed)
    weights = np.maximum(1, np.rint(rng.lognormal(mu, sigma, size=n))).astype(int)
    edges = [[i, j] for i in range(n) for j in range(i + 1, n)
             if rng.random() < edge_prob]
    return {
        "id": f"n{n}_s{seed}",
        "n": n,
        "weights": weights.tolist(),
        "edges": edges,
        "lambda": lam,
        "seed": seed,
    }


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
    """Weights divided by max(w). Use these for the cost function AND the Hamiltonian."""
    w = np.asarray(inst["weights"], dtype=np.float64)
    return w / w.max()


# ---------------------------------------------------------------- conversions

def bitstring_to_spins(bitstring):
    return 1 - 2 * np.array([int(b) for b in bitstring], dtype=np.int8)


def spins_to_bitstring(z):
    return "".join("0" if s == 1 else "1" for s in z)


def _index_to_bitstring(k, n):
    return "".join(str((k >> i) & 1) for i in range(n))


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

    opt_states = [_index_to_bitstring(int(k), n) for k in opt_idx]
    result = {
        "c_min": float(c_min),
        "c_max": float(c_max),
        "opt_states": opt_states,
        "opt_violations": int(violations(inst, bitstring_to_spins(opt_states[0]))),
        "max_state": _index_to_bitstring(max_idx, n),
    }
    if return_costs:
        result["costs"] = all_costs
    return result


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
            line = f"{path}  weights={inst['weights']}  edges={len(inst['edges'])}"
            if args.solve:
                bf = brute_force(inst)
                line += (f"  C_min={bf['c_min']:.4f}  C_max={bf['c_max']:.4f}"
                         f"  #opt={len(bf['opt_states'])}  viol={bf['opt_violations']}")
            print(line)


if __name__ == "__main__":
    main()
