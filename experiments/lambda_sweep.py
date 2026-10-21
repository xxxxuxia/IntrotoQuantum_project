"""Pick the conflict penalty lambda (P2, W8).

Rule from the plan: the smallest lambda in LAMBDAS whose brute-force optimum has
zero violations on at least 90 % of instances, at every size tested.
Smaller is better: a large lambda swamps the balance couplings once the
Hamiltonian is scaled to max |J| = 1, which makes the makespan part harder for QAOA.

    python experiments/lambda_sweep.py            # writes results/lambda_sweep.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import problem as P

LAMBDAS = [0.1, 0.5, 1.0, 2.0, 5.0]
THRESHOLD = 0.9


def run(sizes, count, base_seed):
    rows = []
    for n in sizes:
        for k in range(count):
            seed = base_seed + k
            # lambda does not touch the RNG, so every lambda sees the same weights and edges
            free = P.brute_force(P.generate_instance(n, seed, lam=0.0))
            free_makespan = P.makespan(P.generate_instance(n, seed),
                                       P.bitstring_to_spins(free["opt_states"][0]))
            for lam in LAMBDAS:
                inst = P.generate_instance(n, seed, lam=lam)
                bf = P.brute_force(inst)
                spins = [P.bitstring_to_spins(s) for s in bf["opt_states"]]
                opt_makespan = P.makespan(inst, spins[0])
                rows.append({
                    "lambda": lam,
                    "n": n,
                    "instance_id": inst["id"],
                    "seed": seed,
                    "edges": len(inst["edges"]),
                    "n_opt": len(bf["opt_states"]),
                    "violations": max(int(P.violations(inst, z)) for z in spins),
                    "makespan": opt_makespan,
                    "makespan_vs_unconstrained": opt_makespan / free_makespan,
                    "min_abs_J": min(abs(J) for *_, J in P.build_hamiltonian(inst)["terms"]),
                })
    return rows


def summarise(rows, sizes):
    print(f"{'lambda':>7} | " + " ".join(f"n={n:<5}" for n in sizes) + " | all sizes >= 90%?")
    chosen = None
    for lam in LAMBDAS:
        fracs = []
        for n in sizes:
            v = [r["violations"] for r in rows if r["lambda"] == lam and r["n"] == n]
            fracs.append(np.mean(np.array(v) == 0))
        ok = all(f >= THRESHOLD for f in fracs)
        if ok and chosen is None:
            chosen = lam
        print(f"{lam:>7} | " + " ".join(f"{f:<7.0%}" for f in fracs) + f" | {'yes' if ok else 'no'}")
    print(f"\nchosen lambda = {chosen}")
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, nargs="+", default=[6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "results" / "lambda_sweep.csv"))
    args = parser.parse_args()

    rows = run(args.n, args.count, args.base_seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}\n")
    summarise(rows, args.n)


if __name__ == "__main__":
    main()
