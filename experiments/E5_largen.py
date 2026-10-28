"""E5: large-n QAOA sampling with transferred angles, no optimisation (P3).

At n = 20..28 a fresh COBYLA optimisation loop is too slow to run, so this
samples directly from angles P1 already optimised and exported for a smaller
size (results/E1_angles.json, "by_n"."<source-n>") -- the SAME angles at
every target size, just re-run through Q.sample() with 10k shots each time.
There is no optimize() call anywhere in this script; that absence is the
whole point of "transferred" vs E1's own per-size optimisation.

Reference optima for approx_ratio: brute_force (n <= 24) or CP-SAT (n > 24),
both via baselines.reference_optimum -- same convention as E3 and E4.

    python experiments/E5_largen.py                          # p = 1..6, n = 20..28, nvidia
    python experiments/E5_largen.py --sizes 20 24 28 --p 6    # quick check, deepest angles only
    python experiments/E5_largen.py --target nvidia-fp64
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bench as BM
import baselines as B
import problem as P
import qaoa as Q

SOURCE_N = 20  # largest size P1 actually optimised in E1; angles transfer from here
SIZES = [20, 22, 24, 26, 28]
INSTANCES = 10
SHOTS = 10_000


def load_angles(angles_path, source_n):
    angles = json.loads(Path(angles_path).read_text())
    by_n = angles["by_n"].get(str(source_n))
    if by_n is None:
        raise RuntimeError(f"no angles for n={source_n} in {angles_path}; "
                            f"available sizes: {sorted(angles['by_n'])}")
    return by_n  # {"1": {"gammas": [...], "betas": [...], "count": k}, "2": {...}, ...}


def params_for(by_n, p):
    if str(p) not in by_n:
        raise RuntimeError(f"no depth p={p} in the transferred angles; "
                            f"available depths: {sorted(int(k) for k in by_n)}")
    rec = by_n[str(p)]
    return rec["gammas"] + rec["betas"]


def build_tasks(sizes, instances, ps, target, source_n):
    precision = "fp64" if target == "nvidia-fp64" else "fp32"
    return [{
        "exp": "E5_largen", "instance_id": f"n{n}_s{seed}", "n": n, "p": p,
        "init": f"transfer_n{source_n}", "optimizer": "none", "target": target,
        "precision": precision, "noise": 0, "seed": seed,
    } for n in sizes for seed in range(instances) for p in ps]


def make_runner(by_n):
    def runner(task):
        n, p, seed = task["n"], task["p"], task["seed"]
        inst = P.generate_instance(n, seed)
        ref = B.reference_optimum(inst)
        data = Q.problem_data(inst)
        params = params_for(by_n, p)

        t0 = time.perf_counter()
        e = Q.energy(data, params)
        counts = Q.sample(inst, params, shots=SHOTS, seed=seed, data=data)
        wall = time.perf_counter() - t0

        m = P.sample_metrics(inst, counts, ref)
        print(f"  n={n} p={p} seed={seed}: r={m['approx_ratio']:.3f} p_opt={m['p_opt']:.3f}")
        return {
            "energy": e, "approx_ratio": m["approx_ratio"], "p_opt": m["p_opt"],
            "best_cost": m["best_cost"], "violations": m["violations"],
            "makespan": m["makespan"], "n_evals": 0, "wall_time_s": wall,
        }
    return runner


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument("--p", type=int, nargs="+", default=list(range(1, 7)),
                         help="depths to sample at (must exist in E1_angles.json)")
    parser.add_argument("--instances", type=int, default=INSTANCES)
    parser.add_argument("--source-n", type=int, default=SOURCE_N,
                         help="which E1 size's optimised angles to transfer from")
    parser.add_argument("--target", default="nvidia", choices=["nvidia", "nvidia-fp64"])
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    by_n = load_angles(out_dir / "E1_angles.json", args.source_n)

    if args.target == "nvidia-fp64":
        import cudaq
        cudaq.set_target("nvidia", option="fp64")
    else:
        BM.set_target(args.target)

    tasks = build_tasks(args.sizes, args.instances, args.p, args.target, args.source_n)
    rows = BM.run_sweep(tasks, make_runner(by_n), out_dir / "E5_largen.csv")
    print(f"{len(rows)} new rows -> {out_dir / 'E5_largen.csv'}")


if __name__ == "__main__":
    main()
