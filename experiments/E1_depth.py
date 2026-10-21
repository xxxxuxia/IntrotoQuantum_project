"""E1: QAOA depth sweep p = 1..6 with INTERP warm starts (P1).

For every instance, p = 1 starts from qaoa.ramp_init and each p + 1 starts from
qaoa.interp_init of the optimised depth-p angles, so the p values of one
instance must run in order. Rows go to results/E1_depth.csv through
bench.run_sweep; the optimised angles of every run go to results/E1_params.jsonl
so a resumed sweep can warm-start from runs finished before the crash.

At the end the median angles per p are written to results/E1_angles.json
(pooled over all instances, and per n) for E5's transferred-angle runs.

    python experiments/E1_depth.py --sizes 8 10 12                       # CPU (laptop)
    python experiments/E1_depth.py --sizes 14 16 18 20 --target nvidia   # GPU
    python experiments/E1_depth.py --sizes 8 --instances 2 --p-max 3     # quick test

On qpp-cpu one energy evaluation takes ~0.7 s at n = 14 and ~4 s at n = 16, and
a run needs a few hundred, so sizes above 12 belong on the GPU. Both halves
append to the same CSV and jsonl, so they can be run on different machines and
the files concatenated (header once) before the angle export.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bench as BM
import problem as P
import qaoa as Q

PRECISION = {"qpp-cpu": "fp64", "nvidia": "fp32", "nvidia-fp64": "fp64"}


def load_params(path):
    """{(instance_id, p): params} from the jsonl log; later lines win."""
    params = {}
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            params[(rec["instance_id"], rec["p"])] = rec["params"]
    return params


def append_params(path, instance_id, n, p, params):
    with path.open("a") as f:
        f.write(json.dumps({"instance_id": instance_id, "n": n, "p": p,
                            "params": [float(x) for x in params]}) + "\n")


def build_tasks(sizes, instances, p_max, target):
    # p is the innermost loop so each instance's depths run in order.
    return [{
        "exp": "E1_depth", "instance_id": f"n{n}_s{seed}", "n": n, "p": p,
        "init": "interp", "optimizer": "COBYLA", "target": target,
        "precision": PRECISION.get(target, ""), "noise": 0, "seed": seed,
    } for n in sizes for seed in range(instances) for p in range(1, p_max + 1)]


def make_runner(params_path, shots):
    params = load_params(params_path)
    cache = {}

    def runner(task):
        n, seed, p = task["n"], task["seed"], task["p"]
        if (n, seed) not in cache:
            inst = P.generate_instance(n, seed)
            cache[(n, seed)] = (inst, P.brute_force(inst), Q.problem_data(inst))
        inst, ref, data = cache[(n, seed)]

        prev = params.get((inst["id"], p - 1))
        if p > 1 and prev is None:
            raise RuntimeError(f"{inst['id']} p={p}: no p={p - 1} angles to warm-start from")
        init = None if p == 1 else Q.interp_init(prev)

        x, row = Q.run(inst, p, ref, init_params=init, shots=shots, seed=seed, data=data)
        params[(inst["id"], p)] = x.tolist()
        append_params(params_path, inst["id"], n, p, x)
        print(f"  {inst['id']} p={p}: r={row['approx_ratio']:.3f} "
              f"p_opt={row['p_opt']:.3f} evals={row['n_evals']}")
        return {k: row[k] for k in ("energy", "approx_ratio", "p_opt", "best_cost",
                                    "violations", "makespan", "n_evals", "wall_time_s")}

    return runner


def export_median_angles(params_path, out_path):
    pooled, by_n = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for line in params_path.read_text().splitlines():
        rec = json.loads(line)
        # (gamma, beta) and (-gamma, -beta) give the same probabilities (H is real),
        # so put every run on the gamma_1 >= 0 branch before taking medians.
        x = np.asarray(rec["params"])
        x = -x if x[0] < 0 else x
        pooled[rec["p"]].append(x)
        by_n[rec["n"]][rec["p"]].append(x)

    def medians(groups):
        out = {}
        for p, runs in sorted(groups.items()):
            med = np.median(np.array(runs), axis=0)
            out[str(p)] = {"gammas": med[:p].tolist(), "betas": med[p:].tolist(),
                           "count": len(runs)}
        return out

    angles = {"pooled": medians(pooled),
              "by_n": {str(n): medians(g) for n, g in sorted(by_n.items())}}
    out_path.write_text(json.dumps(angles, indent=2) + "\n")
    print(f"median angles for p = {', '.join(angles['pooled'])} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 10, 12, 14, 16, 18, 20])
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument("--p-max", type=int, default=6)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--target", default="qpp-cpu", help="qpp-cpu, nvidia, ...")
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "E1_params.jsonl"

    if args.target == "nvidia-fp64":
        import cudaq
        cudaq.set_target("nvidia", option="fp64")
    else:
        BM.set_target(args.target)

    tasks = build_tasks(args.sizes, args.instances, args.p_max, args.target)
    rows = BM.run_sweep(tasks, make_runner(params_path, args.shots), out_dir / "E1_depth.csv")
    print(f"{len(rows)} new rows -> {out_dir / 'E1_depth.csv'}")
    export_median_angles(params_path, out_dir / "E1_angles.json")


if __name__ == "__main__":
    main()
