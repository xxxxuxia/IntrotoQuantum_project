"""E2: how should QAOA's angles be initialised? (P2, written with P1)

Four strategies on the same instances, same target, same COBYLA settings:

    interp          p = 1 from qaoa.ramp_init, then INTERP p -> p + 1 (as in E1)
    random5         best of 5 qaoa.random_init starts (n_evals and wall time summed)
    transfer        start from E1's median angles at n = SOURCE_N, then optimise
    transfer_noopt  E1's median angles at n = SOURCE_N used as-is, no optimisation

SOURCE_N = 8 is smaller than every E2 size, so transferred angles never come
from the instances they are tested on. interp runs every p up to the largest
requested one (it needs the chain); the other strategies run only at --p.

    python experiments/E2_init.py --target nvidia                     # GPU, n = 10..16
    python experiments/E2_init.py --sizes 10 --instances 2 --p 2      # quick CPU test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bench as BM
import problem as P
import qaoa as Q
from E1_depth import PRECISION, append_params, load_params

SOURCE_N = 8
RANDOM_STARTS = 5
INITS = ["interp", "random5", "transfer", "transfer_noopt"]


def build_tasks(sizes, instances, ps, target):
    tasks = []
    for n in sizes:
        for seed in range(instances):
            for init in INITS:
                depths = range(1, max(ps) + 1) if init == "interp" else ps
                for p in depths:
                    tasks.append({
                        "exp": "E2_init", "instance_id": f"n{n}_s{seed}", "n": n, "p": p,
                        "init": init, "optimizer": "none" if init == "transfer_noopt" else "COBYLA",
                        "target": target, "precision": PRECISION.get(target, ""), "noise": 0,
                        "seed": seed,
                    })
    return tasks


def make_runner(params_path, transfer, shots):
    params = load_params(params_path)
    cache = {}

    def runner(task):
        n, seed, p, init = task["n"], task["seed"], task["p"], task["init"]
        if (n, seed) not in cache:
            inst = P.generate_instance(n, seed)
            cache[(n, seed)] = (inst, P.brute_force(inst), Q.problem_data(inst))
        inst, ref, data = cache[(n, seed)]
        moved = transfer[str(p)]["gammas"] + transfer[str(p)]["betas"]

        if init == "interp":
            prev = params.get((inst["id"], p - 1))
            if p > 1 and prev is None:
                raise RuntimeError(f"{inst['id']} p={p}: no p={p - 1} angles to warm-start from")
            x, row = Q.run(inst, p, ref, init_params=None if p == 1 else Q.interp_init(prev),
                           shots=shots, seed=seed, data=data)
            params[(inst["id"], p)] = x.tolist()
            append_params(params_path, inst["id"], n, p, x)
        elif init == "random5":
            rng = np.random.default_rng(1000 * seed + p)
            runs = [Q.run(inst, p, ref, init_params=Q.random_init(p, rng), shots=shots,
                          seed=seed, data=data)[1] for _ in range(RANDOM_STARTS)]
            row = min(runs, key=lambda r: r["energy"])
            row = {**row, "n_evals": sum(r["n_evals"] for r in runs),
                   "wall_time_s": sum(r["wall_time_s"] for r in runs)}
        elif init == "transfer":
            _, row = Q.run(inst, p, ref, init_params=moved, shots=shots, seed=seed, data=data)
        else:
            t0 = time.perf_counter()
            e = Q.energy(data, moved)
            counts = Q.sample(inst, moved, shots=shots, seed=seed, data=data)
            m = P.sample_metrics(inst, counts, ref)
            row = {"energy": e, "n_evals": 0, "wall_time_s": time.perf_counter() - t0,
                   "approx_ratio": float(P.approx_ratio(data["scale"] * e, ref["c_min"], ref["c_max"])),
                   **{k: m[k] for k in ("p_opt", "best_cost", "violations", "makespan")}}

        print(f"  {inst['id']} p={p} {init}: r={row['approx_ratio']:.3f} "
              f"p_opt={row['p_opt']:.4f} evals={row['n_evals']}")
        return {k: row[k] for k in ("energy", "approx_ratio", "p_opt", "best_cost",
                                    "violations", "makespan", "n_evals", "wall_time_s")}

    return runner


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 12, 14, 16])
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument("--p", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--target", default="qpp-cpu", help="qpp-cpu, nvidia, nvidia-fp64")
    parser.add_argument("--angles", default=str(ROOT / "results" / "E1_angles.json"))
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "E2"))
    args = parser.parse_args()

    transfer = json.loads(Path(args.angles).read_text())["by_n"][str(SOURCE_N)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.target == "nvidia-fp64":
        import cudaq
        cudaq.set_target("nvidia", option="fp64")
    else:
        BM.set_target(args.target)

    tasks = build_tasks(args.sizes, args.instances, args.p, args.target)
    rows = BM.run_sweep(tasks, make_runner(out_dir / "E2_params.jsonl", transfer, args.shots),
                        out_dir / "E2_init.csv")
    print(f"{len(rows)} new rows -> {out_dir / 'E2_init.csv'}")


if __name__ == "__main__":
    main()
