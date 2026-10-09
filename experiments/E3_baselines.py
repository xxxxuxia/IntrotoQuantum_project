"""E3 (preliminary, W8): first CPU timings for the classical baselines, n = 10-24.

Generates one fixed-seed instance per size, runs KK, greedy LPT and SA on
each, and checkpoints cost/violations/wall_time_s/approx_ratio to
results/baselines_timing.csv. Re-running this script skips rows already on
disk (see bench.run_sweep).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import baselines as B
import bench as BM
import problem as P

ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "results" / "baselines_timing.csv"
BASE_SEED = 0
SIZES = range(10, 25)

_bf_cache = {}


def _brute_force_cached(inst):
    if inst["id"] not in _bf_cache:
        _bf_cache[inst["id"]] = P.brute_force(inst)
    return _bf_cache[inst["id"]]


def build_tasks():
    tasks = []
    for n in SIZES:
        for optimizer in ("KK", "LPT", "SA"):
            tasks.append({
                "exp": "E3_baselines",
                "instance_id": f"n{n}_s{BASE_SEED}",
                "n": n,
                "p": "",
                "init": "",
                "optimizer": optimizer,
                "target": "n/a",
                "precision": "",
                "noise": "",
                "seed": BASE_SEED,
            })
    return tasks


def runner(task):
    inst = P.generate_instance(task["n"], seed=task["seed"])
    bf = _brute_force_cached(inst)
    c_min, c_max = bf["c_min"], bf["c_max"]

    if task["optimizer"] == "KK":
        r = B.karmarkar_karp(inst)
    elif task["optimizer"] == "LPT":
        r = B.greedy_lpt(inst)
    else:
        r = B.simulated_annealing(inst, seed=task["seed"])

    denom = c_max - c_min
    approx_ratio = 1.0 if denom <= 1e-9 else (c_max - r["cost"]) / denom

    return {
        "energy": r["cost"],
        "approx_ratio": approx_ratio,
        "p_opt": "",
        "best_cost": c_min,
        "violations": r["violations"],
        "n_evals": "",
        "wall_time_s": r["wall_time_s"],
    }


def main():
    BM.set_target(None)  # classical baselines need no quantum backend
    rows = BM.run_sweep(build_tasks(), runner, RESULTS_CSV)
    print(f"{len(rows)} new rows -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
