"""E3: classical baselines (KK, LPT, SA, CP-SAT) timing + quality, n = 8-28.

Runs all four baselines on the same instances as E1 (seeds 0-9) for n <= 20,
and on one instance (seed 0) per size for n = 21-28, where the brute-force
reference is too slow to repeat. Results (cost/violations/wall_time_s/
approx_ratio) are checkpointed to results/baselines_timing.csv. Re-running this script skips rows already on
disk (see bench.run_sweep).

Ground truth for approx_ratio comes from baselines.reference_optimum, which
uses problem.brute_force up to n=24 (exhaustive) and CP-SAT above that (exact,
proven optimal for every size tested up to 30 during development -- see
tests/test_baselines.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import baselines as B
import bench as BM
import problem as P

ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "results" / "baselines_timing.csv"
SIZES = range(8, 29)
MULTI_SEED_MAX_N = 20
INSTANCES = 10
OPTIMIZERS = ("KK", "LPT", "SA", "CPSAT")

_ref_cache = {}


def _reference_cached(inst):
    if inst["id"] not in _ref_cache:
        _ref_cache[inst["id"]] = B.reference_optimum(inst)
    return _ref_cache[inst["id"]]


def build_tasks():
    tasks = []
    for n in SIZES:
        for seed in range(INSTANCES if n <= MULTI_SEED_MAX_N else 1):
            for optimizer in OPTIMIZERS:
                tasks.append({
                    "exp": "E3_baselines",
                    "instance_id": f"n{n}_s{seed}",
                    "n": n,
                    "p": "",
                    "init": "",
                    "optimizer": optimizer,
                    "target": "n/a",
                    "precision": "",
                    "noise": "",
                    "seed": seed,
                })
    return tasks


def runner(task):
    inst = P.generate_instance(task["n"], seed=task["seed"])
    ref = _reference_cached(inst)
    c_min, c_max = ref["c_min"], ref["c_max"]

    if task["optimizer"] == "KK":
        r = B.karmarkar_karp(inst)
    elif task["optimizer"] == "LPT":
        r = B.greedy_lpt(inst)
    elif task["optimizer"] == "SA":
        r = B.simulated_annealing(inst, seed=task["seed"])
    else:
        r = B.cp_sat_solve(inst)

    denom = c_max - c_min
    approx_ratio = 1.0 if denom <= 1e-9 else (c_max - r["cost"]) / denom
    z = P.bitstring_to_spins(r["bitstring"])

    return {
        "energy": r["cost"],
        "approx_ratio": approx_ratio,
        "p_opt": "",
        "best_cost": r["cost"],
        "violations": r["violations"],
        "makespan": P.makespan(inst, z),
        "n_evals": "",
        "wall_time_s": r["wall_time_s"],
    }


def main():
    BM.set_target(None)  # classical baselines need no quantum backend
    rows = BM.run_sweep(build_tasks(), runner, RESULTS_CSV)
    print(f"{len(rows)} new rows -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
