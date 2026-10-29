"""E6: how much does gate noise degrade QAOA? (P2, written with P1)

Angles are optimised noiselessly (taken from E1 where available, otherwise an
INTERP chain is run here) and then evaluated under depolarising noise without
re-optimising, i.e. "trained in simulation, deployed on a noisy device".

Noise model, with p_err the two-qubit error rate (the dominant one on hardware):
    CX          two-qubit depolarising, p_err
    H, RX, RZ   single-qubit depolarising, p_err / 10

The density-matrix simulator (density-matrix-cpu) is exact, so <C> and P(opt)
are read off the diagonal of rho instead of being estimated from shots; the
10k-shot sample is drawn from that diagonal only for the best-sample metrics.
Every p_err = 0 row is checked against the noiseless observe value.

Cost: one noisy simulation at n = 8, p = 4 takes ~45 s on a laptop CPU and
n = 10 is ~15x more, so the default sizes are n = 6, 8.

    python experiments/E6_noise.py                                  # n = 6, 8
    python experiments/E6_noise.py --sizes 6 --instances 2          # quick test
"""

import argparse
import sys
import time
from pathlib import Path

import cudaq
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import problem as P
import qaoa as Q
import bench as BM
from E1_depth import append_params, load_params

NOISE_LEVELS = [0.0, 0.001, 0.005, 0.01]
SHOTS = 10_000

# cudaq.set_noise keeps only a pointer: a model that Python garbage-collects
# while it is set crashes the simulator (SIGBUS), so every model lives here.
_noise_models = {}


def noise_model(p_err):
    if p_err not in _noise_models:
        nm = cudaq.NoiseModel()
        for gate in ("h", "rx", "rz"):
            nm.add_all_qubit_channel(gate, cudaq.DepolarizationChannel(p_err / 10))
        nm.add_all_qubit_channel("x", cudaq.Depolarization2(p_err), num_controls=1)
        _noise_models[p_err] = nm
    return _noise_models[p_err]


def probabilities(data, params, p_err):
    """Diagonal of the (noisy) density matrix, in problem.index_to_bitstring order."""
    if p_err > 0:
        cudaq.set_noise(noise_model(p_err))
    try:
        rho = np.array(cudaq.get_state(Q.qaoa_kernel, *Q._args(data, params)))
    finally:
        cudaq.unset_noise()
    return np.clip(np.real(np.diag(rho)), 0, None)


def build_tasks(sizes, instances, p_max, levels):
    # p_err = 0 first, so each instance's noiseless check runs before its noisy rows.
    return [{
        "exp": "E6_noise", "instance_id": f"n{n}_s{seed}", "n": n, "p": p,
        "init": "interp", "optimizer": "COBYLA", "target": "density-matrix-cpu",
        "precision": "fp64", "noise": p_err, "seed": seed,
    } for n in sizes for seed in range(instances) for p_err in levels
        for p in range(1, p_max + 1)]


def make_runner(params_path):
    params = {**load_params(ROOT / "results" / "E1_params.jsonl"), **load_params(params_path)}
    cache = {}

    def angles(inst, data, p):
        """Noiseless optimised angles: from E1 if present, else INTERP here (and logged)."""
        key = (inst["id"], p)
        if key not in params:
            init = None if p == 1 else Q.interp_init(angles(inst, data, p - 1))
            x, _, _ = Q.optimize(inst, p, init, data=data)
            params[key] = x.tolist()
            append_params(params_path, inst["id"], inst["n"], p, x)
        return params[key]

    def runner(task):
        n, seed, p, p_err = task["n"], task["seed"], task["p"], float(task["noise"])
        if (n, seed) not in cache:
            inst = P.generate_instance(n, seed)
            ref = P.brute_force(inst, return_costs=True)
            cache[(n, seed)] = (inst, ref, Q.problem_data(inst))
        inst, ref, data = cache[(n, seed)]
        x = angles(inst, data, p)

        t0 = time.perf_counter()
        probs = probabilities(data, x, p_err)
        wall = time.perf_counter() - t0

        mean_c = float(probs @ ref["costs"])
        if p_err == 0 and not np.isclose(mean_c, Q.expected_cost(inst, x, data)):
            raise RuntimeError(f"{inst['id']} p={p}: noiseless density matrix disagrees with observe")
        opt_idx = [P.bitstring_to_index(s) for s in ref["opt_states"]]
        rng = np.random.default_rng(seed)
        draws = rng.multinomial(SHOTS, probs / probs.sum())
        counts = {P.index_to_bitstring(k, n): int(c) for k, c in enumerate(draws) if c}
        m = P.sample_metrics(inst, counts, ref)

        row = {"energy": mean_c / data["scale"],
               "approx_ratio": float(P.approx_ratio(mean_c, ref["c_min"], ref["c_max"])),
               "p_opt": float(probs[opt_idx].sum()),
               "best_cost": m["best_cost"], "violations": m["violations"],
               "makespan": m["makespan"], "n_evals": "", "wall_time_s": wall}
        print(f"  {inst['id']} p={p} p_err={p_err}: r={row['approx_ratio']:.3f} "
              f"p_opt={row['p_opt']:.4f} ({wall:.1f}s)")
        return row

    return runner


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument("--p-max", type=int, default=4)
    parser.add_argument("--noise", type=float, nargs="+", default=NOISE_LEVELS)
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "E6"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cudaq.set_target("density-matrix-cpu")

    tasks = build_tasks(args.sizes, args.instances, args.p_max, sorted(args.noise))
    rows = BM.run_sweep(tasks, make_runner(out_dir / "E6_params.jsonl"), out_dir / "E6_noise.csv")
    print(f"{len(rows)} new rows -> {out_dir / 'E6_noise.csv'}")


if __name__ == "__main__":
    main()
