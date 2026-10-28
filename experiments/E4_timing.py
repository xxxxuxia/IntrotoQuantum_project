"""E4: CPU (qpp-cpu) vs GPU (nvidia) vs GPU fp64 -- time & memory (P3).

One depth (p = P_FIXED) is held fixed so only the *target* changes; wall-time
and peak memory are then comparable like-for-like across backends. Each
target is a separate process (own --target flag, same pattern as
E1_depth.py), so e.g. qpp-cpu can run on one machine and nvidia on another --
both append to the same CSV and are checkpointed by bench.run_sweep.

qpp-cpu is capped at n <= CPU_MAX_N: a single COBYLA run already takes
minutes per E1_depth.py's own timing notes, and the point of this experiment
is to show where CPU falls off a cliff, not to wait it out past that point.

GPU memory is read via pynvml if it's installed; otherwise peak_mem_mb is
left blank rather than failing the run (pynvml is optional, not a hard dep).

    python experiments/E4_timing.py --target qpp-cpu
    python experiments/E4_timing.py --target nvidia
    python experiments/E4_timing.py --target nvidia-fp64
    python experiments/E4_timing.py --target nvidia --sizes 10 14 18 --instances 1  # quick test
"""

import argparse
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bench as BM
import baselines as B
import problem as P
import qaoa as Q

P_FIXED = 3
CPU_MAX_N = 20
SIZES = list(range(10, 31, 2))  # 10, 12, ..., 30
INSTANCES = 3
PRECISION = {"qpp-cpu": "fp64", "nvidia": "fp32", "nvidia-fp64": "fp64"}
FIELDS = BM.CSV_FIELDS + ["peak_mem_mb"]


def set_target_for(name):
    if name == "nvidia-fp64":
        import cudaq
        cudaq.set_target("nvidia", option="fp64")
    else:
        BM.set_target(name)


def gpu_mem_mb():
    """Used GPU memory in MB via pynvml, or None if pynvml/the GPU isn't there."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.used / (1024 ** 2)
    except Exception:
        return None


def cpu_mem_mb():
    """Peak RSS of this process in MB (ru_maxrss is KB on Linux, bytes on macOS)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024 if sys.platform.startswith("linux") else rss / (1024 ** 2)


def build_tasks(sizes, instances, target):
    sizes = [n for n in sizes if target != "qpp-cpu" or n <= CPU_MAX_N]
    return [{
        "exp": "E4_timing", "instance_id": f"n{n}_s{seed}", "n": n, "p": P_FIXED,
        "init": "ramp", "optimizer": "COBYLA", "target": target,
        "precision": PRECISION.get(target, ""), "noise": 0, "seed": seed,
    } for n in sizes for seed in range(instances)]


def make_runner(target):
    def runner(task):
        n, seed = task["n"], task["seed"]
        inst = P.generate_instance(n, seed)
        ref = B.reference_optimum(inst)
        data = Q.problem_data(inst)
        _, row = Q.run(inst, P_FIXED, ref, init_params=Q.ramp_init(P_FIXED),
                        seed=seed, data=data)
        row["peak_mem_mb"] = gpu_mem_mb() if target != "qpp-cpu" else cpu_mem_mb()
        print(f"  n={n} seed={seed} target={target}: "
              f"wall={row['wall_time_s']:.2f}s mem={row['peak_mem_mb']}")
        return row
    return runner


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", required=True, choices=["qpp-cpu", "nvidia", "nvidia-fp64"])
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument("--instances", type=int, default=INSTANCES)
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_target_for(args.target)

    tasks = build_tasks(args.sizes, args.instances, args.target)
    if not tasks:
        print(f"no sizes <= {CPU_MAX_N} in {args.sizes} for target={args.target}; nothing to run")
        return

    rows = BM.run_sweep(tasks, make_runner(args.target), out_dir / "E4_timing.csv",
                         fields=FIELDS)
    print(f"{len(rows)} new rows -> {out_dir / 'E4_timing.csv'}")


if __name__ == "__main__":
    main()
