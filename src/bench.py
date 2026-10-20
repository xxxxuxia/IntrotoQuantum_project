"""Experiment runner: CSV logging, checkpointing, and backend target switching.

Backend-agnostic by design: it knows nothing about problem.py or baselines.py,
just task dicts and a runner callable. Rows are appended to disk immediately
after each run, and a task already present in the CSV (matched on KEY_FIELDS)
is skipped on the next pass, so a crashed sweep resumes instead of redoing
finished work.
"""

import csv
import warnings
from pathlib import Path

CSV_FIELDS = [
    "exp", "instance_id", "n", "p", "init", "optimizer", "target", "precision",
    "noise", "energy", "approx_ratio", "p_opt", "best_cost", "violations",
    "makespan", "n_evals", "wall_time_s", "seed",
]

KEY_FIELDS = [
    "exp", "instance_id", "n", "p", "init", "optimizer", "target", "precision",
    "noise", "seed",
]


def set_target(name):
    """Switch the CUDA-Q simulation backend (e.g. 'qpp-cpu', 'nvidia').

    No-ops with a warning if cudaq isn't installed (e.g. on Apple Silicon,
    where CUDA-Q doesn't run natively) or if name is falsy, so CPU-only
    baseline work is never blocked. Returns the target actually in effect.
    """
    if not name or str(name).lower() in ("none", "n/a", "cpu-only"):
        return "n/a"
    try:
        import cudaq
    except ImportError:
        warnings.warn(f"cudaq not installed; skipping set_target({name!r}).")
        return "n/a"
    cudaq.set_target(name)
    return name


def _row_key(row, key_fields=KEY_FIELDS):
    return tuple(str(row.get(k, "")) for k in key_fields)


def load_done_keys(csv_path, key_fields=KEY_FIELDS):
    path = Path(csv_path)
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        return {_row_key(row, key_fields) for row in csv.DictReader(f)}


def append_row(csv_path, row, fields=CSV_FIELDS):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})
        f.flush()


def run_sweep(tasks, runner, csv_path, key_fields=KEY_FIELDS, fields=CSV_FIELDS):
    """Run `runner(task)` for each task not already checkpointed in csv_path.

    `runner` takes a task dict and returns a dict of result fields, merged
    with the task dict into one CSV row. Returns the list of rows actually
    run (skipped/already-done tasks are not included).
    """
    done = load_done_keys(csv_path, key_fields)
    new_rows = []
    for task in tasks:
        key = _row_key(task, key_fields)
        if key in done:
            print(f"skip (done): {key}")
            continue
        result = runner(task)
        row = {**task, **result}
        append_row(csv_path, row, fields)
        done.add(key)
        new_rows.append(row)
        print(f"done: {key}  wall_time_s={result.get('wall_time_s')}")
    return new_rows
