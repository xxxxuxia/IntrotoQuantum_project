"""Turn results/*.csv into report-ready figures. Figures come only from this
script (per the team's own rule) -- never hand-edited.

    python experiments/make_figures.py        # writes figures/*.png
"""

import csv
import glob
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

# Validated categorical palette (dataviz skill, references/palette.md), light mode.
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

SLOTS = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
}

METHOD_COLOR = {
    "KK": SLOTS["blue"],
    "LPT": SLOTS["orange"],
    "SA": SLOTS["aqua"],
    "CPSAT": SLOTS["yellow"],
}
METHOD_LABEL = {"KK": "Karmarkar-Karp", "LPT": "Greedy LPT", "SA": "Simulated annealing", "CPSAT": "CP-SAT (exact)"}
METHOD_ORDER = ["KK", "LPT", "SA", "CPSAT"]


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def _legend(ax):
    return ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9,
                      loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)


def load_baselines():
    rows = list(csv.DictReader(open(ROOT / "results" / "baselines_timing.csv")))
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["optimizer"]].append(r)
    for rs in by_method.values():
        rs.sort(key=lambda r: int(r["n"]))
    return by_method


def _mean_by_n(rows, key, sizes=None):
    """Mean of `key` per n over instances (E3 has 10 per size up to n = 20, 1 above)."""
    by_n = defaultdict(list)
    for r in rows:
        if sizes is None or int(r["n"]) in sizes:
            by_n[int(r["n"])].append(float(r[key]))
    xs = sorted(by_n)
    return xs, [np.mean(by_n[n]) for n in xs]


MULTI_SEED_MAX_N = 20  # E3 has 10 instances per n up to here, 1 above


def _plot_baseline(ax, rows, key, method):
    """Mean per n; the n > 20 single-instance points are dotted and shaded, so
    the jump in noise at n = 21 isn't read as a real trend."""
    x, y = _mean_by_n(rows, key)
    lo = [i for i, n in enumerate(x) if n <= MULTI_SEED_MAX_N]
    hi = [i for i, n in enumerate(x) if n >= MULTI_SEED_MAX_N]
    style = dict(color=METHOD_COLOR[method], marker="o", markersize=5,
                 solid_capstyle="round", solid_joinstyle="round")
    ax.plot([x[i] for i in lo], [y[i] for i in lo], linewidth=2, label=METHOD_LABEL[method], **style)
    ax.plot([x[i] for i in hi], [y[i] for i in hi], linewidth=1.2, linestyle=":", **style)


def _shade_single_instance(ax, max_n):
    ax.axvspan(MULTI_SEED_MAX_N + 0.5, max_n + 0.5, color=GRID, alpha=0.5, linewidth=0)
    ax.text(MULTI_SEED_MAX_N + 0.8, 0.02, "1 instance per n", transform=ax.get_xaxis_transform(),
            fontsize=8, color=INK_MUTED)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))


def fig_approx_ratio(by_method):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for method in METHOD_ORDER:
        _plot_baseline(ax, by_method[method], "approx_ratio", method)
    _shade_single_instance(ax, max(int(r["n"]) for r in by_method["KK"]))
    ax.set_ylim(0.75, 1.03)
    ax.set_xlabel("number of jobs (n)  (mean of 10 instances for n ≤ 20)")
    ax.set_ylabel("approx. ratio (1.0 = optimal)")
    ax.set_title("How close each baseline gets to the true best answer", color=INK, fontsize=11)
    _style_axes(ax)
    _legend(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "baseline_approx_ratio.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_wall_time(by_method):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for method in METHOD_ORDER:
        _plot_baseline(ax, by_method[method], "wall_time_s", method)
    _shade_single_instance(ax, max(int(r["n"]) for r in by_method["KK"]))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}s"))
    ax.set_xlabel("number of jobs (n)  (mean of 10 instances for n ≤ 20)")
    ax.set_ylabel("wall time (log scale)")
    ax.set_title("How long each baseline takes to run", color=INK, fontsize=11)
    _style_axes(ax)
    _legend(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "baseline_wall_time.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_violations(by_method):
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    totals = [np.mean([int(r["violations"]) for r in by_method[m]]) for m in METHOD_ORDER]
    colors = [METHOD_COLOR[m] for m in METHOD_ORDER]
    bars = ax.bar(METHOD_ORDER, totals, color=colors, width=0.55)
    for rect, val in zip(bars, totals):
        ax.annotate(f"{val:.2f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("conflict violations per solution (mean over all instances)")
    ax.set_title("Which baselines actually respect the conflict rule", color=INK, fontsize=11)
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "baseline_violations.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def load_lambda_sweep():
    rows = list(csv.DictReader(open(ROOT / "results" / "lambda_sweep.csv")))
    by_lam_n = defaultdict(list)
    for r in rows:
        by_lam_n[(float(r["lambda"]), int(r["n"]))].append(int(r["violations"]) == 0)
    return by_lam_n


def fig_lambda_sweep(by_lam_n):
    sizes = sorted({n for (_, n) in by_lam_n})
    lambdas = sorted({lam for (lam, _) in by_lam_n})
    # Sizes are ordered, so shade them light (small n) to dark (large n) on one
    # hue instead of cycling the categorical slots, which repeat past 6 sizes.
    ramp = plt.get_cmap("Blues")
    size_colors = [ramp(0.35 + 0.65 * i / max(1, len(sizes) - 1)) for i in range(len(sizes))]
    # Several sizes land on identical fractions at every lambda (e.g. n=8 and
    # n=10 here) and would otherwise sit perfectly hidden under each other --
    # distinct marker shapes keep every series visible even when fully overlapped.
    markers = ["o", "s", "^", "D", "v", "P", "X", "p", "h", "*"]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for i, n in enumerate(sizes):
        y = [sum(by_lam_n[(lam, n)]) / len(by_lam_n[(lam, n)]) for lam in lambdas]
        ax.plot(lambdas, y, color=size_colors[i], linewidth=2,
                 marker=markers[i % len(markers)], markersize=6,
                 markeredgecolor=SURFACE, markeredgewidth=0.8,
                 solid_capstyle="round", solid_joinstyle="round", label=f"n={n}")
    ax.axhline(0.9, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("90% threshold", (lambdas[-1], 0.9), xytext=(-70, 6),
                textcoords="offset points", color=INK_MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(lambdas)
    ax.set_xticklabels([str(l) for l in lambdas])
    ax.set_xlabel("lambda (conflict penalty weight)")
    ax.set_ylabel("fraction of instances with 0 violations at the optimum")
    ax.set_title("Picking lambda: smallest value clearing 90% zero-violation", color=INK, fontsize=11)
    _style_axes(ax)
    _legend(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "lambda_sweep.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def load_e1():
    """{n: {p: [row, ...]}} plus a p = 0 entry per instance: the uniform state |+>^n,
    i.e. random sampling, with <C> = sum(w^2) and P(opt) = #optima / 2^n."""
    sys.path.insert(0, str(ROOT / "src"))
    import problem as P

    rows = list(csv.DictReader(open(ROOT / "results" / "E1_depth.csv")))
    refs = {}
    for r in rows:
        if r["instance_id"] not in refs:
            inst = P.generate_instance(int(r["n"]), int(r["seed"]))
            refs[r["instance_id"]] = (inst, P.brute_force(inst))
    by_n = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ref = refs[r["instance_id"]][1]
        by_n[int(r["n"])][int(r["p"])].append({
            "approx_ratio": float(r["approx_ratio"]), "p_opt": float(r["p_opt"]),
            # best of the 10k shots, the answer QAOA would actually return
            "best_ratio": float(P.approx_ratio(float(r["best_cost"]), ref["c_min"], ref["c_max"]))})
    for inst, ref in refs.values():
        mean_c = (P.normalized_weights(inst) ** 2).sum()
        by_n[inst["n"]][0].append({
            "approx_ratio": float(P.approx_ratio(mean_c, ref["c_min"], ref["c_max"])),
            "p_opt": len(ref["opt_states"]) / 2 ** inst["n"],
        })
    return by_n


def _fig_e1(by_n, key, ylabel, title, filename, log=False):
    sizes = sorted(by_n)
    ramp = plt.get_cmap("Blues")
    colors = [ramp(0.35 + 0.65 * i / max(1, len(sizes) - 1)) for i in range(len(sizes))]
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for i, n in enumerate(sizes):
        ps = sorted(by_n[n])
        vals = [np.array([r[key] for r in by_n[n][p]]) for p in ps]
        mean = np.array([v.mean() for v in vals])
        se = np.array([v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0 for v in vals])
        ax.fill_between(ps, mean - se, mean + se, color=colors[i], alpha=0.18, linewidth=0)
        ax.plot(ps, mean, color=colors[i], linewidth=2, marker=markers[i % len(markers)],
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=0.8,
                solid_capstyle="round", solid_joinstyle="round", label=f"n={n}")
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(0, max(max(by_n[n]) for n in sizes) + 1))
    ax.set_xticklabels(["0\n(random)"] + [str(p) for p in ax.get_xticks()[1:]])
    ax.set_xlabel("QAOA depth p (shaded: ±1 standard error over instances)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK, fontsize=11)
    _style_axes(ax)
    _legend(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_qaoa_vs_baselines(by_method, by_n):
    """The headline RQ2 chart: QAOA at its deepest p against the four classical
    baselines, on exactly the E1 instances (n = 8..20 even, seeds 0-9).

    A baseline returns one answer, and so does QAOA in practice: the best of its
    10k shots. That is the like-for-like line; the mean <C> line shows how much
    of each shot's probability already sits near the optimum.
    """
    p_best = max(p for n in by_n for p in by_n[n] if p > 0)
    sizes = sorted(n for n in by_n if p_best in by_n[n])

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for method in METHOD_ORDER:
        x, y = _mean_by_n(by_method[method], "approx_ratio", sizes=set(sizes))
        ax.plot(x, y, color=METHOD_COLOR[method], linewidth=2, marker="o", markersize=5,
                 solid_capstyle="round", solid_joinstyle="round", label=METHOD_LABEL[method])
    for key, color, label in (("best_ratio", SLOTS["magenta"], f"QAOA p={p_best}, best of 10k shots"),
                              ("approx_ratio", SLOTS["green"], f"QAOA p={p_best}, mean <C>")):
        y = [np.mean([r[key] for r in by_n[n][p_best]]) for n in sizes]
        ax.plot(sizes, y, color=color, linewidth=2.5, marker="D", markersize=7,
                 markeredgecolor=SURFACE, markeredgewidth=0.8,
                 solid_capstyle="round", solid_joinstyle="round", label=label)

    ax.annotate("simulated annealing, CP-SAT and QAOA's best shot\nall reach 1.00 on every instance",
                (sizes[-1], 1.0), xytext=(0, -10), textcoords="offset points", ha="right", va="top",
                fontsize=8, color=INK_MUTED)
    ax.set_ylim(0.75, 1.03)
    ax.set_xticks(sizes)
    ax.set_xlabel("number of jobs (n)  (mean over the same 10 instances per n)")
    ax.set_ylabel("approx. ratio (1.0 = optimal)")
    ax.set_title("QAOA vs classical baselines on identical instances", color=INK, fontsize=11)
    _style_axes(ax)
    _legend(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "qaoa_vs_baselines.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


E2_ORDER = ["interp", "transfer", "transfer_noopt", "random5"]
E2_LABEL = {"interp": "INTERP", "transfer": "transfer + optimise",
            "transfer_noopt": "transfer, no optimisation", "random5": "best of 5 random"}
E2_COLOR = dict(zip(E2_ORDER, [SLOTS["blue"], SLOTS["orange"], SLOTS["aqua"], SLOTS["yellow"]]))
# INTERP and transfer often land on the same optimum, so shapes keep both visible.
E2_MARKER = dict(zip(E2_ORDER, ["o", "D", "^", "s"]))


def fig_e2(path):
    """Initialisation strategies at the deepest p: quality (left) and optimiser cost (right)."""
    rows = list(csv.DictReader(open(path)))
    p = max(int(r["p"]) for r in rows)
    # INTERP reaches depth p only by optimising every depth 1..p in turn, so its
    # cost is the whole chain's evaluations, not just the final stage's.
    chain = defaultdict(int)
    for r in rows:
        if r["init"] == "interp" and int(r["p"]) <= p:
            chain[r["instance_id"]] += int(r["n_evals"])
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if int(r["p"]) == p:
            if r["init"] == "interp":
                r = {**r, "n_evals": chain[r["instance_id"]]}
            by[r["init"]][int(r["n"])].append(r)

    fig, (ax_r, ax_e) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    all_sizes = sorted({n for init in by for n in by[init]})
    for init in E2_ORDER:
        sizes = sorted(by[init])
        if not sizes:
            continue
        style = dict(color=E2_COLOR[init], linewidth=2, marker=E2_MARKER[init], markersize=7,
                     markeredgecolor=SURFACE, markeredgewidth=0.8, label=E2_LABEL[init])
        ax_r.plot(sizes, [np.mean([float(r["approx_ratio"]) for r in by[init][n]]) for n in sizes], **style)
        if init != "transfer_noopt":
            ax_e.plot(sizes, [np.mean([float(r["n_evals"]) for r in by[init][n]]) for n in sizes], **style)
    ax_r.set_ylabel("approx. ratio of <C> (1.0 = optimal)")
    ax_r.set_title(f"Quality at p = {p}", color=INK, fontsize=11)
    ax_e.set_yscale("log")
    ax_e.set_ylabel("circuit evaluations used by COBYLA")
    ax_e.set_title(f"Optimiser cost at p = {p}", color=INK, fontsize=11)
    for ax in (ax_r, ax_e):
        ax.set_xticks(all_sizes)
        ax.set_xlabel("number of jobs (n)")
        _style_axes(ax)
    # The cost panel has no line for transfer_noopt (it makes 0 evaluations),
    # so the shared legend comes from the quality panel.
    handles, labels = ax_r.get_legend_handles_labels()
    ax_e.legend(handles, labels, frameon=False, labelcolor=INK_SECONDARY, fontsize=9,
                loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(FIGURES / "E2_init.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_e6(path):
    """Noise sweep: approx. ratio vs depth, one line per error rate, one panel per n."""
    rows = list(csv.DictReader(open(path)))
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        by[int(r["n"])][float(r["noise"])][int(r["p"])].append(float(r["approx_ratio"]))
    sizes = sorted(by)
    levels = sorted({lvl for n in sizes for lvl in by[n]})
    ramp = plt.get_cmap("Oranges")
    colors = {lvl: ramp(0.35 + 0.65 * i / max(1, len(levels) - 1)) for i, lvl in enumerate(levels)}
    markers = dict(zip(levels, ["o", "s", "^", "D", "v"]))

    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 4.2), dpi=150,
                             sharey=True, squeeze=False)
    for ax, n in zip(axes[0], sizes):
        for lvl in levels:
            ps = sorted(by[n][lvl])
            vals = [np.array(by[n][lvl][p]) for p in ps]
            mean = np.array([v.mean() for v in vals])
            se = np.array([v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0 for v in vals])
            ax.fill_between(ps, mean - se, mean + se, color=colors[lvl], alpha=0.18, linewidth=0)
            ax.plot(ps, mean, color=colors[lvl], linewidth=2, marker=markers[lvl], markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=0.8,
                    label="noiseless" if lvl == 0 else f"2-qubit error {lvl:g}")
        ax.set_xticks(ps)
        ax.set_xlabel("QAOA depth p")
        ax.set_title(f"n = {n}", color=INK, fontsize=11)
        _style_axes(ax)
    axes[0][0].set_ylabel("approx. ratio of <C> (1.0 = optimal)")
    _legend(axes[0][-1])
    fig.suptitle("Depolarising noise: deeper circuits stop paying off", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "E6_noise.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


E4_ORDER = ["qpp-cpu", "nvidia", "nvidia-fp64"]
E4_LABEL = {"qpp-cpu": "CPU (qpp, fp64)", "nvidia": "A40 GPU, fp32", "nvidia-fp64": "A40 GPU, fp64"}
E4_COLOR = dict(zip(E4_ORDER, [SLOTS["blue"], SLOTS["orange"], SLOTS["aqua"]]))
E4_BYTES = {"nvidia": 8, "nvidia-fp64": 16}  # one complex amplitude


def fig_e4(paths):
    """Backend scaling at fixed p = 3: time per circuit evaluation (left) and GPU
    memory (right). Runs used different numbers of COBYLA evaluations, so raw
    wall time is not comparable; time per evaluation is. CPU memory is the whole
    process (dominated by the brute-force reference), so only GPU memory is drawn."""
    rows = [r for path in paths for r in csv.DictReader(open(path))]
    per_eval = defaultdict(lambda: defaultdict(list))
    mem = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_eval[r["target"]][int(r["n"])].append(float(r["wall_time_s"]) / int(r["n_evals"]))
        if r["target"] in E4_BYTES and r["peak_mem_mb"]:
            mem[r["target"]][int(r["n"])].append(float(r["peak_mem_mb"]))

    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    for target in E4_ORDER:
        sizes = sorted(per_eval[target])
        ax_t.plot(sizes, [np.median(per_eval[target][n]) for n in sizes], color=E4_COLOR[target],
                  linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
                  markeredgewidth=0.8, label=E4_LABEL[target])
    ax_t.set_yscale("log")
    ax_t.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}s"))
    ax_t.set_ylabel("wall time per circuit evaluation (log)")
    ax_t.set_title("Simulation cost per evaluation, p = 3", color=INK, fontsize=11)

    for target, nbytes in E4_BYTES.items():
        idle = min(min(v) for v in mem[target].values())  # CUDA context before any state exists
        sizes = [n for n in sorted(mem[target]) if n >= 20]
        ax_m.plot(sizes, [(max(mem[target][n]) - idle) / 1024 for n in sizes], color=E4_COLOR[target],
                  linewidth=0, marker="o", markersize=7, markeredgecolor=SURFACE,
                  markeredgewidth=0.8, label=f"{E4_LABEL[target]}, measured")
        ax_m.plot(sizes, [2 ** n * nbytes / 2 ** 30 for n in sizes], color=E4_COLOR[target],
                  linewidth=1.5, linestyle="--", label=f"2^n x {nbytes} bytes")
    ax_m.set_yscale("log")
    ax_m.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g} GB"))
    ax_m.set_ylabel("GPU memory above idle (log)")
    ax_m.set_title("State-vector memory matches 2^n amplitudes", color=INK, fontsize=11)
    for ax in (ax_t, ax_m):
        ax.set_xlabel("number of jobs = qubits (n)")
        _style_axes(ax)
        _legend(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "E4_timing.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_e5(path, ref_n=24):
    """Transferred angles (no optimisation) at n = 20..28, one line per depth."""
    rows = list(csv.DictReader(open(path)))
    by_p = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_p[int(r["p"])][int(r["n"])].append(r)
    ps = sorted(by_p)
    ramp = plt.get_cmap("Blues")
    colors = [ramp(0.35 + 0.65 * i / max(1, len(ps) - 1)) for i in range(len(ps))]

    fig, (ax_r, ax_o) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    for i, p in enumerate(ps):
        sizes = sorted(by_p[p])
        for ax, key in ((ax_r, "approx_ratio"), (ax_o, "p_opt")):
            vals = [np.array([float(r[key]) for r in by_p[p][n]]) for n in sizes]
            mean = np.array([v.mean() for v in vals])
            se = np.array([v.std(ddof=1) / np.sqrt(len(v)) for v in vals])
            if ax is ax_r:
                ax.fill_between(sizes, mean - se, mean + se, color=colors[i], alpha=0.18, linewidth=0)
            ax.plot(sizes, mean, color=colors[i], linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, label=f"p={p}")
        all_sizes = sizes
    ax_r.set_ylabel("approx. ratio of <C> (1.0 = optimal)")
    ax_r.set_title("Quality with n = 20 angles, no re-optimisation", color=INK, fontsize=11)
    ax_o.set_yscale("log")
    ax_o.set_ylabel("P(optimal schedule) per shot (log)")
    ax_o.set_title("Probability of sampling an optimum", color=INK, fontsize=11)
    ax_o.axvspan(ref_n + 0.5, max(all_sizes) + 0.5, color=GRID, alpha=0.5, linewidth=0)
    ax_o.text(ref_n + 0.7, 0.02, "lower bound: CP-SAT\nfinds one optimum",
              transform=ax_o.get_xaxis_transform(), va="bottom", ha="left", fontsize=8, color=INK_MUTED)
    for ax in (ax_r, ax_o):
        ax.set_xticks(all_sizes)
        ax.set_xlabel("number of jobs (n)  (shaded band: ±1 s.e. over 10 instances)" if ax is ax_r
                      else "number of jobs (n)  (10k shots x 10 instances: floor 1e-5 = one hit)")
        _style_axes(ax)
    _legend(ax_o)
    fig.tight_layout()
    fig.savefig(FIGURES / "E5_largen.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _qaoa_probabilities(inst, params):
    """Exact probability of every bitstring for the circuit in src/qaoa.py (index
    order of problem.index_to_bitstring). Plain numpy, so it runs without cudaq;
    it reproduces the noiseless E6 rows for n = 6 to six decimals."""
    sys.path.insert(0, str(ROOT / "src"))
    import problem as P

    n, p = inst["n"], len(params) // 2
    z = 1 - 2 * ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1)
    diag = sum(J * z[:, i] * z[:, j] for i, j, J in P.build_hamiltonian(inst)["terms"])
    psi = np.full(1 << n, 2 ** (-n / 2), dtype=complex)
    for layer in range(p):
        gamma, beta = params[layer], params[p + layer]
        psi = (psi * np.exp(-1j * gamma * diag)).reshape((2,) * n)
        rx = np.array([[np.cos(beta), -1j * np.sin(beta)], [-1j * np.sin(beta), np.cos(beta)]])
        for q in range(n):
            psi = np.moveaxis(np.tensordot(rx, psi, axes=([1], [n - 1 - q])), 0, n - 1 - q)
        psi = psi.reshape(-1)
    return np.abs(psi) ** 2


def _qaoa_probabilities(inst, params):
    """Exact probability of every bitstring for the circuit in src/qaoa.py (index
    order of problem.index_to_bitstring). Plain numpy, so it runs without cudaq;
    it reproduces the noiseless E6 rows for n = 6 to six decimals."""
    sys.path.insert(0, str(ROOT / "src"))
    import problem as P

    n, p = inst["n"], len(params) // 2
    z = 1 - 2 * ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1)
    diag = sum(J * z[:, i] * z[:, j] for i, j, J in P.build_hamiltonian(inst)["terms"])
    psi = np.full(1 << n, 2 ** (-n / 2), dtype=complex)
    for layer in range(p):
        gamma, beta = params[layer], params[p + layer]
        psi = (psi * np.exp(-1j * gamma * diag)).reshape((2,) * n)
        rx = np.array([[np.cos(beta), -1j * np.sin(beta)], [-1j * np.sin(beta), np.cos(beta)]])
        for q in range(n):
            psi = np.moveaxis(np.tensordot(rx, psi, axes=([1], [n - 1 - q])), 0, n - 1 - q)
        psi = psi.reshape(-1)
    return np.abs(psi) ** 2


def _qaoa_probabilities(inst, params):
    """Exact probability of every bitstring for the circuit in src/qaoa.py (index
    order of problem.index_to_bitstring). Plain numpy, so it runs without cudaq;
    it reproduces the noiseless E6 rows for n = 6 to six decimals."""
    sys.path.insert(0, str(ROOT / "src"))
    import problem as P

    n, p = inst["n"], len(params) // 2
    z = 1 - 2 * ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1)
    diag = sum(J * z[:, i] * z[:, j] for i, j, J in P.build_hamiltonian(inst)["terms"])
    psi = np.full(1 << n, 2 ** (-n / 2), dtype=complex)
    for layer in range(p):
        gamma, beta = params[layer], params[p + layer]
        psi = (psi * np.exp(-1j * gamma * diag)).reshape((2,) * n)
        rx = np.array([[np.cos(beta), -1j * np.sin(beta)], [-1j * np.sin(beta), np.cos(beta)]])
        for q in range(n):
            psi = np.moveaxis(np.tensordot(rx, psi, axes=([1], [n - 1 - q])), 0, n - 1 - q)
        psi = psi.reshape(-1)
    return np.abs(psi) ** 2


def fig_pipeline(params_path, n=6, seed=0, p=4):
    """Schematic of the whole method on one real instance (n6_s0): the jobs and
    their couplings, the circuit with one ZZ gate per coupling, the exact outcome
    probabilities for the angles E6 optimised, and the schedule the best shot gives.
    Nothing in it is invented: every number comes from src/problem.py or E6."""
    import json
    from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

    sys.path.insert(0, str(ROOT / "src"))
    import problem as P

    inst = P.generate_instance(n, seed)
    raw = inst["weights_raw"]
    w = P.normalized_weights(inst)
    conflicts = {tuple(e) for e in inst["edges"]}
    terms = P.build_hamiltonian(inst)["terms"]
    ref = P.brute_force(inst, return_costs=True)
    best = ref["opt_states"][0]
    angles = next(r["params"] for r in map(json.loads, open(params_path))
                  if r["instance_id"] == inst["id"] and r["p"] == p)
    prob = _qaoa_probabilities(inst, angles)
    costs = np.asarray(ref["costs"])
    order = np.lexsort((np.arange(len(costs)), np.round(costs, 9)))   # lowest cost first
    optimal = {P.bitstring_to_index(b) for b in ref["opt_states"]}
    p_opt = sum(prob[k] for k in optimal)

    blue, aqua, orange = SLOTS["blue"], SLOTS["aqua"], SLOTS["orange"]
    node_colour = {"0": blue, "1": aqua}
    width_u, height_u = 27.0, 9.45                      # drawing units: 3 per inch
    fig = plt.figure(figsize=(9, 3.15), dpi=200)
    fig.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_u)
    ax.set_ylim(0, height_u)
    ax.axis("off")

    def card(x, y, wd, ht, number, title):
        ax.add_patch(FancyBboxPatch((x, y), wd, ht, boxstyle="round,pad=0,rounding_size=0.22",
                                    facecolor="white", edgecolor=GRID, linewidth=1, zorder=0))
        ax.add_patch(Circle((x + 0.52, y + ht - 0.5), 0.25, facecolor=INK, edgecolor="none", zorder=2))
        ax.text(x + 0.52, y + ht - 0.51, str(number), fontsize=7.5, color="white", fontweight="bold",
                ha="center", va="center", zorder=3)
        ax.text(x + 0.95, y + ht - 0.51, title, fontsize=9, color=INK, fontweight="bold",
                ha="left", va="center", zorder=3)

    def small(x, y, text, ha="center", colour=INK_SECONDARY, size=7.5, **kw):
        ax.text(x, y, text, fontsize=size, color=colour, ha=ha, va="center", **kw)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, color=INK_MUTED,
                                     linewidth=1.3, shrinkA=0, shrinkB=0, zorder=2))

    def coupling_style(i, j, J):
        """Same encoding in the graph and in the circuit: width = strength, orange = conflict."""
        return dict(color=orange if (i, j) in conflicts else INK_MUTED, linewidth=0.5 + 1.9 * J,
                    solid_capstyle="round")

    # 1 -- the instance as its Ising graph.
    card(0.15, 0.15, 5.95, 9.15, 1, "Jobs and couplings")
    cx, cy, radius = 3.12, 5.05, 2.1
    theta = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    pos = [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in theta]
    for i, j, J in sorted(terms, key=lambda t: (t[0], t[1]) in conflicts):
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], zorder=1, **coupling_style(i, j, J))
    for i, (x, y) in enumerate(pos):
        ax.add_patch(Circle((x, y), 0.36 + 0.40 * np.sqrt(w[i]), facecolor="white",
                            edgecolor=INK_SECONDARY, linewidth=1.1, zorder=3))
        ax.text(x, y - 0.02, str(raw[i]), fontsize=9, color=INK, ha="center", va="center", zorder=4)
    ax.plot([0.55, 1.25], [1.5, 1.5], color=INK_MUTED, linewidth=1.8, solid_capstyle="round")
    small(1.45, 1.5, r"balance term, $2w_iw_j$", ha="left")
    ax.plot([0.55, 1.25], [0.8, 0.8], color=orange, linewidth=2.2, solid_capstyle="round")
    small(1.45, 0.8, r"conflict, adds $\lambda$", ha="left")

    arrow((6.25, 4.7), (6.85, 4.7))

    # 2 -- the circuit: one wire per job, one ZZ gate per coupling.
    card(7.0, 0.15, 10.3, 9.15, 2, "QAOA circuit, one qubit per job")
    wires = [7.25 - 0.8 * q for q in range(n)]
    gate_x = [9.55 + 0.29 * k for k in range(len(terms))]
    box = dict(facecolor="white", edgecolor=INK_SECONDARY, linewidth=1, zorder=3)
    for q, y in enumerate(wires):
        ax.plot([7.95, 16.25], [y, y], color=AXIS, linewidth=1, zorder=1)
        ax.add_patch(Circle((7.62, y), 0.33, zorder=3, facecolor="white", edgecolor=INK_SECONDARY, linewidth=1))
        ax.text(7.62, y - 0.02, str(raw[q]), fontsize=7.5, color=INK, ha="center", va="center", zorder=4)
        ax.add_patch(Rectangle((8.22, y - 0.28), 0.56, 0.56, **box))
        ax.text(8.5, y - 0.02, "H", fontsize=7.5, color=INK, ha="center", va="center", zorder=4)
        ax.add_patch(Rectangle((14.1, y - 0.28), 0.9, 0.56, **box))
        ax.text(14.55, y - 0.02, r"$R_X$", fontsize=7.5, color=INK, ha="center", va="center", zorder=4)
        ax.add_patch(Rectangle((15.67, y - 0.28), 0.56, 0.56, **box))
        ax.add_patch(Arc((15.95, y - 0.13), 0.34, 0.34, theta1=0, theta2=180, color=INK_SECONDARY,
                         linewidth=0.9, zorder=4))
        ax.plot([15.95, 16.08], [y - 0.13, y + 0.15], color=INK_SECONDARY, linewidth=0.9, zorder=4)
        ax.text(16.78, y - 0.02, best[q], fontsize=9.5, color=node_colour[best[q]], fontweight="bold",
                ha="center", va="center")
    for x, (i, j, J) in zip(gate_x, terms):
        style = coupling_style(i, j, J)
        ax.plot([x, x], [wires[i], wires[j]], zorder=2, **style)
        ax.plot([x, x], [wires[i], wires[j]], linestyle="none", marker="o", markersize=3.4,
                color=style["color"], zorder=3)
    ax.add_patch(FancyBboxPatch((9.2, 2.85), 6.05, 4.8, boxstyle="round,pad=0,rounding_size=0.18",
                                facecolor="none", edgecolor=INK_MUTED, linewidth=0.9,
                                linestyle=(0, (3, 2)), zorder=2))
    small(12.22, 8.0, r"one layer, repeated $p$ times")
    small(16.78, 8.05, "best\nshot", size=7, linespacing=1.0)
    small(11.58, 2.38, r"$ZZ(2\gamma J_{ij})$ on every pair")
    small(14.55, 2.38, r"$R_X(2\beta)$")
    # Feedback loop: measurements -> classical optimiser -> angles of every layer.
    ax.add_patch(FancyBboxPatch((9.8, 0.72), 5.6, 0.78, boxstyle="round,pad=0,rounding_size=0.18",
                                facecolor=SURFACE, edgecolor=AXIS, linewidth=0.9, zorder=2))
    small(12.6, 1.1, r"optimiser tunes $\gamma, \beta$ to lower $\langle H_C \rangle$")
    ax.plot([15.95, 15.95], [2.95, 1.11], color=INK_MUTED, linewidth=1.3, solid_capstyle="round")
    arrow((15.95, 1.11), (15.45, 1.11))
    ax.plot([9.8, 9.45], [1.11, 1.11], color=INK_MUTED, linewidth=1.3, solid_capstyle="round")
    arrow((9.45, 1.11), (9.45, 2.8))

    arrow((17.45, 7.3), (18.05, 7.3))

    # 3 -- exact outcome probabilities for the angles optimised in E6.
    card(18.2, 5.25, 8.65, 4.05, 3, f"Outcome probabilities at $p = {p}$")
    x0, base, span, top = 18.7, 5.95, 7.7, 1.75
    step = span / len(order)
    for rank, k in enumerate(order):
        ax.add_patch(Rectangle((x0 + rank * step + 0.012, base), step - 0.024, top * prob[k] / prob.max(),
                               facecolor=INK if k in optimal else AXIS, edgecolor="none", zorder=2))
    uniform_y = base + top / (len(order) * prob.max())
    ax.plot([x0, x0 + span], [uniform_y, uniform_y], color=INK_MUTED, linewidth=0.9,
            linestyle=(0, (3, 2)), zorder=3)
    ax.plot([x0, x0 + span], [base, base], color=AXIS, linewidth=1, zorder=3)
    ax.add_patch(Rectangle((21.45, 7.73), 0.26, 0.26, facecolor=INK, edgecolor="none"))
    small(21.9, 7.86, f"2 optimal schedules: {100 * p_opt:.0f} %", ha="left")
    ax.plot([21.4, 21.76], [7.26, 7.26], color=INK_MUTED, linewidth=0.9, linestyle=(0, (3, 2)))
    small(21.9, 7.26, f"random guessing: {100 * len(optimal) / len(order):.0f} %", ha="left")
    small(x0, 5.6, f"all {len(order)} bitstrings, lowest cost first", ha="left", size=7)

    arrow((22.5, 5.15), (22.5, 4.55))

    # 4 -- what the best shot means: bit 0 = node A, bit 1 = node B.
    card(18.2, 0.15, 8.65, 4.3, 4, "Best shot as a schedule")
    unit, bar_x, bar_h = 0.175, 20.35, 0.8
    centre = {}
    for bit, y, name in (("0", 2.35, "Node A"), ("1", 0.75, "Node B")):
        colour = node_colour[bit]
        jobs = [i for i in range(n) if best[i] == bit]
        x = bar_x
        for i in jobs:
            ax.add_patch(Rectangle((x, y), raw[i] * unit, bar_h, facecolor=colour, edgecolor="white",
                                   linewidth=1.5, zorder=3))
            ax.text(x + raw[i] * unit / 2, y + bar_h / 2 - 0.02, str(raw[i]), fontsize=8, color="white",
                    fontweight="bold", ha="center", va="center", zorder=4)
            centre[i] = (x + raw[i] * unit / 2, y)
            x += raw[i] * unit
        small(18.6, y + bar_h / 2 + 0.17, name, ha="left", colour=colour, size=8, fontweight="bold")
        small(18.6, y + bar_h / 2 - 0.2, f"bit {bit}", ha="left", size=7)
        small(x + 0.25, y + bar_h / 2, f"load {sum(raw[i] for i in jobs)}", ha="left", colour=INK, size=8)
    for i, j in conflicts:
        (xa, ya), (xb, yb) = sorted((centre[i], centre[j]), key=lambda c: -c[1])
        ax.plot([xa, xb], [ya, yb + bar_h], color=orange, linewidth=2, solid_capstyle="round", zorder=2)

    fig.savefig(FIGURES / "pipeline.png", facecolor=SURFACE)
    plt.close(fig)
    return p_opt


def main():
    FIGURES.mkdir(exist_ok=True)
    by_method = load_baselines()
    fig_approx_ratio(by_method)
    fig_wall_time(by_method)
    fig_violations(by_method)
    fig_lambda_sweep(load_lambda_sweep())
    count = 4
    if (ROOT / "results" / "E1_depth.csv").exists():
        by_n = load_e1()
        _fig_e1(by_n, "approx_ratio", "approx. ratio of <C> (1.0 = optimal)",
                "QAOA gets closer to the optimum with every layer", "E1_approx_ratio.png")
        _fig_e1(by_n, "p_opt", "P(optimal schedule) per shot",
                "Probability of sampling an optimal schedule", "E1_p_opt.png", log=True)
        fig_qaoa_vs_baselines(by_method, by_n)
        count += 3
    e4 = sorted(glob.glob(str(ROOT / "results" / "E4_cluster" / "*" / "E4_timing.csv")))
    if e4:
        fig_e4(e4)
        count += 1
    for name, fig in (("E2/E2_init.csv", fig_e2), ("E5_cluster/E5_largen.csv", fig_e5),
                      ("E6/E6_noise.csv", fig_e6)):
        if (ROOT / "results" / name).exists():
            fig(ROOT / "results" / name)
            count += 1
    e6_params = ROOT / "results" / "E6" / "E6_params.jsonl"
    if e6_params.exists():
        fig_pipeline(e6_params)
        count += 1
    print(f"wrote {count} figures to {FIGURES}")


if __name__ == "__main__":
    main()
