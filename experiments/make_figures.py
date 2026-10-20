"""Turn results/*.csv into report-ready figures. Figures come only from this
script (per the team's own rule) -- never hand-edited.

    python experiments/make_figures.py        # writes figures/*.png
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

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


def fig_approx_ratio(by_method):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for method in METHOD_ORDER:
        rs = by_method[method]
        x = [int(r["n"]) for r in rs]
        y = [float(r["approx_ratio"]) for r in rs]
        ax.plot(x, y, color=METHOD_COLOR[method], linewidth=2, marker="o", markersize=5,
                 solid_capstyle="round", solid_joinstyle="round", label=METHOD_LABEL[method])
    ax.set_ylim(0.75, 1.03)
    ax.set_xlabel("number of jobs (n)")
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
        rs = by_method[method]
        x = [int(r["n"]) for r in rs]
        y = [float(r["wall_time_s"]) for r in rs]
        ax.plot(x, y, color=METHOD_COLOR[method], linewidth=2, marker="o", markersize=5,
                 solid_capstyle="round", solid_joinstyle="round", label=METHOD_LABEL[method])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}s"))
    ax.set_xlabel("number of jobs (n)")
    ax.set_ylabel("wall time (log scale)")
    ax.set_title("How long each baseline takes to run", color=INK, fontsize=11)
    _style_axes(ax)
    _legend(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "baseline_wall_time.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_violations(by_method):
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    totals = [sum(int(r["violations"]) for r in by_method[m]) for m in METHOD_ORDER]
    colors = [METHOD_COLOR[m] for m in METHOD_ORDER]
    bars = ax.bar(METHOD_ORDER, totals, color=colors, width=0.55)
    for rect, val in zip(bars, totals):
        ax.annotate(str(val), (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("total conflict violations (summed across n=8..28)")
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
    size_colors = list(SLOTS.values())
    # Several sizes land on identical fractions at every lambda (e.g. n=8 and
    # n=10 here) and would otherwise sit perfectly hidden under each other --
    # distinct marker shapes keep every series visible even when fully overlapped.
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for i, n in enumerate(sizes):
        y = [sum(by_lam_n[(lam, n)]) / len(by_lam_n[(lam, n)]) for lam in lambdas]
        ax.plot(lambdas, y, color=size_colors[i % len(size_colors)], linewidth=2,
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


def main():
    FIGURES.mkdir(exist_ok=True)
    by_method = load_baselines()
    fig_approx_ratio(by_method)
    fig_wall_time(by_method)
    fig_violations(by_method)
    fig_lambda_sweep(load_lambda_sweep())
    print(f"wrote 4 figures to {FIGURES}")


if __name__ == "__main__":
    main()
