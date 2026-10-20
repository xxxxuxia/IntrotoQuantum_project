# Quantum Load Balancing for HPC Clusters: QAOA Simulated at Scale with GPUs

**SC4055 Group Project** · Implemented in NVIDIA CUDA-Q
**Team:** P1 `<name>` · P2 `<name>` · P3 `<name>`

> Project-wide plan: goals, shared conventions, experiments, and weekly milestones.
> Detailed problem spec: [`docs/P2_problem_formulation.md`](P2_problem_formulation.md).

**Key dates**

| Milestone | Date |
|---|---|
| W9 checkpoint | Wed 14 Oct |
| 🔒 Experiment freeze | **Sun 25 Oct** |
| Report + code final | **Sun 1 Nov** |
| Video final | **Sun 8 Nov** |
| Submission | Mon/Tue 9–10 Nov (W13) |

---

## 1. Project overview

### 1.1 Background
HPC clusters run many jobs at once, and a scheduler (e.g. Slurm) decides which node runs each job. Poor placement overloads one node while another idles, so everyone waits longer. Some jobs also should not share a node (e.g. two memory-heavy jobs) — *anti-affinity*. Finding the best placement is an optimisation problem whose search space grows as `2^n`.

### 1.2 Problem
Split `n` jobs across **2 nodes** to minimise makespan, keeping conflicting jobs on different nodes. Without conflicts this is Number Partitioning (NP-hard), though strong classical heuristics exist.

### 1.3 Quantum approach
We encode the problem as an Ising Hamiltonian (one qubit per job) and solve it with the **Quantum Approximate Optimization Algorithm (QAOA)**: alternating cost and mixer layers whose angles `(γ, β)` are tuned by a classical optimiser so that measurement favours good schedules.

### 1.4 HPC component
Simulating `n` qubits requires `2^n` amplitudes (≈ 8 GB at 30 qubits in fp32). We use GPU statevector simulation (CUDA-Q `nvidia` target) to reach problem sizes a CPU cannot, and benchmark the speedup.

---

## 2. Objectives

**Main goal:** determine how well QAOA solves a realistic scheduling problem, and what it takes to simulate it at scale.

| # | Focus | Question |
|---|---|---|
| RQ1 | Solution quality | How close does QAOA get to the optimal schedule, and does it improve with circuit depth? |
| RQ2 | Classical comparison | How does QAOA compare with simulated annealing and standard partitioning heuristics? |
| RQ3 | Noise robustness | How much do realistic gate errors degrade QAOA? |
| RQ4 | HPC scalability | How much faster are GPUs than CPUs for these simulations, and how large can we go? |

**Scope.** We do not aim to show quantum advantage — at simulable sizes, classical methods win, and GPU acceleration speeds up the *simulation*, not the algorithm. We aim for a correct formulation, a reproducible implementation, and an honest analysis of where QAOA works, where it struggles, and why.

---

## 3. Formulation (summary)

Full details, derivations and the worked example: [`P2_problem_formulation.md`](P2_problem_formulation.md).

```
C(z) = (Σ_i w_i z_i)^2 + λ Σ_{(i,j)∈E} z_i z_j ,    z_i = +1 (node A) / −1 (node B)
     = Σ_i w_i^2 + Σ_{i<j} J_ij z_i z_j ,            J_ij = 2 w_i w_j + λ·1[(i,j)∈E]
```

- `makespan = (W + |D|)/2` with `D = Σ w_i z_i` → minimising `D²` = minimising makespan.
- **Z₂ symmetry:** `C(z) = C(−z)` → always ≥ 2 optimal bitstrings.
- **Weights:** lognormal runtimes, normalised (`w / max w`), Hamiltonian scaled so `max|J_ij| = 1`.
- **Conflicts:** planted bipartition (conflicts only across a random split) so a zero-violation schedule always exists.
- **λ:** smallest value from `{0.1, 0.5, 1, 2, 5}` giving 0 violations at the optimum on ≥ 90 % of instances.

---

## 4. Roles

| Role | Owns | Report sections | Video |
|---|---|---|---|
| **P1 — Quantum Core** | QAOA kernel in CUDA-Q, optimisation loop, E1, angle export for E5 | Method (QAOA), E1 results | 4–9 min |
| **P2 — Problem, Noise & Analysis** | `problem.py`, E0 (with P1), E2, E6, optional E7, report integration | Intro, Background, Formulation, E2/E6/E7 results, Discussion, Conclusion | 0–4, 12–15 min |
| **P3 — HPC, Baselines & Infra** | experiment runner, baselines, E3, E4, E5, figures pipeline, repo, demo | Implementation, Setup, E3/E4/E5 results, Reproducibility | 9–12 min |

Expected load: ~10–12 h/person/week; ~12–15 h in W9–W10.

---

## 5. Shared conventions

### 5.1 Repository layout
```
sc4055-qaoa/
├── docs/
│   ├── PROJECT_PLAN.md
│   └── P2_problem_formulation.md
├── src/
│   ├── problem.py      # instances, Hamiltonian, cost, brute force, metrics   (P2)
│   ├── qaoa.py         # kernel, observe, optimise, sample                    (P1)
│   ├── baselines.py    # Karmarkar–Karp, greedy LPT, SA, CP-SAT                (P3)
│   └── bench.py        # timing harness, target switching                     (P3)
├── experiments/        # E0_validate.py … E7_xy_mixer.py — one script per experiment
├── instances/          # generated JSON instances (deterministic from seed)
├── results/            # raw CSVs — never hand-edited
├── figures/            # generated only by scripts
├── tests/              # unit tests (E0)
├── notebooks/demo.ipynb
├── requirements.txt
└── README.md
```

### 5.2 Instance format (JSON)
`{id, n, seed, weights_raw[], weights[], edges[[i,j]], planted_sides[], lambda, scale, params}`

### 5.3 Results schema (one CSV row per run)
`exp, instance_id, n, p, init, optimizer, target, precision, noise, energy, approx_ratio, p_opt, best_cost, violations, makespan, n_evals, wall_time_s, seed`

### 5.4 Bit / qubit convention
- bit `0` ↔ `z = +1` ↔ node A; bit `1` ↔ `z = −1` ↔ node B; qubit `i` = job `i`.
- All CUDA-Q bitstring parsing goes through `problem.bitstring_to_index()` (ordering verified in E0).

### 5.5 Metrics (defined once in `problem.py`)
| Metric | Definition |
|---|---|
| Approximation ratio | `r = (C_max − ⟨C⟩)/(C_max − C_min)`; also for best sampled bitstring |
| `P(opt)` | total probability of all optimal bitstrings (≥ 2 by symmetry) |
| Violations | conflict edges with both jobs on the same node (best sample) |
| Makespan | `(W_raw + |D_raw|)/2` of best sample, real units |
| TTS₉₉ | `t_run · ln(0.01)/ln(1 − P(opt))` |

Reference optimum: brute force for `n ≤ 24`; CP-SAT best-known for `n > 24`.

### 5.6 Working rules
- Every CSV row stores its seed; every run checkpoints to CSV immediately.
- Figures come only from scripts in `experiments/`.
- **Write as you go:** each person drafts method/setup text for their part in the week they build it.
- Sync: two 15-min check-ins per week (Mon, Thu).

### 5.7 Interfaces
| From → To | What | When |
|---|---|---|
| P2 → P1 | instance JSON, `build_hamiltonian()`, normalisation | end W8 |
| P2 → P1, P3 | metric functions, brute-force optima (n ≤ 24) | end W8 |
| P3 → P1, P2 | experiment runner + CSV schema | end W8 |
| P1 ↔ P2 | E0 session (statevector, bit ordering) | W8 Fri |
| P1 → P2 | `optimize(instance, p, init_params) → (params, energy, n_evals)` | W9 Tue |
| P3 → all | CP-SAT optima for n > 24 | W9 |
| P1 → P3 | median optimised angles per `p` | W10 Tue |

---

## 6. Experiments

| ID | Experiment | Owner | Sizes | Done by |
|---|---|---|---|---|
| E0 | Validation / unit tests | P1 + P2 | n = 4–8 | W8 |
| E1 | Depth sweep `p = 1–6`, INTERP warm-start | P1 | n = 8–20, 10 instances | W9 (error bars W10) |
| E2 | Initialisation: random multi-start vs INTERP vs transfer | P2 | n = 10–16, 10 instances | W10 |
| E3 | Classical baselines (KK, LPT, SA, CP-SAT) + TTS vs QAOA | P3 | n = 8–28 | W9 |
| E4 | CPU (`qpp-cpu`) vs GPU (`nvidia`) vs fp64 — time & memory | P3 | n = 10–30 | W9 |
| E5 | Large-n QAOA with transferred angles (no optimisation) | P3 (angles from P1) | n = 20–28 | W10 |
| E6 | Depolarising noise `p_err ∈ {0, 0.001, 0.005, 0.01}` × `p = 1–4` | P2 | n = 6, 8, 10 | W10 Wed |
| E7 | *Optional:* 3 nodes, penalty/X-mixer vs XY-mixer | P2 | 4–6 jobs (12–18 qubits) | W10, only if approved at checkpoint |

**Compute limits:** full optimisation loops `n ≤ ~20–22`; single-pass timing/sampling up to ~30 qubits; density-matrix noise ≤ ~12 qubits.

---

## 7. Weekly milestones

### Recess (28 Sep – 4 Oct) — Kickoff ✅
- All: CUDA-Q installed (Linux / WSL2 / Docker), GHZ example runs; CUDA-Q QAOA tutorial.
- P1: Farhi et al. 2014. P2: Lucas 2014 §2.1, Zhou et al. 2020, Hadfield et al. 2019. P3: GPU access (CCDS cluster or Colab/Kaggle T4).

### W8 (5–11 Oct) — Foundations
| Owner | Tasks |
|---|---|
| P1 | QAOA kernel (H → ZZ phase layers → RX mixer), `observe`, COBYLA loop at n = 6–8 |
| P2 | `problem.py` (generator, Hamiltonian, brute force, metrics); λ sweep; report §1–3 draft; Overleaf + notation table; noise-model skeleton |
| P3 | Runner (CSV logging, checkpointing, target switching); first CPU timings n = 10–24; baselines KK, LPT, SA |

**✅ Milestone:** E0 passes — `⟨C⟩ = Σw²` at zero angles; sampled costs match classical cost; bit ordering verified; p = 3 at n = 6 finds the optimum with high probability. Report §1–3 drafted.

### W9 (12–18 Oct) — Core experiments
| Owner | Tasks |
|---|---|
| P1 | E1 preliminary Mon, full by Fri; COBYLA vs parameter-shift + L-BFGS → lock optimiser; draft Method §4 |
| P2 | E2 starts Tue; noise setup validated (zero noise = statevector) |
| P3 | Full E4 incl. fp64; CP-SAT + reference optima; full E3; plotting pipeline; draft Implementation §5 |

**⚠ Checkpoint (Wed 14 Oct):** (1) QAOA clearly beats random sampling at n = 10; (2) GPU runs at n ≥ 24. If either fails, the other two swarm on it Thu–Fri (check normalisation → Hamiltonian signs → γ/β bounds). If both pass, E7 is approved.

**✅ Milestone:** E1, E3, E4 complete; §4–5 drafted.

### W10 (19–25 Oct) — Finish all experiments
| Owner | Tasks |
|---|---|
| P1 | E1 error bars; export median angles per `p` to P3 by Tue |
| P2 | E2 complete; E6 sweep by Wed; E7 Thu–Sat if approved |
| P3 | E5 at n = 20–28 on GPU (10k shots); experimental setup §6 |
| All | Final versions of all figures |

**🔒 FREEZE Sun 25 Oct** — after this, runs only to fix bugs.

### W11 (26 Oct – 1 Nov) — Report + code final
- **Mon:** team review of all figures → agree on 4–6 key findings (storyline for report and video).
- **Mon–Wed:** owners write their Results sections; P2 writes Discussion, Limitations, Conclusion.
- **Thu–Fri:** cross-review (P1 → P3, P3 → P2, P2 → P1); P2 integrates into one voice.
- **P3:** repo cleanup — README with one command per figure, `requirements.txt`, fixed seeds, `demo.ipynb` runs < 5 min.

**✅ Done Sun 1 Nov:** final report PDF + final repo.

### W12 (2–8 Nov) — Video
- Mon–Tue: slides from report figures + script (target ~13.5 min).
- Wed: full timed rehearsal. Thu–Sat: record and edit.

| Time | Presenter | Content |
|---|---|---|
| 0–4 min | P2 | Motivation + problem formulation |
| 4–9 min | P1 | QAOA method + E1 results |
| 9–12 min | P3 | HPC (E4/E5) + baselines (E3) + short demo |
| 12–15 min | P2 | E2/E6 (+E7) + limitations + conclusion |

**✅ Done Sun 8 Nov.**

### W13 (9–13 Nov) — Buffer + submit
- Re-record weak segments if needed; final read-through.
- **Submit Mon/Tue:** video link, source-code link and report PDF to the course instructors.

---

## 8. Report outline (10–12 pages)

| # | Section | Owner | Drafted |
|---|---|---|---|
| 1 | Introduction & Motivation | P2 | W8 |
| 2 | Background: Ising/QUBO, QAOA, GPU statevector simulation | P2 | W8 |
| 3 | Problem Formulation | P2 | W8 |
| 4 | Method: QAOA circuit, optimiser, initialisation | P1 | W9 |
| 5 | Implementation: CUDA-Q kernel, targets, runner | P3 | W9 |
| 6 | Experimental Setup: instances, metrics, seeds, hardware | P3 | W10 |
| 7 | Results: E1–E6 (+E7) | each owner | W11 |
| 8 | Discussion & Limitations | P2 | W11 |
| 9 | Conclusion + Contribution Statement | P2 | W11 |

Discussion must state explicitly: no quantum advantage at simulable scale; GPU accelerates simulation, not the algorithm; the noise model is simplified.

---

## 9. Contingencies

| Risk | Response |
|---|---|
| QAOA fails the W9 checkpoint | Swarm Thu–Fri; if still failing, drop E2/E5/E7, keep E1 + E3 + E4 + E6 |
| No GPU access | Colab/Kaggle T4 — E4/E5 still reach ~30 qubits |
| Behind in W10 | Cut E7 → shrink E6 to one `n`. Minimum viable set: E1, E3, E4 |
| Writing slips | §1–3 not drafted by end W8 = early warning; fix immediately |
| Notebook/session dies | Checkpoint every run to CSV; never rely on in-memory state |

---

## References

- E. Farhi, J. Goldstone, S. Gutmann, *A Quantum Approximate Optimization Algorithm*, arXiv:1411.4028 (2014).
- A. Lucas, *Ising formulations of many NP problems*, Front. Phys. 2:5 (2014).
- L. Zhou et al., *Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices*, PRX 10, 021067 (2020).
- S. Hadfield et al., *From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz*, Algorithms 12(2):34 (2019).
- N. Karmarkar, R. M. Karp, *The Differencing Method of Set Partitioning*, UC Berkeley Tech. Report (1982).
- NVIDIA CUDA-Q documentation and QAOA tutorial.
