# QFF 2026 · Phase 1 · PS-Q2 — Quantum Kernel Estimator & QSVM

**Track 4 — Quantum Machine Learning · Problem Statement PS-Q2** (exactly one problem statement selected)
Team: `<TEAM NAME>` · Members: `<NAMES>` · SRM University-AP · Qiskit Fall Fest 2026

> **One-line result.** A 4-qubit fidelity quantum kernel + classical SVM reaches **100 % test accuracy** (n=20) on `make_moons`;
> accuracy saturates, so the architectural story is told by *kernel fidelity*: on the densest feature map our co-designed
> **Processor B cuts routed CX by 12.5 %, depth by 20 % and SWAPs by 33 % vs. our Processor A**, and our **Processor A cuts CX by 22 % and SWAPs by 50 % vs. the reference linear chain** —
> which directly lowers kernel error (MAE 0.0484 vs 0.0533 vs 0.0633).

---
## 1. Problem statement (PS-Q2, verbatim requirements → how we meet them)

| Requirement | Where / how |
|---|---|
| Quantum kernel matrix from fidelity overlap `|⟨φ(x)\|φ(x')⟩|²` on a **4-qubit** register | `src/problem.py` (`feature_map`, `kernel_circuit`, compute–uncompute, `P(0000)`) |
| Classical SVM trained with the quantum kernel | `SVC(kernel="precomputed")`, `src/problem.py::svm_accuracy` |
| Dataset `make_moons(n_samples=100, noise=0.1, random_state=42)` | `src/problem.py::make_dataset` → `data/raw/moons_raw.csv` |
| Compare **kernel evaluation cost** on Processor A vs B | `results/tables/kernel_evaluation_cost.csv`, report §3 |
| Processor A (5q) & B (9q), basis `{rz,sx,x,cx}`, level 1, seed 42, 4096 shots | `src/routing.py`, `processors/*.json` |
| Identical synthetic noise (1q 0.1 %, CX 0.5 %, readout 1 %) on A and B | `src/processors.py::build_noise_model` |
| Coupling maps **designed by us** and justified | §3 below, `processors/processor_{A,B}.json`, `report.md` |
| Phase 1 only A & B (no C / D) | ✔ |

## 2. Repository layout (as required by the Master Guide, plus `docs/ai/`)
```
README.md  main.ipynb  requirements.txt  report.md
data/raw/moons_raw.csv      data/processed/{moons_scaled.npz, kernels_main.npz, kernels_stress.npz}
processors/processor_A.json processors/processor_B.json
src/{__init__,problem,processors,routing,metrics,visualization}.py
results/tables/*.csv|json   results/figures/*.png
assets/qff_phase1_latex_images/eq_*.png
docs/ai/{adrs,issues,patterns,plan-evolution,postmortem,prompts}.md
```
(`src/__init__.py` is the standard package file — the guide's `_init _.py` is a typesetting artefact.)

## 3. What makes this submission different

1. **Coupling maps are *searched*, not copied.** Processor A = exhaustive search over all 6 connected 5-node graphs (≤5 edges, degree ≤3).
   Processor B = 250 random graphs + hill-climbing over 12-edge, **degree ≤3** graphs (the reference heavy-hex graph has a degree-5 hub on qubit 6, which is unrealistic).
   Objective = mean routed CX over the entangler family {linear, ring, full}, then depth, then average path length. A **held-out `star` workload** checks for over-fitting.
2. **Algorithm ↔ architecture ablation.** Four entangler patterns × six topologies (our two + four baselines) → CX / depth / SWAP table (`results/tables/topology_benchmark_all_workloads.csv`).
3. **Cost-aware model selection.** Hyper-parameters chosen by *train-only* CV on the exact kernel; among near-ties the *cheapest-CX* feature map wins (→ `linear`, 1 rep, γ=1, C=1).
   Because that cheap map maps perfectly onto both processors, we also run a **stress workload** (full K4 entangler, 2 reps) where connectivity must matter.
4. **Error decomposition.** Exact kernel → shot-noise-only → noisy A → noisy B (+ reference baselines).
5. **Analytic noise model validated.** Global depolarising noise predicts `K_noisy ≈ a·K + b` with `a ≈ F_gates·(1-p_r)^4`. Measured slopes match within ≈0.02–0.03 (e.g. A: 0.709 vs 0.686).
6. **Noise-scaling sweep** (0×–4× the spec noise) and a Bell-pair noise-model sanity check (⟨ZZ⟩ = 0.955 measured vs 0.954 predicted).
7. Honest statistics: n_test = 20 so every accuracy has a Wilson 95 % interval; classical RBF-SVM baseline included.

## 4. Key results (seed 42, 4096 shots, 4950 circuits per Gram matrix)

**Designed processors**

| | Processor A | Processor B |
|---|---|---|
| Qubits / edges / max degree | 5 / 5 / 3 | 9 / 12 / 3 |
| Coupling map | `[(0,1),(1,3),(1,4),(2,3),(2,4)]` — 4-cycle 1-3-2-4 + pendant 0 | `[(0,2),(0,4),(0,6),(1,3),(1,5),(1,8),(2,7),(3,5),(3,6),(4,8),(5,6),(7,8)]` |
| Diameter / avg path | 3 / 1.60 | 3 / 1.94 |

**Routed kernel circuit (reps = 2) — CX / depth / SWAPs**

| Topology | linear | ring | full | star (held-out) |
|---|---|---|---|---|
| A reference linear | 22 / 44 / 0 | 64 / 90 / 12 | 82 / 120 / 12 | 34 / 67 / 4 |
| A reference tree | 22 / 44 / 0 | 48 / 92 / 6 | 64 / 120 / 6 | 22 / 50 / 0 |
| **A custom** | 22 / 44 / 0 | **30 / 62 / 0** | 64 / 120 / 6 | 22 / 50 / 0 |
| B reference heavy-hex (degree-5 hub) | 22 / 44 / 0 | 30 / 62 / 0 | 56 / 96 / 4 | 22 / 50 / 0 |
| B 3×3 grid | 22 / 44 / 0 | 30 / 62 / 0 | 64 / 120 / 6 | 22 / 50 / 0 |
| **B custom (degree ≤3)** | 22 / 44 / 0 | 30 / 62 / 0 | **56 / 96 / 4** | 22 / 50 / 0 |

**Classification and kernel quality**

| Workload | Backend | CX | Depth | SWAP | Kernel MAE | Rel. Frobenius err | KTA (train) | Test acc [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| MAIN `linear-r1` | Exact | – | – | – | 0 | 0 | 0.3722 | 1.00 [0.84, 1.00] |
| | Shot-only | – | – | – | 0.0025 | 0.014 | 0.3721 | 1.00 |
| | Noisy A | 10 | 24 | 0 | 0.0098 | 0.072 | 0.3688 | 1.00 |
| | Noisy B | 10 | 24 | 0 | 0.0098 | 0.072 | 0.3688 | 1.00 |
| STRESS `full-r2` | Exact | – | – | – | 0 | 0 | 0.3578 | 1.00 |
| | Shot-only | – | – | – | 0.0042 | 0.017 | 0.3579 | 1.00 |
| | Noisy A (custom) | 64 | 120 | 6 | 0.0533 | 0.244 | 0.3460 | 1.00 |
| | Noisy B (custom) | 56 | 96 | 4 | **0.0484** | **0.222** | **0.3478** | 1.00 |
| | Noisy A reference linear | 82 | 120 | 12 | 0.0633 | 0.290 | 0.3414 | 1.00 |
| | Noisy B reference heavy-hex | 56 | 96 | 4 | 0.0485 | 0.222 | 0.3477 | 1.00 |
| Classical | RBF-SVM (same features) | | | | | | | 1.00 (CV 0.992) |

**Kernel-evaluation cost** (one full 100×100 Gram): 4950 circuits × 4096 shots = 20.3 M shots (4760 circuits if only train×train + test×train is needed).
Executed CX: 203 M (main, both processors); stress: 1298 M (A), 1135 M (B), 1663 M (A reference linear).

### Honest reading of the results
* Accuracy is **saturated at 100 %** for every backend (the moons data is easy and n_test = 20 gives a 95 % CI of [0.84, 1.00]); accuracy alone cannot separate the processors. Kernel error, alignment and CX/SWAP overhead do, and they order consistently with routed CX count.
* For the cost-optimal main feature map both processors are *equivalent* (0 SWAPs, 10 CX) — connectivity only matters once the entangler graph is denser than the hardware graph. That is exactly what the ring / full / star ablation shows.
* Our B matches the reference heavy-hex CX exactly but with max degree 3 instead of 5 — a realism gain, not a CX gain.

## 5. Reproduce
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.12
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace main.ipynb   # or open main.ipynb and Run All
```
Cached kernel matrices (`data/processed/kernels_*.npz`) make a re-run take ≈3–4 min; set `RECOMPUTE = True` in the first cell to re-simulate everything (≈10 min on one CPU). Outputs are deterministic (verified by running the notebook twice).

## 6. Limitations
Simulated noise only (no real hardware); `rz` treated as virtual/noise-free; kernel diagonal fixed to 1 (ADR-006); topologies tuned on a 3-workload family (checked on held-out `star`); sabre routing at level 1 is heuristic; n_test = 20.

See `report.md` for the full analysis and `docs/ai/` for decision records, issues, patterns, plan evolution, post-mortem and the AI-assistance prompt log.
