# Report — PS-Q2 Quantum Kernel Estimator & QSVM on custom Processors A and B

All numbers come from `main.ipynb` (seed 42, 4096 shots, optimisation level 1, basis `{rz,sx,x,cx}`) and are stored in `results/tables/`.

## 1. Method in brief
* **Data**: `make_moons(100, noise=0.1, random_state=42)`, stratified 80/20 split, MinMax scaling fitted on the training rows only.
* **Encoding**: 2 raw features → 4 lifted features `[x0, x1, x0·x1, x0²−x1²]`, one per qubit, angle `γ·f_i`.
* **Feature map U(x)**: ZZFeatureMap-style (`H`, `P(2a_i)`, `CX–P(2(π−a_i)(π−a_j))–CX`), with a configurable entangler graph (linear / ring / full / star) and 1–2 repetitions.
* **Kernel**: `k(x,x') = P(0000)` of `U(x)·U(x')†` — a compute–uncompute estimate of `|⟨φ(x)|φ(x')⟩|²`. The parametrised circuit is routed once per processor and parameters are bound per pair.
* **Classifier**: `SVC(kernel="precomputed")`. Hyper-parameters (entangler, reps, γ, C) from train-only repeated 5-fold CV on the exact kernel with a *cheapest-CX-within-1 %* tie-break.
* **Noise**: 1q depolarising 0.001 (on `sx`,`x`), CX depolarising 0.005, symmetric readout 0.01, identical on A and B.

## 2. Architecture co-design (Challenge Step 1)
| | Processor A | Processor B |
|---|---|---|
| Search space | all 6 connected 5-node graphs, ≤5 edges, Δ≤3 | 12-edge connected 9-node graphs, Δ≤3 (250 random + hill-climb) |
| Result | 4-cycle (1-3-2-4) + pendant qubit 0 | degree-3 graph, diameter 3, 2 triangles, 1 four-cycle |
| Why | the 4-cycle hosts a ring-entangled 4-qubit map with **0 SWAPs**; pendant keeps Δ=3 | hits the CX floor of the workload family (36.0 mean CX) with Δ=3; reference needs a degree-5 hub |

**Evidence** (reps=2, routed CX / SWAPs): ring entangler — A custom 30 / 0 vs reference linear 64 / 12 and reference tree 48 / 6 (**−53 % CX** vs the chain). Full entangler — A custom 64 / 6, B custom 56 / 4, reference chain 82 / 12.
Held-out `star` workload: A custom 22 CX equals the reference tree and beats the chain (34); B custom equals the reference heavy-hex (22) — no sign of over-fitting to the tuned family.
**Negative / neutral findings we report on purpose**: the B search plateaus: many random graphs tie at the CX floor, so B's advantage over the reference is realism (Δ=3 vs 5), not fewer gates. The 3×3 grid is *worse* on the full entangler (64 CX) than the sparser custom graph.

## 3. Kernel-evaluation cost on A vs B
| Workload | Proc. | Layout (phys. qubits) | Depth | CX depth | CX | SWAP | Executed CX per Gram (M) | Est. gate fidelity | Sim time |
|---|---|---|---|---|---|---|---|---|---|
| main linear-r1 | A | [2,3,1,0] | 24 | 10 | 10 | 0 | 203 | 0.944 | 45 s |
| main linear-r1 | B | [7,2,0,4] | 24 | 10 | 10 | 0 | 203 | 0.944 | 45 s |
| stress full-r2 | A | [1,4,3,0] | 120 | 64 | 64 | 6 | 1298 | 0.714 | 62 s |
| stress full-r2 | B | [3,5,6,1] | 96 | 48 | 56 | 4 | 1135 | 0.743 | 63 s |
| stress full-r2 | A ref. linear | [1,2,0,3] | 120 | 68 | 82 | 12 | 1663 | 0.652 | 64 s |

A full Gram needs 4950 circuits × 4096 shots (20.3 M shots); only 4760 circuits are needed for train×train + test×train.
B vs A on the stress workload: −12.5 % CX, −20 % depth, −33 % SWAPs. Note that only 4 physical qubits are active in every run: B's advantage comes from the denser *local* neighbourhood the router found (physical qubits 3, 5, 6, 1), **not** from having more qubits. (Simulated wall time is a property of the simulator and is reported only as a reference.)

## 4. Quality of the estimated kernel
| Backend (stress) | Kernel MAE | Rel. Frobenius | KTA (train) | slope a (fit / model) | intercept b | eff. rank |
|---|---|---|---|---|---|---|
| Exact | 0 | 0 | 0.3578 | 1 | 0 | 14.2 |
| Shot-only | 0.0042 | 0.017 | 0.3579 | 1.000 | 0 | 15.2 |
| Noisy A | 0.0533 | 0.244 | 0.3460 | 0.709 / 0.686 | 0.0169 | 34.3 |
| Noisy B | 0.0484 | 0.222 | 0.3478 | 0.736 / 0.714 | 0.0154 | 32.2 |
| Noisy A ref. linear | 0.0633 | 0.290 | 0.3414 | 0.653 / 0.627 | 0.0203 | 38.9 |

* **Contraction law**: hardware noise acts on the kernel as `K_noisy ≈ a·K + b` with `a ≈ (1−p₁)^{#sx+#x}(1−p₂)^{#cx}(1−p_r)^4`. Fitted slopes agree with the gate-count model to within 0.02–0.03 on all runs, so CX count is a *predictive* cost metric for kernel error, not just a proxy.
* **Concentration**: off-diagonal spread falls (0.218 → 0.155 for A) and the effective rank rises (14 → 34) because the diagonal stays at 1 while off-diagonals shrink — the kernel drifts toward the identity, the classic route to poor generalisation. Kernel–target alignment drops monotonically with routed CX: 0.3578 → 0.3478 (B) → 0.3460 (A) → 0.3414 (chain).
* **Error scaling** (noise sweep, 300 pairs): kernel MAE grows roughly linearly with the noise scale (stress/A: 0.030, 0.055, 0.093, 0.139 at 0.5×, 1×, 2×, 4×; B is 5–10 % lower at every non-zero scale). Main workload: 0.0058 → 0.0348 for both processors (identical, as expected from identical routed circuits).
* **Noise-model sanity**: Bell-pair ⟨ZZ⟩ = 0.955 on every edge of both processors vs 0.954 predicted analytically → both processors really do see the same noise; only routing differs.

## 5. Classification
Test accuracy is 1.00 for exact, shot-only, A, B, the reference baselines and the classical RBF-SVM (95 % Wilson CI [0.84, 1.00] with n=20). Repeated 5-fold CV over all 100 points is also 1.00 for the stress map (informational; hyper-parameters were fixed from training data). The moons task is therefore *too easy to discriminate processors by accuracy*; a harder dataset or a larger test set would be required, and we say so rather than over-claim.

## 6. Trade-offs and recommendations
1. **Match the entangler graph to the coupling graph.** A ring feature map on a 4-cycle costs 30 CX; on a chain it costs 64. Choosing a *cheaper* entangler (linear) removed the processor dependence entirely at no accuracy cost.
2. A small, well-chosen graph (A: 5 qubits / 5 edges) is competitive with a larger one on 4-qubit workloads; extra qubits help only when they add useful *local* connectivity (triangles / diamonds), not when they add nodes.
3. For a fixed noise budget, kernel error ≈ linear in routed CX, so topology design should minimise **CX after routing**, then depth.

## 7. Limitations
Noise is simulated and global-depolarising-like; `rz` is noise-free; the kernel diagonal is set to 1 (ADR-006), which makes the contraction visible as diagonal dominance; the topology search objective is workload-specific (guarded by a held-out workload); level-1 sabre routing is heuristic and seed-dependent; n_test = 20.
