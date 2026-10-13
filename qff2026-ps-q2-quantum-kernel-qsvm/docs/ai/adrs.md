# Architecture Decision Records — PS-Q2 Quantum Kernel QSVM

Format: Context → Decision → Consequences. Status of all records: **Accepted**.

## ADR-001 — Select PS-Q2 (Quantum Kernel Estimator & QSVM)
**Context.** Phase 1 allows exactly one of 20 problem statements; judging gives 35 pts to Processor A-vs-B analysis. PS-Q2 has a closed-form exact reference (statevector kernel), a tiny dataset (cheap to simulate thousands of noisy circuits) and a natural cost metric ("kernel evaluation cost on A vs B").
**Decision.** Implement PS-Q2 only.
**Consequences.** Every circuit is 4 qubits, so *connectivity effects must be created by the entangler graph*, not by qubit count (→ ADR-003, ADR-004).

## ADR-002 — Compute–uncompute kernel circuit with a *parametrised, routed-once* transpilation
**Context.** 4950 kernel entries; transpiling each bound circuit separately is slower and, more importantly, would let the optimiser exploit per-pair angle coincidences.
**Decision.** Build `U(x)·U(y)†` with `ParameterVector`s, transpile once per processor, bind parameters per pair, estimate `P(0000)`.
**Consequences.** Metrics (depth/CX/SWAP) are properties of the *processor+circuit family*, as on real hardware. ~12 ms per noisy circuit on Aer.

## ADR-003 — Design coupling maps by search, not by hand
**Context.** The guide states the reference maps are non-binding and that creativity of the topology is judged.
**Decision.** A: exhaustive over all connected 5-node graphs with ≤5 edges, Δ≤3 (6 graphs). B: random + hill-climb over 12-edge graphs, Δ≤3. Lexicographic objective: mean routed CX over {linear, ring, full} → depth → avg path length → #edges → Δ.
**Alternatives rejected.** Hand-drawing a graph (not reproducible/justifiable); optimising for a single entangler (over-fits); allowing arbitrary degree (reference heavy-hex has a degree-5 hub — unrealistic).
**Consequences.** Search is deterministic (seed 42), 25 s. Over-fitting guarded by held-out `star` workload.

## ADR-004 — Two workloads: cost-aware MAIN and densest-entangler STRESS
**Context.** Cost-aware CV selected a `linear`, 1-rep map (12 logical CX). It embeds in both processors with 0 SWAPs, so A and B tie.
**Decision.** Keep MAIN (honest model selection) and add STRESS = `full` entangler, 2 reps, best CV among that family. Report both.
**Consequences.** Differences between processors are visible and the neutral MAIN result is not hidden.

## ADR-005 — Hyper-parameter selection on train-only CV with the exact kernel
**Decision.** Repeated stratified 5×3-fold CV on the 80 training rows only; scaler fitted on train; tie-break = fewest logical CX within 1 % of best.
**Consequences.** No test leakage; test accuracy is a clean (if low-power, n=20) estimate. "All-data CV" in tables is flagged as informational.

## ADR-006 — Kernel diagonal fixed to 1
**Context.** The diagonal k(x,x)=1 analytically. Simulating it would add 100 circuits that test only the device noise floor, not the data, and is not standard practice for fidelity kernels.
**Decision.** Do not simulate diagonal entries; set `K_ii = 1`.
**Consequences.** Noisy kernels show diagonal dominance (off-diagonals shrink, the diagonal does not). This is reported as a "concentration" diagnostic (effective rank 14 -> 32-39) rather than hidden; a measured-diagonal variant would show smaller contrast and is listed as future work.

## ADR-007 — Noise model details
**Decision.** Depolarising 0.001 on `sx`,`x`; `rz` virtual (noise-free); depolarising 0.005 on `cx`; symmetric readout 0.01 on all qubits; identical on A and B. `build_noise_model(scale, noisy_rz)` exposes sweeps and a pessimistic rz-noise variant.
**Consequences.** Analytic contraction `a ≈ F_gates (1-p_r)^4` matches measurements within ~0.03 and validates the setup (Bell ⟨ZZ⟩ check).

## ADR-008 — SWAP counting through a second, swap-aware transpilation
**Decision.** Same seed, basis `{rz,sx,x,cx,swap}` to count router-inserted SWAPs; the official `{rz,sx,x,cx}` transpile gives depth/CX and is what is simulated.
**Consequences.** SWAP count may differ marginally from the CX excess/3 because SWAPs can cancel adjacent CX; both are reported/derivable.

## ADR-009 — Cache kernel matrices in `data/processed/`
**Decision.** Gram matrices are saved as `.npz`; `RECOMPUTE=True` re-simulates.
**Consequences.** Judges can re-run the notebook in minutes; results remain bit-identical.
