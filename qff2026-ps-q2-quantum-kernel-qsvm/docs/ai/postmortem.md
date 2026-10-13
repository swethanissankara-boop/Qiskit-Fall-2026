# Post-mortem

**Outcome.** All Phase 1 checklist items met for PS-Q2: exact reference, A/B coupling maps designed and justified, depth/CX/SWAP/accuracy comparison, required repo structure, runnable notebook with pinned requirements.

**What went well**
* Searching the topology space exposed a real constraint violation in the reference graph (degree-5 hub) that a hand design would have copied.
* The route-once parametrised circuit made 8 full Gram matrices + sweeps feasible on one CPU in ≈10 min.
* The affine-contraction law matched measurements (slopes within ~0.03), giving a quantitative explanation of *why* topology matters.

**What went wrong / surprises**
* The cost-aware model selection made the headline workload topology-independent; we needed a deliberate stress workload.
* Accuracy saturated at 100 % for every backend; the n=20 test split cannot rank processors.
* B could not beat the reference heavy-hex on CX (plateau) — only on realism (Δ=3).
* Tooling slips: shell brace expansion, background job lifetime (I-01, I-05); an inaccurate hand-written rationale (I-06).

**Lessons**
1. Verify constraints on *baselines* too before using them as search seeds.
2. Generate prose claims from computed values.
3. Plan for a metric that still discriminates when accuracy saturates.

**If we had more time**: harder dataset / larger test set, error mitigation (PS-Q3 overlap), kernel-diagonal measurement variant, multiple router seeds to get variance on SWAP counts, true hardware noise from a fake backend.
