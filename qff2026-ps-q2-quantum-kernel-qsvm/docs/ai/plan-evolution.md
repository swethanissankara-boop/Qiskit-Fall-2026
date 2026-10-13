# Plan evolution

**v0 — Read the brief (Oct 12, 2026).** Deadline Wed Oct 14, 23:59 IST. Required repo layout fixed; add `docs/ai/`. Rubric: 30 correctness / 35 architecture / 20 reproducibility / 15 docs → most effort on architecture analysis and reproducibility.

**v1 — Initial plan (considered, never built).** Ring-entangled ZZ feature map, hand-drawn A (tree) and B (heavy-hex copy), compare depth/CX/SWAP, run SVM. *Rejected*: copying reference maps ignores the "creative freedom" criterion; ring map on a 5-qubit tree is arbitrary.

**v2 — Co-design.** Make the coupling map a *search result*; add entangler-pattern ablation (linear/ring/full/star) so the algorithm side is also a variable. Added kernel-error metrics because accuracy on 100 moons points saturates.

**v3 — Realism & honesty fixes.** Discovered degree-5 hub in the reference (I-02) → Δ≤3 constraint enforced correctly; search plateau (I-03) → tie-breakers; MAIN workload showed A==B (I-04) → added STRESS workload; added analytic contraction law, noise sweep, Bell sanity check.

**v4 — Packaging.** Parametrised/route-once pipeline, cached Gram matrices, deterministic re-run verified, JSON rationales generated from stats, README/report with only measured numbers, ADR/issue/postmortem docs, zip.

**Dropped ideas** (time/benefit): ZNE/readout mitigation (belongs to PS-Q3), real-hardware run, PSD projection of noisy kernels, bigger datasets (spec fixes n=100).
