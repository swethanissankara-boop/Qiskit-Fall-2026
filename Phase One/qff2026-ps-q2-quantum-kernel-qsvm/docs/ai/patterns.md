# Reusable patterns

1. **Route once, bind many.** Parametrised kernel circuit + one transpile per processor; `assign_parameters` per data pair.
2. **Compute–uncompute fidelity.** `P(0…0)` of `U(x)U(y)†` is the fidelity kernel; no swap-test ancilla needed (saves qubits and CX).
3. **Error-source ladder.** exact → shot-only → noisy-A → noisy-B → baselines; each rung isolates one cause (finite shots, noise, routing).
4. **Topology as an optimisation variable.** Enumerate (small n) or random-search + hill-climb (larger n) under *hardware-realism constraints* (degree, edge budget), lexicographic objective, held-out workload as over-fitting check.
5. **Predictive noise law.** Fit `K_noisy = a·K + b`, compare `a` with `∏(1−p_g)`; if they agree, gate counts are a trustworthy cost metric.
6. **Cost-aware model selection.** Among CV-ties choose the cheapest circuit; then add a stress workload so the cheap choice cannot hide architectural differences.
7. **Report-the-null.** When A == B or accuracy saturates, say so and show the metric that does discriminate.
8. **Generated documentation.** Processor JSON rationales are produced from computed numbers, not typed by hand.
9. **Cache + RECOMPUTE flag** for expensive simulations.
