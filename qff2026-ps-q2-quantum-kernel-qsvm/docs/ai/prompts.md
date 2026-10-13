# AI-assistance disclosure & prompt log

An AI assistant (Anthropic Claude) generated the code, notebook, figures and documentation in this repository from the prompt below. The team is responsible for reviewing, running and understanding every artefact before submission. All numbers in README/report are copied from files in `results/tables/` produced by `main.ipynb`.

## P-01 — Task prompt (summarised faithfully)
> Provided: *QFF Phase 1 Master Guide* (PDF) and an image of a `docs/ai/` folder containing `adrs.md, issues.md, patterns.md, plan-evolution.md, postmortem.md, prompts.md`.
> Request: add `docs/ai` to the required repository structure; problem statement chosen = Track 4 Quantum Machine Learning **PS-Q2 (quantum kernel classifier)**; follow the exact PDF details (A/B only, creative coupling maps, basis gates, noise model, level 1, seed 42, 4096 shots, `make_moons(100, 0.1, 42)`, required file tree, checklist); make the execution unique with a good chance of reaching the finals; deliver everything as a zip, following every detail.

## P-02 — Internal generation steps (assistant)
1. Parse brief → derive requirement table (README §1).
2. Prototype parametrised kernel circuit; time noisy Aer (≈12 ms/circuit).
3. Implement `src/` modules; run model selection and topology searches.
4. Build and execute `main.ipynb`; inspect figures/tables; fix issues I-02, I-06.
5. Write README, report, `docs/ai/*`; package zip.

## Prompt-engineering notes
* Ask the assistant to *report nulls* (A==B, saturated accuracy) rather than hide them.
* Ask for claims in prose to be traceable to a CSV row.
