# Issues log

| ID | Found | Issue | Impact | Resolution | Status |
|---|---|---|---|---|---|
| I-01 | Build | Shell did not support `{a,b}` brace expansion; folder tree and first source files were not created | Lost one tool call | Created directories explicitly, rewrote files | Closed |
| I-02 | Search | Reference heavy-hex graph has a **degree-5 hub** (qubit 6) and the first hill-climb *started from it*, so the "max degree 3" B design actually had degree 5 | Would have silently violated the realism constraint | Baselines are scored but excluded from search seeds; winner filtered on Δ≤3 | Closed |
| I-03 | Search | B search plateaus: many graphs tie at CX = 36.0 (incl. reference) | Plain CX objective cannot rank them | Added depth → average-path tie-breakers; reported honestly that B equals the reference in CX | Closed |
| I-04 | Modelling | Cost-aware CV picked a linear 1-rep map; A and B give *identical* results | No A-vs-B signal on the headline workload | Added STRESS workload (ADR-004) | Closed |
| I-05 | Execution | Background notebook execution was killed when the tool call returned ("Parent appears to have exited") | Wasted ~5 min | Relaunched with `setsid nohup … < /dev/null` | Closed |
| I-06 | Docs | First draft of the B rationale claimed a *lower* average path and "diamond cells" — neither is true (equal to reference; 2 triangles, 1 four-cycle) | Inaccurate claim in `processor_B.json` | Rationale is now generated from computed statistics; notebook re-executed | Closed |
| I-07 | Spec | Guide's tree shows `src/_init _.py` (typesetting artefact) | Possible mismatch with checklist | Used standard `__init__.py`, noted in README | Closed |
| I-08 | Statistics | n_test = 20 → accuracy CI is [0.84, 1.00] even at 100 % | Accuracy cannot discriminate processors | Report Wilson CIs, kernel-level metrics and say so explicitly | Open (inherent to PS-Q2 split) |
| I-09 | Fidelity | Level-1 sabre routing is heuristic; SWAP counts could move with Qiskit version | Reproducibility across versions | Versions pinned in `requirements.txt`; seed 42 | Open (documented) |
