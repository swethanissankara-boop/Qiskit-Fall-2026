# QFF2026 QML Track — Results Report

**Team name:** QML Explorers
**Selected PS:** [ ] PS-Q1  [x] PS-Q2  [ ] PS-Q3
**Date:** 2026-10-29
**Dependency versions:**
```text
pennylane==0.44.1
PennyLane-qiskit==0.44.1
qiskit==2.3.0
qiskit-aer==0.17.2
scikit-learn==1.9.1
numpy==2.4.6
pandas==3.1.0
scipy==1.17.1
matplotlib==3.11.2
```
**Noise model version:** QFF2026-v2

---

## 1. Approach Summary
We tackle PS-Q2 (High-Dimensional Quantum-Kernel Classification, 50 features) by recognizing that the non-linear classification structure is embedded into a lower-dimensional latent subspace obscured by orthogonal projection and Gaussian noise. We perform Principal Component Analysis (fitted strictly on the 800 training samples) to extract the 6 dominant signal dimensions, followed by scaling to \([ -0.1\pi, 0.1\pi ]\) to avoid phase scrambling and barren-plateau kernel concentration. We construct a 6-qubit second-order Pauli \(ZZ\)-FeatureMap (\(U_{\Phi(\mathbf{x})}\)) with linear nearest-neighbour entanglement and evaluate the quantum kernel matrix \(K_{ij} = |\langle \psi(\mathbf{x}_i) | \psi(\mathbf{x}_j) \rangle|^2\). A Support Vector Classifier (SVC) is trained on the resulting quantum Gram matrix, providing non-linear classification in a \(2^6 = 64\)-dimensional Hilbert space that directly competes with classical RBF kernels.

---

## 2. Results

### Classical baseline
Evaluated with 5-Fold Stratified Cross-Validation on the 800 training samples (`hd_data_train.csv`):

| Model | Official metric (Balanced Accuracy) | Macro-F1 (Reference) | Notes |
|-------|----------------|----------------------|-------|
| Linear SVM (50D raw) | 0.7075 ± 0.0295 | 0.7069 ± 0.0294 | Demonstrates non-trivial linear projection correlation |
| RBF-SVM (50D raw, C=1.0) | 0.7154 ± 0.0282 | 0.7154 ± 0.0282 | Strong non-linear classical kernel baseline |
| Poly-SVM (50D raw, degree=3) | 0.6959 ± 0.0101 | 0.6959 ± 0.0101 | Polynomial interaction baseline |
| Random Forest (50D raw, n=100) | 0.6787 ± 0.0338 | 0.6785 ± 0.0339 | Non-parametric decision tree ensemble |
| Linear SVM (PCA 6D) | 0.7145 ± 0.0213 | 0.7145 ± 0.0213 | PCA compresses signal and filters ambient noise |
| RBF-SVM (PCA 6D, C=1.0) | 0.7158 ± 0.0270 | 0.7157 ± 0.0270 | Primary classical benchmark in 6D feature space |

### Quantum / hybrid model
Official metric: Balanced Accuracy (F1 reported for reference):

| Configuration | Ideal | Noisy (benchmark) | Notes |
|---------------|-------|--------------------|-------|
| 6-Qubit ZZ-FeatureMap + Precomputed SVC (Full 800 Train, 5-Fold CV) | **0.7175 ± 0.0217** (Macro-F1: 0.7177) | N/A (Full matrix O(N²) prohibitive for 800 samples) | Achieves competitive performance beating RBF-SVM by 0.2% |
| 6-Qubit ZZ-FeatureMap + Precomputed SVC (60-Sample Subset Benchmark) | 0.6500 | 0.6500 | Evaluated on 40-train/20-val slice with 1024-shot sampling |

### Noise analysis on kernel matrix (60-sample benchmark slice)
| Metric | Ideal Simulation | Noisy Simulation (QFF2026-v2) | Delta / Impact |
|--------|------------------|--------------------------------|----------------|
| Diagonal elements (\(K_{ii}\)) | 1.0000 | 0.8789 | Attenuated by readout bit-flip noise: \((1-0.02)^6 \approx 0.8858\) |
| Mean off-diagonal elements | 0.2180 | 0.0843 | Contrast reduction and phase damping due to 2Q depolarizing noise |
| Frobenius matrix distance \(\|K_{\text{noisy}} - K_{\text{ideal}}\|_F\) | 0.0000 | 11.5402 | Uniform contraction towards maximally mixed state |

---

## 3. Task A — Bell state |Φ+⟩ (required)

Produced by `task_a_bell()` in `benchmark/qff_qml_benchmark.py` (8192 shots per basis):

**Fidelity definition used:**  
\(F = \frac{1 + \langle XX \rangle - \langle YY \rangle + \langle ZZ \rangle}{4}\), with \(\langle P \rangle = \frac{N_{\text{even}} - N_{\text{odd}}}{N}\), \(\text{SE}(\langle P \rangle) = \sqrt{\frac{1 - \langle P \rangle^2}{N}}\), and \(\text{SE}(F) = \frac{\sqrt{\text{SE}_{XX}^2 + \text{SE}_{YY}^2 + \text{SE}_{ZZ}^2}}{4}\).  
*(Population overlap \(P(00)+P(11)\) is not accepted.)*

| Condition | ⟨XX⟩ ± SE | ⟨YY⟩ ± SE | ⟨ZZ⟩ ± SE | F ± SE |
|-----------|-----------|-----------|-----------|--------|
| Ideal | 1.000000 ± 0.000000 | -1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| Noisy (benchmark) | 0.819092 ± 0.006338 | -0.827881 ± 0.006197 | 0.873535 ± 0.005378 | 0.880127 ± 0.002592 |

*What the difference says about noise and entanglement (3–5 sentences):*  
The ideal Bell state exhibits perfect quantum correlations (\(\langle XX \rangle = 1\), \(\langle YY \rangle = -1\), \(\langle ZZ \rangle = 1\)) yielding unit fidelity \(F = 1.0\). Under the organizer benchmark noise model (gate depolarizing error rates \(p_1=0.01, p_2=0.05\) and measurement error \(p_{\text{readout}}=0.02\)), the fidelity degrades to \(F = 0.8801 \pm 0.0026\). The two-qubit entangling CX gate contributes the largest single-step entropy increase due to depolarizing noise, while measurement readout bit-flips symmetrically erode the observable parities across all bases. Crucially, the transverse correlators \(\langle XX \rangle\) and \(\langle YY \rangle\) decay more significantly than the longitudinal \(\langle ZZ \rangle\) correlator, illustrating that relative quantum phase coherence is substantially more vulnerable to depolarizing channel mixing than computational basis populations.

---

## 4. Processor D Benchmark (Task B)

Produced by `python benchmark/qff_qml_benchmark.py processor_d_config.json`:

**QFT-3 fidelity definition used:** mean over the 8 computational-basis inputs \(|j\rangle\) of \(F_j = \langle \psi_j | \rho_j | \psi_j \rangle\), where \(\psi_j = \text{QFT}|j\rangle\) (ideal) and \(\rho_j\) is the density matrix of the 3 logical qubits after transpilation under `QFF2026-v2`. Ideal must equal 1.0.

**Processor ID:** `qml_q2_hexaring_6q_v1`  
**n_qubits:** 6  
**Topology:** custom (triangular core 0-1-2 with ring extension 3-4-5)

| Benchmark | Value |
|-----------|-------|
| QFT-3 depth (before transpile) | 6 |
| QFT-3 depth (after transpile) | 18 |
| Two-qubit gate count | 9 |
| SWAP overhead | 0 |
| Fidelity (ideal) | 1.000000 |
| Fidelity (noisy) | 0.668377 |
| Coupling map direction (one-way / two-way links) | Two-way bidirectional links on all 7 paired edges |

*What do the results tell you about your topology? (2–4 sentences)*  
By providing complete bidirectional connectivity among the central 3 qubits (0, 1, 2), the textbook 3-qubit QFT compiles directly without inserting any routing SWAP gates (SWAP overhead = 0). Compared to standard 5-qubit linear chains which require 1 SWAP, 12 two-qubit gates, and reach depth 22 (fidelity 0.6036), our Processor D reduces transpiled depth to 18 and two-qubit gates to 9. This directly translates to an elevated noisy fidelity of 0.6684 (+6.5 percentage points), while simultaneously providing 3 extension qubits (3, 4, 5) with ring connectivity to support 6-qubit quantum feature map circuits.

---

## 5. Resource Costs
| Resource | Value |
|----------|-------|
| Qubits used | 6 |
| Circuit depth (feature map) | 17 (untranspiled), 32 (transition circuit transpiled to native basis) |
| Trainable parameters | 0 (pure quantum feature map kernel; dual coefficients trained classically via SVM) |
| Training time (approx) | ~0.64 s (800 statevectors + Gram matrix computation) |
| Total circuit evaluations | 1,000 statevectors (800 train + 200 test) for ideal; 1,830 transition circuits for noisy subset |

---

## 6. Limitations and Honest Assessment
*What didn't work? Is quantum competitive with classical here? Why or why not?*  
- **Feature Scaling Sensitivity:** Initially, scaling PCA-reduced features into standard angle ranges \([-\pi, \pi]\) caused quantum kernel performance to collapse near random guessing (~0.48 Balanced Accuracy). This was caused by the high non-linear phase rotations in the \(ZZ\) generator \(2(\pi - x_j)(\pi - x_k)\) producing near-orthogonal statevectors and barren-plateau kernel concentration. Restricting the feature scaling domain to \([-0.1\pi, 0.1\pi]\) restored smooth inner-product geometry and elevated Balanced Accuracy to 0.7175.
- **Competitiveness vs. Classical:** The quantum kernel matches and marginally exceeds classical RBF-SVM (0.7175 vs 0.7158). However, classical RBF achieves this with negligible latency and zero vulnerability to physical hardware gate and readout errors.
- **Hardware Noise Vulnerability:** Under `QFF2026-v2` noise, global kernel contrast drops by ~61%, with diagonal elements decaying from 1.0 to 0.88 and off-diagonals dropping from 0.22 to 0.08. While rank order of similarities is partially preserved on small subsets, scaling to larger datasets under unmitigated noise degrades decision margin separation.

---

## 7. Reproducibility
```python
SEED = 42
# Set random seeds across all libraries
import numpy as np
import random
from qiskit import transpile
from qiskit_aer import AerSimulator

np.random.seed(42)
random.seed(42)
```

Commands to run and verify:
```bash
python benchmark/qff_qml_benchmark.py processor_d_config.json
python tools/validate_predictions.py q2 predictions_q2.csv
```

---

## 8. Optional Extensions
- [x] **Ablation Studies:**
  1. Dimensionality comparison: PCA 4 components (0.6563) vs. PCA 6 components (0.7175) vs. raw top 6 features (0.5344). Confirms orthogonal latent space projection.
  2. Phase scaling ablation: scale factors 0.05, 0.10, 0.20, 0.50, 1.00 demonstrating kernel concentration phenomenon.
  3. Processor D topology comparison: linear-5 chain (0.6036 fidelity, 1 SWAP) vs. hexaring-6q core (0.6684 fidelity, 0 SWAPs).

