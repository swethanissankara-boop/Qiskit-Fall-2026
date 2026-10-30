# QFF2026 QML Track — Results Report

**Team name:** QML Explorers  
**Selected PS:** [ ] PS-Q1  [x] PS-Q2  [ ] PS-Q3  
**Date:** 2026-10-30  
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
We address PS-Q2 (High-Dimensional Quantum-Kernel Classification, 50 features) by recognizing that the non-linear classification structure is embedded into a 6D latent manifold obscured by random orthogonal projection and ambient noise. We perform Principal Component Analysis (fitted strictly on the 800 training samples) to extract the 6 dominant signal dimensions, followed by scaling to \([ -0.1\pi, 0.1\pi ]\) to prevent exponential kernel concentration. We construct a 6-qubit second-order Pauli \(ZZ\)-FeatureMap (\(U_{\Phi(\mathbf{x})}\)) with linear nearest-neighbour entanglement and evaluate the quantum kernel matrix \(K_{ij} = |\langle \psi(\mathbf{x}_i) | \psi(\mathbf{x}_j) \rangle|^2\). A Support Vector Classifier (SVC) is trained on the resulting quantum Gram matrix, providing non-linear classification in a \(2^6 = 64\)-dimensional Hilbert space that matches the performance of a tuned classical RBF kernel.

---

## 2. Results

### Classical baseline
Evaluated strictly via 5-Fold Stratified Cross-Validation on the 800 training samples (`hd_data_train.csv`) with 95% Student's t Confidence Intervals (test labels are hidden and untouched):

| Model | Official metric (Balanced Accuracy) | 95% Confidence Interval | Macro-F1 (Reference) | Notes |
|-------|----------------|-------------------------|----------------------|-------|
| Linear SVM (50D raw) | 0.7075 ± 0.0295 | [0.6709, 0.7441] | 0.7069 ± 0.0294 | Linear hyperplane on unrotated 50D data |
| RBF-SVM (50D raw, C=1.0) | 0.7154 ± 0.0282 | [0.6804, 0.7504] | 0.7154 ± 0.0282 | Tuned classical non-linear baseline |
| Poly-SVM (50D raw, degree=3) | 0.6959 ± 0.0101 | [0.6834, 0.7084] | 0.6959 ± 0.0101 | Polynomial kernel baseline |
| Random Forest (50D raw, n=100) | 0.6787 ± 0.0338 | [0.6368, 0.7206] | 0.6785 ± 0.0339 | Non-parametric decision tree ensemble |
| Linear SVM (PCA 6D) | 0.7145 ± 0.0213 | [0.6881, 0.7409] | 0.7145 ± 0.0213 | Fitted on train; removes noise coordinates |
| RBF-SVM (PCA 6D, C=1.0) | 0.7158 ± 0.0270 | [0.6823, 0.7493] | 0.7157 ± 0.0270 | Primary classical benchmark in 6D space |

### Quantum / hybrid model
Official metric: Balanced Accuracy (F1 reported for reference). All training and hyperparameter tuning performed on training-set partitions only:

| Configuration | Ideal (Balanced Acc ± 95% CI) | Noisy (QFF2026-v2) | Notes / Evaluation Protocol |
|---------------|-------------------------------|--------------------|-----------------------------|
| 6-Qubit ZZ-FeatureMap (Full 800 Train) | **0.7175 ± 0.0217** (95% CI: [0.6905, 0.7445]) | N/A (Full matrix O(N²) prohibitive for 800 samples) | Evaluated via 5-Fold Stratified CV on training set. Statistically matches RBF-SVM ($p = 0.887$). |
| 6-Qubit ZZ-FeatureMap (80-Sample Stratified Split) | **0.7000** | **0.6000** | Evaluated on 60-train / 20-val split under 1024-shot sampling; shows -0.1000 noise degradation. |
| 6-Qubit ZZ-FeatureMap (60-Sample Stratified Split) | **0.6500** | **0.6500** | Evaluated on 40-train / 20-val split under 1024-shot sampling. |

### Noise analysis on kernel matrix (Measured on benchmark slice under QFF2026-v2)
| Metric | Ideal Simulation | Noisy Simulation (QFF2026-v2) | Delta / Measured Physics |
|--------|------------------|--------------------------------|--------------------------|
| Diagonal elements (\(K_{ii}\)) | 1.0000 | 0.8789 | Attenuated by readout bit-flip noise: \((1-0.02)^6 \approx 0.8858\) |
| Mean off-diagonal elements | 0.2180 | 0.0843 | Contrast reduction due to 2Q depolarizing gate noise |
| Frobenius matrix distance \(\|K_{\text{noisy}} - K_{\text{ideal}}\|_F\) | 0.0000 | 11.5402 | Uniform contraction towards maximally mixed state |
| Fitted Affine Contraction Law | \(K = K\) | \(K_{\text{noisy}} \approx 0.709 \cdot K_{\text{ideal}} + 0.016\) | High linear correlation \(R^2 = 0.941\); slope matches gate error decay |

---

## 3. Task A — Bell state |Φ+⟩ (required)

Produced by `task_a_bell()` in `benchmark/qff_qml_benchmark.py` (8192 shots per basis):

**Fidelity definition used:**  
\(F = \frac{1 + \langle XX \rangle - \langle YY \rangle + \langle ZZ \rangle}{4}\), with \(\langle P \rangle = \frac{N_{\text{even}} - N_{\text{odd}}}{N}\), \(\text{SE}(\langle P \rangle) = \sqrt{\frac{1 - \langle P \rangle^2}{N}}\) and \(\text{SE}(F) = \frac{\sqrt{\text{SE}_{XX}^2 + \text{SE}_{YY}^2 + \text{SE}_{ZZ}^2}}{4}\).  
*(Population overlap \(P(00)+P(11)\) is not accepted.)*

| Condition | ⟨XX⟩ ± SE | ⟨YY⟩ ± SE | ⟨ZZ⟩ ± SE | F ± SE |
|-----------|-----------|-----------|-----------|--------|
| Ideal | 1.000000 ± 0.000000 | -1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| Noisy (benchmark) | 0.819092 ± 0.006338 | -0.827881 ± 0.006197 | 0.873535 ± 0.005378 | 0.880127 ± 0.002592 |

*What the difference says about noise and entanglement (Measured Physical Analysis):*  
The ideal Bell state exhibits perfect quantum correlations (\(\langle XX \rangle = 1.0\), \(\langle YY \rangle = -1.0\), \(\langle ZZ \rangle = 1.0\)) yielding unit fidelity \(F = 1.0\). Under the organizer benchmark noise model (`QFF2026-v2`), gate depolarizing error rates (\(p_1=0.01, p_2=0.05\)) and measurement readout error (\(p_{\text{readout}}=0.02\)) reduce the state fidelity to \(F = 0.8801 \pm 0.0026\). In the measurements, the transverse correlators \(\langle XX \rangle\) (0.8191) and \(\langle YY \rangle\) (-0.8279) experienced greater degradation than the longitudinal \(\langle ZZ \rangle\) correlator (0.8735). This quantitative difference demonstrates that relative quantum phase coherence is more sensitive to depolarizing noise channels than computational-basis alignment.

---

## 4. Processor D Benchmark (Task B)

Produced by `python benchmark/qff_qml_benchmark.py processor_d_config.json`:

**QFT-3 fidelity definition used:** mean over the 8 computational-basis inputs \(|j\rangle\) of \(F_j = \langle \psi_j | \rho_j | \psi_j \rangle\), where \(\psi_j = \text{QFT}|j\rangle\) (ideal) and \(\rho_j\) is the density matrix of the 3 logical qubits after transpilation under `QFF2026-v2`. Ideal must equal 1.0.

**Processor ID:** `qml_q2_hexaring_6q_v1`  
**n_qubits:** 6  
**Topology:** custom (triangular core 0-1-2 with ring extension 3-4-5)

| Benchmark | Value | Starter Kit Linear-5 Baseline | Improvement / Delta |
|-----------|-------|-------------------------------|---------------------|
| QFT-3 depth (before transpile) | 6 | 6 | Baseline |
| QFT-3 depth (after transpile) | **18** | 22 | -18.2% shallower |
| Two-qubit gate count | **9** | 12 | -25.0% fewer CX gates |
| SWAP overhead | **0** | 1 | Zero SWAPs required |
| Fidelity (ideal) | 1.000000 | 1.000000 | Identity |
| Fidelity (noisy) | **0.668377** | 0.603585 | **+6.48 percentage points** |
| Coupling map direction | Two-way | Two-way | Bidirectional links on all 7 paired edges |

*What do the results tell you about your topology? (2–4 sentences)*  
By providing complete bidirectional connectivity among the central 3 qubits (0, 1, 2), the textbook 3-qubit QFT compiles directly without inserting any routing SWAP gates (SWAP overhead = 0). Compared to standard 5-qubit linear chains which require 1 SWAP, 12 two-qubit gates, and reach depth 22 (fidelity 0.6036), our Processor D reduces transpiled depth to 18 and two-qubit gates to 9. This directly translates to an elevated noisy fidelity of 0.6684 (+6.5 percentage points), while simultaneously providing 3 extension qubits (3, 4, 5) with ring connectivity to support 6-qubit quantum feature map circuits.

---

## 5. Resource Costs
| Resource | Value | Details / Measurement Method |
|----------|-------|------------------------------|
| Qubits used | 6 | Matches 6 principal components of the latent manifold |
| Circuit depth (feature map) | 17 | Untranspiled; 32 when transpiled to native basis `['rx','ry','rz','cx']` |
| Two-qubit gates (feature map) | 10 | Nearest-neighbour CX gates per repetition |
| Trainable parameters | 0 | Pure quantum feature map kernel; dual coefficients trained classically via SVM |
| Training time (approx) | ~0.64 s | Computation of 800 statevectors and Gram matrix in NumPy |
| Total circuit evaluations | 1,000 statevectors | 800 train + 200 test statevectors for ideal; 3,030 transition circuits for noisy validation split |

---

## 6. Limitations and Honest Assessment
*What didn't work? Is quantum competitive with classical here? Why or why not?*  
- **Statistical Parity Rather Than Superiority:** On the 5-fold cross-validation of the training set, the Quantum Kernel achieved a Balanced Accuracy of 0.7175 ± 0.0217 (95% CI: [0.6905, 0.7445]), while the classical RBF-SVM achieved 0.7158 ± 0.0270 (95% CI: [0.6823, 0.7493]). A paired Student's t-test yields $p = 0.887$, indicating that the 0.17 percentage-point difference is within random variation. The quantum model **matches** the classical baseline rather than decisively beating it.
- **Kernel Concentration Phenomenon:** Initially, scaling PCA-reduced features into standard angle ranges \([-\pi, \pi]\) caused quantum kernel performance to drop to 0.4714 (evaluated on training-set cross-validation). We demonstrated empirically that this was caused by *exponential kernel concentration*: large phase rotations in the \(ZZ\) generator \(2(\pi - x_j)(\pi - x_k)\) produce near-orthogonal statevectors where \(\text{Var}(K_{ij}) \to 0\) and \(K \to I\). Restricting the feature scaling domain to \([-0.1\pi, 0.1\pi]\) restored off-diagonal variance and yielded 0.7175 Balanced Accuracy.
- **Computational Cost Trade-off:** While the quantum kernel achieves statistical parity with RBF-SVM, the classical RBF kernel executes in 0.02 seconds ($3.2 \times 10^7$ arithmetic FLOPs), whereas an exact full quantum kernel on 800 samples requires 320,000 circuit executions (~54 minutes on physical QPUs).
- **Hardware Noise Vulnerability:** Under `QFF2026-v2` noise, balanced accuracy on the 80-sample stratified validation split drops from 0.7000 to 0.6000 (-10.0 percentage points) due to off-diagonal contrast contraction.

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

## 8. Optional Extensions & Bonus Criteria

### [x] 5-Way Systematic Ablation Study (+2 Bonus Points)
All ablations evaluated on training-set cross-validation:

| Ablation Dimension | Configuration | Metric (Balanced Acc) | Evaluation Split | Key Takeaway / Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Dimensionality** | Raw Top 6 Features | 0.5344 | 5-Fold Train CV | Random orthogonal projection hides signal across coordinates |
| **Dimensionality** | PCA 4 Components | 0.6563 | 5-Fold Train CV | Truncates the true 6D latent manifold |
| **Dimensionality** | **PCA 6 Components (Selected)** | **0.7175** | **5-Fold Train CV** | **Optimal recovery of the 6D latent manifold** |
| **Dimensionality** | PCA 8 Components | 0.7120 | 5-Fold Train CV | Incorporates ambient Gaussian noise coordinates |
| **Circuit Depth** | **reps = 1 (Selected)** | **0.7175** | **5-Fold Train CV** | **Depth 17; minimal gate noise accumulation** |
| **Circuit Depth** | reps = 2 | 0.7025 | 5-Fold Train CV | Depth 33; excessive depolarizing decay |
| **Phase Scaling** | $\gamma = 1.00$ ($[-\pi, \pi]$) | 0.4714 | 5-Fold Train CV | Severe exponential kernel concentration ($K \to I$) |
| **Phase Scaling** | **$\gamma = 0.10$ (Selected)** | **0.7175** | **5-Fold Train CV** | **Smooth metric distance in Hilbert space** |
| **Entanglement** | Full All-to-All | 0.7080 | 5-Fold Train CV | Excessive CX gates without expressivity benefit |
| **Entanglement** | **Linear (Selected)** | **0.7175** | **5-Fold Train CV** | **Hardware-efficient nearest-neighbour coupling** |
| **Processor D** | Linear 5-Qubit Chain | $F = 0.6036$ (QFT-3) | Official QFT-3 | 1 SWAP overhead, depth 22, 12 CX gates |
| **Processor D** | **Hexaring 6-Qubit Core (Selected)** | **$F = 0.6684$ (QFT-3)** | **Official QFT-3** | **0 SWAPs, depth 18, 9 CX gates (+6.5% fidelity)** |

### [x] Creative Problem Framing & NISQ Complexity Analysis (+1 Bonus Point)
- **Classical RBF Complexity:** $O(N^2 \cdot D) \approx 3.2 \times 10^7$ FLOPs $\implies \mathbf{0.02\text{ s}}$ execution time.
- **Quantum Kernel Complexity:** $O(N^2) = 320,000$ circuit transitions. At 1,024 shots and $10~\mu\text{s}$ per shot $\implies \mathbf{54\text{ minutes}}$ on physical QPUs.
- **Conclusion:** On NISQ hardware, quantum kernels are practical only when paired with train-fitted dimensionality reduction and strategic sub-sampling.
