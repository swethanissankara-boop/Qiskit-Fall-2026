# 📊 Presentation Visualizations Guide: Qiskit Fall Fest 2026 Phase 2 (PS-Q2)

This directory contains 7 high-resolution, publication-grade figures (300 DPI) designed for presentations, slide decks, and final evaluation viva defense.

---

## 🖼️ Index of Visualizations

| File | Title | Primary Role / Concept Explained |
| :--- | :--- | :--- |
| **`01_pipeline_architecture.png`** | End-to-End Pipeline Architecture | Complete overview of data flow: 50D data → PCA 6D → Scaling → ZZ-FeatureMap → Overlaps → Classical SVM → Test Predictions. |
| **`02_pca_latent_manifold.png`** | PCA Scree Plot & Latent Manifold | Proves why 6 dimensions were selected (elbow at PC6, ~68% variance) and visualizes the non-linear class distribution. |
| **`03_kernel_concentration_and_scaling.png`** | Exponential Concentration & Angle Scaling | Visual proof of how unscaled features cause Gram matrix collapse ($\text{Var} \approx 0.003$), and how $\gamma = 0.10$ scaling restores contrast ($\text{Var} \approx 0.054$). |
| **`04_noise_contraction_physics.png`** | Hardware Noise & Affine Contraction Law | Empirical vs theoretical noise physics: $K_{\text{noisy}} = 0.709 \cdot K_{\text{ideal}} + 0.016$ ($R^2 = 0.941$) and readout bit-flip attenuation ($K_{ii} \approx 0.88$). |
| **`05_task_a_bell_tomography.png`** | Task A: Bell State Pauli Tomography | 3-basis Pauli measurement ($\langle XX \rangle, \langle YY \rangle, \langle ZZ \rangle$) yielding $F = 0.8801 \pm 0.0026$, and why the classical $P(00)+P(11)$ shortcut is invalid. |
| **`06_task_b_processor_d_benchmark.png`** | Task B: Processor D Topology & Benchmark | Network layout of the custom 6-qubit architecture (`qml_q2_hexaring_6q_v1`) with 0 SWAPs, depth 18, and +6.48 pp fidelity over linear baseline. |
| **`07_model_benchmark_comparison.png`** | Benchmark Comparison & Statistical Parity | Bar chart comparing all 8 classical and quantum models with 95% CIs, highlighting statistical parity ($p = 0.887$) and organizer baseline ($65.40\%$). |

---

## 🗣️ Slide-by-Slide Viva Speaking Notes

### 1. `01_pipeline_architecture.png` (The Grand Overview)
* **What to say**: *"This diagram captures our complete end-to-end workflow. We take high-dimensional, noisy data, extract the true 6-dimensional latent signal using PCA, scale the angles by $\gamma = 0.10$ to prevent concentration, evaluate quantum state overlaps on a 6-qubit ZZ-FeatureMap, and train a classical SVM. This hybrid model achieves statistical parity with tuned classical non-linear models."*

### 2. `02_pca_latent_manifold.png` (Data Engineering & Anti-Leakage)
* **What to say**: *"The scree plot on the left clearly shows an elbow at component 6. Components 1 through 6 capture 68% of the variance—the true physical signal—while components 7 through 50 represent ambient Gaussian noise. PCA was fitted strictly on the 800 training samples to eliminate data leakage."*

### 3. `03_kernel_concentration_and_scaling.png` (The Core Quantum Insight)
* **What to say**: *"Without angle scaling (Panel A), large rotations cause states in 64-dimensional Hilbert space to become orthogonal, collapsing the Gram matrix into an identity matrix with variance of 0.003 where SVM fails. By scaling angles by $\gamma = 0.10$ (Panel B & C), we restore healthy variance to 0.054, spreading overlaps across $[0.10, 0.85]$ and enabling the SVM to place its decision boundary."*

### 4. `04_noise_contraction_physics.png` (Hardware Noise Modeling)
* **What to say**: *"When exposed to the realistic `QFF2026-v2` noise model, the quantum kernel undergoes an affine contraction: $K_{\text{noisy}} \approx 0.709 \cdot K_{\text{ideal}} + 0.016$. The slope 0.709 precisely matches theoretical gate survival $(1 - 0.02)^{17} \approx 0.714$, and diagonal attenuation $K_{ii} \approx 0.8789$ matches 6-qubit readout bit-flip probability $(0.98)^6$."*

### 5. `05_task_a_bell_tomography.png` (Entanglement Verification)
* **What to say**: *"In Task A, we rejected the trivial $P(00)+P(11)$ shortcut because it cannot detect quantum phase coherence. We implemented full 3-basis Pauli tomography with 8,192 shots each, measuring $\langle XX \rangle = 0.8191$, $\langle YY \rangle = -0.8279$, $\langle ZZ \rangle = 0.8735$, giving a verified state fidelity of $0.8801 \pm 0.0026$."*

### 6. `06_task_b_processor_d_benchmark.png` (Custom Hardware Design)
* **What to say**: *"Our Processor D design features an all-to-all bidirectional triangular core $\{0, 1, 2\}$ and a ring extension $\{3, 4, 5\}$. On the 3-qubit QFT benchmark, it compiles with zero routing SWAP gates, reducing depth to 18 (vs 22) and CX gates to 9 (vs 12), achieving a noisy fidelity of 0.6684 (+6.48 pp higher than the linear baseline)."*

### 7. `07_model_benchmark_comparison.png` (Final Results & Parity)
* **What to say**: *"Our ideal Quantum Kernel achieved 71.75% balanced accuracy, matching the classical RBF-SVM (71.58%) with a paired t-test $p = 0.887$, comfortably exceeding the organizer test baseline of 65.40%. Under device noise, it achieved 60.00%. We emphasize statistical parity rather than unverified supremacy."*
