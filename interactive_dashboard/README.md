# Quantum Kernel & Hardware Co-Design Explorer (Interactive Dashboard)

An interactive, zero-dependency browser application designed to visualize and demonstrate the end-to-end Quantum Machine Learning methodology developed for **QFF 2026 Phase 2 (PS-Q2)**.

---

## 🚀 How to Launch

### Option 1: Direct Browser Launch (Zero-Setup)
Simply double-click `index.html` in your file explorer, or open it in any modern web browser (Chrome, Edge, Firefox, Safari).

### Option 2: Python Local Server
```bash
python interactive_dashboard/run_dashboard.py
```
This automatically starts a local server on port 8000 and opens the dashboard in your default browser.

---

## 🌟 Interactive Features

1. **📊 Phase Scaling & Kernel Concentration Explorer:**
   - Real-time slider for the phase rotation parameter $\gamma \in [0.05, 1.00]$.
   - Dynamically renders the $30 \times 30$ Gram matrix heatmap.
   - Demonstrates why standard $[-\pi, \pi]$ scaling causes barren plateau kernel concentration ($K \to I$), and why our selected $\gamma = 0.10$ achieves peak Balanced Accuracy (**71.75%**).

2. **📉 Hardware Noise & Contraction Dynamics:**
   - Live toggle between **Ideal Simulation** and the official **QFF2026-v2** benchmark noise model.
   - Demonstrates the empirical Affine Contraction Law ($K_{\text{noisy}} \approx 0.709 \cdot K_{\text{ideal}} + 0.016$) and compares it to the theoretical gate error decay rate ($a_{\text{theory}} \approx 0.714$).
   - Displays readout bit-flip attenuation: $(1 - 0.02)^6 \approx 0.8858$.

3. **🛰️ Processor D Hardware Architecture Graph:**
   - Interactive 2D graph of our custom 6-qubit processor (`qml_q2_hexaring_6q_v1`).
   - Visually explains why the central triangular core provides all-to-all connectivity with **zero SWAP overhead** for 3-qubit QFT, reducing transpiled depth to 18 and boosting noisy fidelity to **0.6684**.

4. **🎯 Held-Out Test Sample Inspector:**
   - Dropdown selector across test samples ($s_0 \dots s_{199}$).
   - Displays the normalized 6D latent feature coordinates.
   - Computes live quantum classification decisions (`Class A` vs `Class B`) matching `predictions_q2.csv`.
