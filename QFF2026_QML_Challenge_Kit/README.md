# QFF 2026 Phase 2 — QML Track Submission (PS-Q2)

**Team Name:** QML Explorers  
**Selected Track:** Quantum Machine Learning (Phase 2)  
**Selected Problem Statement:** **PS-Q2 (High-Dimensional Quantum-Kernel Classification)**  
**Benchmark Noise Model:** Organizer Model `QFF2026-v2`  
**Official Metric:** Balanced Accuracy  

---

## 1. Submission Overview and File Manifest

All submission requirements outlined in `QFF2026_QML_Participant_Problem_Statements_v3.docx` and the submission checklist are satisfied:

| File | Description | Status |
|------|-------------|--------|
| `solution.ipynb` | Complete, runnable Jupyter Notebook executing Task A, Task B, and PS-Q2 top-to-bottom | Verified (Runs with fixed seeds) |
| `processor_d_config.json` | 6-qubit custom Processor D design with official `benchmark_results` filled in | Validated by `qff_qml_benchmark.py` |
| `predictions_q2.csv` | Official test predictions for `hd_data_test_features.csv` (200 rows, format `sample_id,label`) | Validated by `tools/validate_predictions.py` |
| `results.md` | Comprehensive technical report based on `results_template.md` | Completed with exact metrics & analysis |
| `README.md` | Reproduction instructions, environment setup, and `pip freeze` record | This document |

---

## 2. Environment Setup (Pinned Dependencies)

The project runs on Python 3.11 with the exact pinned package versions specified by the organizers.

### Dedicated Virtual Environment Creation:
```bash
# From workspace root
python -m venv qml-env
# Activate virtual environment
# Windows:
.\qml-env\Scripts\activate
# Linux/macOS:
source qml-env/bin/activate

# Upgrade pip and install exact pinned requirements
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. End-to-End Execution and Validation Commands

To reproduce all results, execute the following commands in order:

### 1. Verify Organizer Noise Model:
```bash
python noise_model/benchmark_noise_model.py
```

### 2. Run and Validate Task B (Processor D Benchmark):
```bash
python benchmark/qff_qml_benchmark.py processor_d_config.json
```
*Expected Output:*
- QFT-3 depth (before transpile): 6
- QFT-3 depth (after transpile): 18
- Two-qubit gate count: 9
- SWAP overhead: 0
- Fidelity (ideal): 1.000000
- Fidelity (noisy): 0.668377

### 3. Validate Test Set Predictions:
```bash
python tools/validate_predictions.py q2 predictions_q2.csv
```
*Expected Output:* `OK - predictions file is valid.`

### 4. Run the Full Notebook:
```bash
jupyter notebook solution.ipynb
# or run headlessly:
jupyter nbconvert --to notebook --execute solution.ipynb --output solution_executed.ipynb
```

---

## 4. Key Results Summary

- **Task A (Bell State Fidelity |Φ+⟩):**
  - Ideal Fidelity: \(F = 1.000000 \pm 0.000000\)
  - Noisy Fidelity (`QFF2026-v2`): \(F = 0.880127 \pm 0.002592\) (\(\langle XX \rangle = 0.8191\), \(\langle YY \rangle = -0.8279\), \(\langle ZZ \rangle = 0.8735\))
- **Task B (Processor D - `qml_q2_hexaring_6q_v1`):**
  - Transpiled depth: 18, 2Q gates: 9, SWAP overhead: 0, Noisy fidelity: 0.668377 (beating 5Q linear baseline by +6.5%)
- **PS-Q2 (Classical vs Quantum Kernel on 800-sample 50D Dataset):**
  - Classical Linear SVM: Balanced Acc = \(0.7075 \pm 0.0295\)
  - Classical RBF-SVM: Balanced Acc = \(0.7154 \pm 0.0282\)
  - Classical RBF-SVM (PCA 6D): Balanced Acc = \(0.7158 \pm 0.0270\)
  - **Quantum Kernel Classifier (6Q ZZ-FeatureMap, Ideal): Balanced Acc = \(0.7175 \pm 0.0217\)**
  - Quantum Kernel Classifier (Noisy Benchmark): \(0.6500\) (evaluated on benchmark subset under 1024-shot sampling)

---

## 5. Pinned Dependency Record (`pip freeze`)

```text
annotated-types==0.8.0
anyio==4.15.1
appdirs==1.4.4
argon2-cffi==25.1.0
argon2-cffi-bindings==26.1.0
arrow==1.4.0
asttokens==3.0.2
astunparse==1.6.3
async-lru==2.4.0
attrs==26.1.0
autograd==1.9.1
autoray==0.8.2
babel==2.18.0
beautifulsoup4==4.15.0
bleach==6.4.0
cachetools==7.2.7
certifi==2026.10.23
cffi==2.1.1
charset-normalizer==3.5.2
cloudpickle==3.1.2
colorama==0.4.6
comm==0.2.3
contourpy==1.3.3
cryptography==50.0.2
cycler==0.12.1
debugpy==1.8.22
defusedxml==0.7.1
diastatic-malt==2.15.2
dill==0.4.1
executing==2.3.0
fastjsonschema==2.22.2
fonttools==4.68.0
fqdn==1.6.0
gast==0.7.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
ibm-cloud-sdk-core==3.28.1
ibm-platform-services==0.77.1
idna==3.20
ipykernel==7.4.0
ipython==9.17.1
ipython_pygments_lexers==1.1.1
isoduration==20.11.0
jedi==0.20.1
Jinja2==3.1.6
joblib==1.6.0
json5==0.17.3
jsonpointer==3.6
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter-events==0.12.1
jupyter-lsp==2.3.1
jupyter_builder==1.2.4
jupyter_client==8.10.0
jupyter_core==5.9.1
jupyter_server==2.21.1
jupyter_server_terminals==0.5.4
jupyterlab==4.6.5
jupyterlab_pygments==0.3.0
jupyterlab_server==2.28.1
kiwisolver==1.5.1
lark==1.3.1
MarkupSafe==3.0.4
matplotlib==3.11.2
matplotlib-inline==0.2.2
mistune==3.4.0
mpmath==1.3.0
narwhals==2.28.0
nbclient==0.11.0
nbconvert==7.17.2
nbformat==5.11.1
nest-asyncio2==1.7.4
networkx==3.6.1
notebook==7.6.3
notebook_shim==0.2.4
numpy==2.4.6
overrides==7.7.0
packaging==26.3
pandas==3.1.0
pandocfilters==1.5.1
parso==0.8.7
pennylane==0.44.1
PennyLane-qiskit==0.44.1
pennylane_lightning==0.45.0
pillow==13.0.0
platformdirs==4.14.0
prometheus_client==0.26.0
prompt_toolkit==3.0.53
psutil==7.2.2
pure_eval==0.2.4
pycparser==3.12
pydantic==2.14.1
pydantic_core==2.50.1
Pygments==2.21.0
PyJWT==2.15.1
pyparsing==3.3.3
pyspnego==0.12.4
python-dateutil==3.0.0
python-json-logger==4.2.0
pywinpty==3.1.2
PyYAML==6.0.3
pyzmq==27.2.0
qiskit==2.3.0
qiskit-aer==0.17.2
qiskit-ibm-runtime==0.45.1
referencing==0.37.0
requests==2.34.2
requests_ntlm==1.3.0
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rfc3987-syntax==1.1.0
rpds-py==2026.9.1
rustworkx==0.18.1
scikit-learn==1.9.1
scipy==1.17.1
scipy-openblas32==0.3.34.237.2
Send2Trash==2.1.0
six==1.17.0
soupsieve==3.0.4
sspilib==0.6.0
stack-data==0.6.3
stevedore==5.9.1
sympy==1.14.0
termcolor==3.3.0
terminado==0.18.1
threadpoolctl==3.7.0
tinycss2==1.5.1
tomlkit==0.15.2
tornado==6.6
traitlets==5.16.1
typing-inspection==0.4.4
typing_extensions==4.16.0
tzdata==2026.6
uri-template==1.3.0
urllib3==2.8.0
wcwidth==0.9.4
webcolors==25.10.0
webencodings==0.6.1
websocket-client==1.9.2
```
