"""Problem formulation for PS-Q2 (Quantum Kernel Estimator & QSVM).

Task (from the Phase 1 Master Guide)
    * Dataset  : sklearn.datasets.make_moons(n_samples=100, noise=0.1, random_state=42)
    * Kernel   : fidelity overlap  k(x, x') = |<phi(x)|phi(x')>|^2  on a 4-qubit register
    * Classifier: classical SVM trained on the (precomputed) quantum kernel matrix

Design notes
    * make_moons has only 2 raw features but the register must be 4 qubits wide, so we
      use a *polynomial lift* f(x) = [x0, x1, x0*x1, x0^2 - x1^2] (after MinMax scaling
      to [0,1]) and one lifted feature per qubit.  gamma is a bandwidth hyper-parameter.
    * The feature map is a ZZFeatureMap-style circuit (H, P(2a_i), CX-P-CX entanglers)
      with a *configurable entangler graph* (linear / ring / full / star).  The entangler
      graph is the knob that couples the algorithm to the hardware coupling map.
    * Hyper-parameters are selected by cross-validation on the TRAINING split only,
      using the exact (noise-free) kernel, with a cost-aware tie-break that prefers
      cheaper (fewer-CX) feature maps.  The test split is never used for selection.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from sklearn.datasets import make_moons
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

SEED = 42
N_SAMPLES = 100
DATA_NOISE = 0.1
N_QUBITS = 4
SHOTS = 4096
TEST_SIZE = 0.2

# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

def make_dataset():
    """Exact dataset required by PS-Q2."""
    return make_moons(n_samples=N_SAMPLES, noise=DATA_NOISE, random_state=SEED)


def split_indices(y):
    """Stratified 80/20 split (indices into the 100-sample dataset)."""
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    return np.sort(tr), np.sort(te)


def scale_features(X, train_idx):
    """MinMax scaler fitted on the training rows only (no test leakage)."""
    scaler = MinMaxScaler(feature_range=(0.0, 1.0)).fit(X[train_idx])
    return scaler.transform(X)


def lift_features(X01):
    """2 -> 4 polynomial lift: one lifted feature per qubit."""
    x0, x1 = X01[:, 0], X01[:, 1]
    return np.column_stack([x0, x1, x0 * x1, x0**2 - x1**2])


def feature_angles(X01, gamma):
    """Rotation angles a_i = gamma * f_i(x) fed to the feature map, shape (N, 4)."""
    return gamma * lift_features(X01)


# --------------------------------------------------------------------------------------
# Feature map / kernel circuits
# --------------------------------------------------------------------------------------

def entangler_pairs(kind: str, n: int = N_QUBITS):
    """Logical entangler graph used by the feature map."""
    if kind == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if kind == "ring":
        return [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]
    if kind == "full":
        return list(itertools.combinations(range(n), 2))
    if kind == "star":  # held-out workload used only to test topology over-fitting
        return [(0, j) for j in range(1, n)]
    raise ValueError(f"unknown entanglement '{kind}'")


def feature_map(theta, entanglement="ring", reps=2) -> QuantumCircuit:
    """ZZFeatureMap-style circuit U(theta) with a configurable entangler graph."""
    qc = QuantumCircuit(N_QUBITS, name=f"FM[{entanglement},r{reps}]")
    pairs = entangler_pairs(entanglement)
    for _ in range(reps):
        qc.h(range(N_QUBITS))
        for i in range(N_QUBITS):
            qc.p(2 * theta[i], i)
        for i, j in pairs:
            qc.cx(i, j)
            qc.p(2 * (np.pi - theta[i]) * (np.pi - theta[j]), j)
            qc.cx(i, j)
    return qc


def kernel_circuit(entanglement="ring", reps=2, measure=True):
    """Compute-uncompute circuit  U(x)  followed by  U(x')^dagger.

    P(measuring |0000>) = |<phi(x')|phi(x)>|^2 = k(x, x').
    Returns (circuit, x_params, y_params); parameters are bound per data pair so the
    circuit is transpiled/routed ONCE per processor (as on real hardware)."""
    xs, ys = ParameterVector("x", N_QUBITS), ParameterVector("y", N_QUBITS)
    qc = feature_map(xs, entanglement, reps).compose(feature_map(ys, entanglement, reps).inverse())
    if measure:
        qc.measure_all()
    return qc, xs, ys


def ideal_states(theta_matrix, entanglement="ring", reps=2):
    """Exact statevectors |phi(x_i)> for all rows, shape (N, 16)."""
    xs = ParameterVector("t", N_QUBITS)
    fm = feature_map(xs, entanglement, reps)
    return np.array(
        [Statevector(fm.assign_parameters(dict(zip(xs, row)))).data for row in theta_matrix]
    )


def ideal_gram(theta_matrix, entanglement="ring", reps=2):
    """Exact fidelity-kernel Gram matrix |<phi(x_i)|phi(x_j)>|^2."""
    S = ideal_states(theta_matrix, entanglement, reps)
    return np.abs(S.conj() @ S.T) ** 2


def logical_cx_count(entanglement="ring", reps=2):
    """CX count of the *logical* (un-routed) kernel circuit = 2 (U and U^dagger) * reps * 2/pair."""
    return 2 * reps * 2 * len(entangler_pairs(entanglement))


# --------------------------------------------------------------------------------------
# Classical SVM on precomputed kernels + model selection
# --------------------------------------------------------------------------------------

def svm_accuracy(K, y, train_idx, test_idx, C=1.0):
    """Train SVC(kernel='precomputed') on K[train,train]; return (accuracy, predictions)."""
    clf = SVC(kernel="precomputed", C=C).fit(K[np.ix_(train_idx, train_idx)], y[train_idx])
    pred = clf.predict(K[np.ix_(test_idx, train_idx)])
    return float((pred == y[test_idx]).mean()), pred


def cv_accuracy(K, y, idx, C, n_splits=5, n_repeats=3, seed=SEED):
    """Repeated stratified k-fold accuracy using only rows in `idx`."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    accs = []
    for a, b in rskf.split(idx, y[idx]):
        accs.append(svm_accuracy(K, y, idx[a], idx[b], C)[0])
    return float(np.mean(accs)), float(np.std(accs))


@dataclass
class ModelChoice:
    entanglement: str
    reps: int
    gamma: float
    C: float
    cv_mean: float
    cv_std: float
    logical_cx: int


def select_model(X01_all, y, train_idx, entanglements=("linear", "ring", "full"),
                 reps_grid=(1, 2), gamma_grid=(0.5, 1.0, 2.0, 3.0), C_grid=(0.1, 1.0, 10.0, 100.0),
                 tolerance=0.01):
    """Grid search on TRAIN-only CV with the exact kernel.

    Cost-aware rule: among all configs within `tolerance` of the best CV accuracy, pick the
    one with the fewest logical CX gates (then smallest reps, then smallest C)."""
    rows = []
    for ent in entanglements:
        for reps in reps_grid:
            for gamma in gamma_grid:
                K = ideal_gram(feature_angles(X01_all, gamma), ent, reps)
                for C in C_grid:
                    m, s = cv_accuracy(K, y, train_idx, C)
                    rows.append(ModelChoice(ent, reps, gamma, C, m, s, logical_cx_count(ent, reps)))
    best = max(r.cv_mean for r in rows)
    pool = [r for r in rows if r.cv_mean >= best - tolerance]
    pool.sort(key=lambda r: (r.logical_cx, r.reps, r.C, -r.cv_mean))
    return pool[0], rows
