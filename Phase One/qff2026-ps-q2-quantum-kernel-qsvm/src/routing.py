"""Transpilation / layout routing routines and kernel execution on a processor model."""
from __future__ import annotations

import time

import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

from .metrics import circuit_metrics

BASIS_GATES = ["rz", "sx", "x", "cx"]
OPT_LEVEL = 1
SEED = 42
SHOTS = 4096


def bidirectional(edges):
    e = [list(map(int, x)) for x in edges]
    return e + [[b, a] for a, b in e]


def transpile_for_processor(qc, edges, n_qubits, seed=SEED, level=OPT_LEVEL, noise_params=None):
    """Transpile `qc` onto the coupling graph (basis {rz,sx,x,cx}, level 1, seed 42).

    Two transpilations with the SAME seed are run: one to the official basis (depth / CX
    metrics, the circuit actually simulated) and one that additionally keeps `swap` as a
    basis gate so that the number of routing SWAPs inserted by the router is observable."""
    cmap = CouplingMap(bidirectional(edges))
    tc = transpile(qc, coupling_map=cmap, basis_gates=BASIS_GATES,
                   optimization_level=level, seed_transpiler=seed)
    ts = transpile(qc, coupling_map=cmap, basis_gates=BASIS_GATES + ["swap"],
                   optimization_level=level, seed_transpiler=seed)
    m = circuit_metrics(tc, noise_params)
    m["swaps"] = int(ts.count_ops().get("swap", 0))
    try:
        layout = list(tc.layout.initial_index_layout(filter_ancillas=True))
    except Exception:  # pragma: no cover
        layout = None
    return {"circuit": tc, "metrics": m, "layout": layout}


def _bind(tc, xs, ys, theta, i, j):
    d = dict(zip(xs, theta[i]))
    d.update(zip(ys, theta[j]))
    return tc.assign_parameters(d)


def run_kernel_pairs(tc, xs, ys, theta, pairs, backend, shots=SHOTS, chunk=300, seed=SEED):
    """Estimate k(x_i, x_j) = P(|0000>) for each (i, j) pair by sampling the (noisy) backend."""
    est = np.zeros(len(pairs))
    zero = "0" * len(tc.cregs[0]) if tc.cregs else "0000"
    t0 = time.perf_counter()
    for s in range(0, len(pairs), chunk):
        sub = pairs[s:s + chunk]
        circs = [_bind(tc, xs, ys, theta, i, j) for i, j in sub]
        res = backend.run(circs, shots=shots, seed_simulator=seed + s).result()
        for k in range(len(sub)):
            est[s + k] = res.get_counts(k).get(zero, 0) / shots
    return est, time.perf_counter() - t0


def gram_from_pairs(pairs, values, n):
    """Symmetric Gram matrix (diagonal := 1, see ADR-006)."""
    K = np.eye(n)
    for (i, j), v in zip(pairs, values):
        K[i, j] = K[j, i] = v
    return K


def all_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def noisy_backend(noise_model):
    return AerSimulator(noise_model=noise_model)


def ideal_backend():
    return AerSimulator()
