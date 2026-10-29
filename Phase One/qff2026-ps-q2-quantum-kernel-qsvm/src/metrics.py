"""Circuit metrics (depth, CX, SWAP, fidelity proxies) and kernel / classifier metrics."""
from __future__ import annotations

import numpy as np

_NON_GATES = {"measure", "barrier"}


# --------------------------------------------------------------------------------------
# Circuit-level metrics
# --------------------------------------------------------------------------------------

def circuit_metrics(tc, noise_params=None):
    """Metrics of a transpiled circuit in the {rz, sx, x, cx} basis.

    depth / cx_depth exclude measurements & barriers.  `est_gate_fidelity` is the
    product of (1-p) over every noisy gate (global-depolarising damping factor)."""
    ops = tc.count_ops()
    n_cx = int(ops.get("cx", 0))
    n_sx, n_x, n_rz = int(ops.get("sx", 0)), int(ops.get("x", 0)), int(ops.get("rz", 0))
    used = set()
    for inst in tc.data:
        if inst.operation.name in _NON_GATES:
            continue
        used.update(tc.find_bit(q).index for q in inst.qubits)
    m = {
        "depth": int(tc.depth(lambda i: i.operation.name not in _NON_GATES)),
        "cx_depth": int(tc.depth(lambda i: i.operation.name == "cx")),
        "cx": n_cx, "sx": n_sx, "x": n_x, "rz": n_rz,
        "one_qubit_gates": n_sx + n_x + n_rz,
        "total_gates": n_cx + n_sx + n_x + n_rz,
        "active_physical_qubits": len(used),
        "swaps": int(ops.get("swap", 0)),
    }
    if noise_params is not None:
        p1, p2, pr = (noise_params["p_1q_depolarizing"], noise_params["p_2q_cx_depolarizing"],
                      noise_params["p_readout"])
        m["est_gate_fidelity"] = float((1 - p1) ** (n_sx + n_x) * (1 - p2) ** n_cx)
        m["est_slope"] = float(m["est_gate_fidelity"] * (1 - pr) ** 4)  # predicted contraction of k
    return m


# --------------------------------------------------------------------------------------
# Kernel-level metrics
# --------------------------------------------------------------------------------------

def offdiag(K):
    iu = np.triu_indices_from(K, k=1)
    return K[iu]


def standard_error(k, shots):
    """Binomial standard error of an estimated kernel entry (shots-limited)."""
    k = np.clip(k, 0, 1)
    return np.sqrt(k * (1 - k) / shots)


def kernel_error_stats(K_est, K_ref):
    d = offdiag(K_est) - offdiag(K_ref)
    return {
        "kernel_MAE": float(np.mean(np.abs(d))),
        "kernel_RMSE": float(np.sqrt(np.mean(d**2))),
        "kernel_max_abs_err": float(np.max(np.abs(d))),
        "kernel_rel_frobenius": float(np.linalg.norm(K_est - K_ref) / np.linalg.norm(K_ref)),
    }


def affine_fit(K_est, K_ref):
    """Fit K_est ~= a*K_ref + b (off-diagonals). Depolarising noise predicts a~F, b~(1-F)/16."""
    a, b = np.polyfit(offdiag(K_ref), offdiag(K_est), 1)
    return float(a), float(b)


def kernel_target_alignment(K, y, centered=True):
    """KTA between K and the label kernel y y^T (y in {-1,+1})."""
    ypm = 2 * np.asarray(y) - 1
    Y = np.outer(ypm, ypm).astype(float)
    if centered:
        n = len(K)
        H = np.eye(n) - np.ones((n, n)) / n
        K, Y = H @ K @ H, H @ Y @ H
    return float(np.sum(K * Y) / (np.linalg.norm(K) * np.linalg.norm(Y)))


def spectral_stats(K):
    w = np.linalg.eigvalsh((K + K.T) / 2)
    return {"min_eigenvalue": float(w.min()), "neg_eig_mass": float(-w[w < 0].sum()),
            "effective_rank": float(np.exp(-np.sum((w.clip(0) / w.clip(0).sum()) * np.log(w.clip(1e-15) / w.clip(0).sum()))))}


def concentration_stats(K):
    o = offdiag(K)
    return {"offdiag_mean": float(o.mean()), "offdiag_std": float(o.std())}


def wilson_interval(correct, n, z=1.96):
    """95% Wilson score interval for an accuracy (important: n_test is only 20)."""
    if n == 0:
        return (0.0, 1.0)
    p = correct / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return float(c - h), float(c + h)
