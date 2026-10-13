"""Processor A / B definitions: coupling graphs, synthetic noise model, topology design search.

Phase 1 rules honoured here
    * Basis gates {rz, sx, x, cx}, optimisation level 1, seed 42, 4096 shots.
    * Identical synthetic noise on A and B:
        1q depolarizing p=0.001, 2q (cx) depolarizing p=0.005, readout error p=0.01.
    * Coupling maps are participant-designed (reference graphs are kept only as baselines).

Design-space search (our creative contribution)
    Processor A : exhaustive search over ALL connected 5-node graphs with <=5 edges and
                  max degree 3 (networkx graph atlas), which includes both reference graphs.
    Processor B : constrained random search + hill-climbing over 9-node graphs with <=12 edges
                  (same edge budget as the heavy-hex-inspired reference) and max degree 3
                  (a realistic crosstalk / frequency-collision constraint).
    Objective   : lexicographic minimisation of
                  (mean transpiled CX over the workload family {linear, ring, full} entanglers,
                   mean depth, average shortest-path length, #edges, max degree)
                  -- see `score_topology`.  The average-path tie-break rewards general routing
                  quality beyond the three tuned workloads (checked on a held-out workload).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import networkx as nx
import numpy as np
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from .problem import kernel_circuit
from .routing import BASIS_GATES, OPT_LEVEL, SEED, SHOTS, transpile_for_processor

NOISE_PARAMS = {"p_1q_depolarizing": 0.001, "p_2q_cx_depolarizing": 0.005, "p_readout": 0.01}

# ---- reference architectures from the Master Guide (used ONLY as baselines) -------------
REF_A_LINEAR = [(0, 1), (1, 2), (2, 3), (3, 4)]
REF_A_TREE = [(0, 1), (1, 2), (1, 3), (3, 4)]
REF_B_HEAVYHEX = [(0, 1), (1, 2), (1, 6), (2, 3), (2, 6), (2, 5), (3, 4), (4, 5), (5, 6), (6, 7), (6, 8), (8, 0)]
GRID_3X3 = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8)]

WORKLOAD_FAMILY = ["linear", "ring", "full"]  # entangler graphs the topology is tuned for
HELD_OUT_WORKLOAD = "star"                    # never used in the search -> over-fitting check
SEARCH_REPS = 2


# --------------------------------------------------------------------------------------
# Noise model
# --------------------------------------------------------------------------------------

def build_noise_model(scale: float = 1.0, noisy_rz: bool = False) -> NoiseModel:
    """Official synthetic noise model, optionally scaled (scale=1 -> spec values).

    rz is a virtual (frame-change) gate on superconducting hardware, so by default it is
    noise-free; sx and x carry the 1q depolarizing error.  Set noisy_rz=True to also
    attach it to rz (pessimistic variant)."""
    p1 = min(NOISE_PARAMS["p_1q_depolarizing"] * scale, 1.0)
    p2 = min(NOISE_PARAMS["p_2q_cx_depolarizing"] * scale, 1.0)
    pr = min(NOISE_PARAMS["p_readout"] * scale, 0.5)
    nm = NoiseModel(basis_gates=BASIS_GATES)
    if scale > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["sx", "x"] + (["rz"] if noisy_rz else []))
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
        nm.add_all_qubit_readout_error(ReadoutError([[1 - pr, pr], [pr, 1 - pr]]))
    return nm


# --------------------------------------------------------------------------------------
# Graph helpers
# --------------------------------------------------------------------------------------

def to_graph(edges, n):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    return g


def canon(edges):
    return sorted({tuple(sorted(e)) for e in edges})


def topology_stats(edges, n):
    g = to_graph(edges, n)
    degs = [d for _, d in g.degree()]
    cyc = [c for c in nx.simple_cycles(g, length_bound=4)]
    return {
        "num_qubits": n,
        "num_edges": g.number_of_edges(),
        "density": round(2 * g.number_of_edges() / (n * (n - 1)), 4),
        "degree_sequence": sorted(degs, reverse=True),
        "max_degree": max(degs),
        "diameter": nx.diameter(g),
        "avg_shortest_path": round(nx.average_shortest_path_length(g), 4),
        "algebraic_connectivity": round(float(nx.algebraic_connectivity(g)), 4),
        "num_triangles": sum(nx.triangles(g).values()) // 3,
        "num_4_cycles": sum(1 for c in cyc if len(c) == 4),
    }


# --------------------------------------------------------------------------------------
# Topology scoring / search
# --------------------------------------------------------------------------------------

def score_topology(edges, n, workloads=tuple(WORKLOAD_FAMILY), reps=SEARCH_REPS):
    """Mean CX / depth / SWAPs of the kernel circuit over the workload family + sort key."""
    cx, depth, swaps = [], [], []
    for ent in workloads:
        qc, _, _ = kernel_circuit(ent, reps)
        m = transpile_for_processor(qc, edges, n)["metrics"]
        cx.append(m["cx"]); depth.append(m["depth"]); swaps.append(m["swaps"])
    g = to_graph(edges, n)
    maxdeg = max(d for _, d in g.degree())
    out = {
        "mean_cx": float(np.mean(cx)), "mean_depth": float(np.mean(depth)), "mean_swaps": float(np.mean(swaps)),
        "per_workload_cx": dict(zip(workloads, cx)), "per_workload_depth": dict(zip(workloads, depth)),
        "num_edges": g.number_of_edges(), "max_degree": maxdeg,
        "avg_shortest_path": float(nx.average_shortest_path_length(g)),
    }
    out["key"] = (out["mean_cx"], round(out["mean_depth"], 2), round(out["avg_shortest_path"], 4),
                  out["num_edges"], maxdeg)
    return out


def search_processor_A(max_edges=5, max_degree=3):
    """Exhaustive search over every connected 5-node graph (networkx atlas)."""
    rows = []
    for g in nx.graph_atlas_g():
        if g.number_of_nodes() != 5 or not nx.is_connected(g):
            continue
        if g.number_of_edges() > max_edges or max(d for _, d in g.degree()) > max_degree:
            continue
        e = canon(g.edges())
        rows.append({"name": "atlas", "edges": e, **score_topology(e, 5)})
    rows.sort(key=lambda r: r["key"])
    return rows


def _random_graph(rng, n, n_edges, max_degree):
    """Random connected graph: n nodes, n_edges edges, max degree <= max_degree."""
    while True:
        deg = [0] * n
        order = list(range(n)); rng.shuffle(order)
        edges, ok = set(), True
        for k in range(1, n):  # random tree by random attachment
            cands = [order[j] for j in range(k) if deg[order[j]] < max_degree]
            if not cands:
                ok = False; break
            p, c = rng.choice(cands), order[k]
            edges.add(tuple(sorted((p, c)))); deg[p] += 1; deg[c] += 1
        if not ok:
            continue
        tries = 0
        while len(edges) < n_edges and tries < 500:
            a, b = rng.sample(range(n), 2)
            e = tuple(sorted((a, b)))
            if e not in edges and deg[a] < max_degree and deg[b] < max_degree:
                edges.add(e); deg[a] += 1; deg[b] += 1
            tries += 1
        if len(edges) == n_edges:
            return sorted(edges)


def _mutate(rng, edges, n, max_degree):
    """Edge-rewire move keeping the graph connected and degree-bounded."""
    for _ in range(200):
        e = rng.choice(edges)
        rest = [x for x in edges if x != e]
        if not nx.is_connected(to_graph(rest, n)):
            continue
        deg = [0] * n
        for a, b in rest:
            deg[a] += 1; deg[b] += 1
        a, b = rng.sample(range(n), 2)
        ne = tuple(sorted((a, b)))
        if ne in rest or ne == e or deg[a] >= max_degree or deg[b] >= max_degree:
            continue
        return sorted(rest + [ne])
    return edges


def search_processor_B(n=9, n_edges=12, max_degree=3, n_random=250, n_restarts=3, n_iters=120, seed=SEED):
    """Random search + hill climbing. Returns (best_edges, leaderboard_rows, climb_trace)."""
    rng = random.Random(seed)
    pool = {"reference_heavy_hex": canon(REF_B_HEAVYHEX), "grid_3x3": canon(GRID_3X3)}
    rows = [{"name": k, "edges": e, **score_topology(e, n)} for k, e in pool.items()]
    seen = {tuple(map(tuple, e)) for e in pool.values()}
    for k in range(n_random):
        e = _random_graph(rng, n, n_edges, max_degree)
        if tuple(map(tuple, e)) in seen:
            continue
        seen.add(tuple(map(tuple, e)))
        rows.append({"name": f"random_{k:03d}", "edges": e, **score_topology(e, n)})
    rows.sort(key=lambda r: r["key"])
    # Baselines (reference heavy-hex has a degree-5 hub, grid has degree 4) are reported for
    # comparison but are NOT feasible under the max-degree constraint -> never search seeds.
    feasible = [r for r in rows if r["max_degree"] <= max_degree]
    trace, climbed = [], []
    for r_i, start in enumerate(feasible[:n_restarts]):
        cur, cur_s = start["edges"], start
        for it in range(n_iters):
            cand = _mutate(rng, cur, n, max_degree)
            if cand == cur:
                continue
            s = score_topology(cand, n)
            if s["key"] <= cur_s["key"]:
                cur, cur_s = cand, {"name": f"hillclimb_{r_i}", "edges": cand, **s}
            trace.append({"restart": r_i, "iter": it, "mean_cx": cur_s["mean_cx"], "mean_depth": cur_s["mean_depth"]})
        climbed.append(cur_s)
    rows = sorted(rows + climbed, key=lambda r: r["key"])
    best = [r for r in rows if r["max_degree"] <= max_degree][0]
    return best["edges"], rows, trace


# --------------------------------------------------------------------------------------
# JSON persistence (processors/processor_A.json, processor_B.json)
# --------------------------------------------------------------------------------------

def save_processor_json(path, name, role, edges, n, rationale, extra=None):
    edges = canon(edges)
    cfg = {
        "name": name,
        "role": role,
        "num_qubits": n,
        "coupling_map_undirected": [list(e) for e in edges],
        "coupling_map_directed_both_ways": [list(e) for e in edges] + [[b, a] for a, b in edges],
        "basis_gates": BASIS_GATES,
        "transpile": {"optimization_level": OPT_LEVEL, "seed_transpiler": SEED, "shots": SHOTS},
        "synthetic_noise_model": {
            **NOISE_PARAMS,
            "notes": "identical on Processor A and B; rz is virtual (noise-free); symmetric readout error on every qubit",
        },
        "topology_stats": topology_stats(edges, n),
        "design_rationale": rationale,
    }
    if extra:
        cfg.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def load_processor(path):
    """Return (name, undirected_edges, n_qubits, config) from a processor JSON file."""
    with open(path) as f:
        cfg = json.load(f)
    return cfg["name"], [tuple(e) for e in cfg["coupling_map_undirected"]], cfg["num_qubits"], cfg
