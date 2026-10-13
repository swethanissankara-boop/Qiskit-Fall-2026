"""Plotting helpers: coupling-graph diagrams, kernel heat-maps, benchmark bars, scatter plots."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .metrics import offdiag

PALETTE = {"ideal": "#4c566a", "shot": "#8fbcbb", "A": "#2f6fed", "B": "#2fa84f", "ref": "#bf616a"}


def draw_coupling(ax, edges, n, title="", used=None, color="#2f6fed", pos=None):
    g = nx.Graph(); g.add_nodes_from(range(n)); g.add_edges_from(edges)
    pos = pos or nx.kamada_kawai_layout(g)
    cols = [color if (used is None or v in used) else "#d8dee9" for v in g.nodes]
    nx.draw_networkx_edges(g, pos, ax=ax, width=2, edge_color="#6b7280")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=cols, node_size=420, edgecolors="black")
    nx.draw_networkx_labels(g, pos, ax=ax, font_color="white", font_size=9, font_weight="bold")
    ax.set_title(title, fontsize=10); ax.axis("off")
    return pos


def plot_dataset(X, y, train_idx, test_idx, path):
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for c, col in [(0, "#2f6fed"), (1, "#e08a1e")]:
        m = y == c
        ax.scatter(X[m & np.isin(np.arange(len(y)), train_idx), 0], X[m & np.isin(np.arange(len(y)), train_idx), 1],
                   c=col, s=28, label=f"class {c} (train)")
        ax.scatter(X[m & np.isin(np.arange(len(y)), test_idx), 0], X[m & np.isin(np.arange(len(y)), test_idx), 1],
                   c=col, s=70, marker="*", edgecolors="k", label=f"class {c} (test)")
    ax.set_title("make_moons(n=100, noise=0.1, seed=42)"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(path, dpi=160); plt.close(fig)


def plot_design_space(rowsA, rowsB, trace, path):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    ax = axs[0]
    labels = [f"A{i}" for i in range(len(rowsA))]
    ax.bar(labels, [r["mean_cx"] for r in rowsA], color=[PALETTE["A"] if i == 0 else "#9db7f5" for i in range(len(rowsA))])
    ax.set_ylabel("mean routed CX (linear/ring/full)"); ax.set_title("Processor A: exhaustive search (all 6 graphs)")
    ax = axs[1]
    ax.hist([r["mean_cx"] for r in rowsB if r["name"].startswith("random")], bins=18, color="#a3d9b1", edgecolor="k")
    for r in rowsB:
        if r["name"] == "reference_heavy_hex":
            ax.axvline(r["mean_cx"], color=PALETTE["ref"], lw=2, label="reference heavy-hex")
        if r["name"] == "grid_3x3":
            ax.axvline(r["mean_cx"], color="#d08770", lw=2, ls="--", label="3x3 grid")
    ax.axvline(rowsB[0]["mean_cx"], color=PALETTE["B"], lw=2.5, label="custom B (best)")
    ax.set_xlabel("mean routed CX"); ax.set_title("Processor B: random-search population"); ax.legend(fontsize=7)
    ax = axs[2]
    for r in sorted({t["restart"] for t in trace}):
        tt = [t for t in trace if t["restart"] == r]
        ax.plot([t["iter"] for t in tt], [t["mean_cx"] for t in tt], label=f"restart {r}")
    ax.set_xlabel("hill-climb iteration"); ax.set_ylabel("best mean CX"); ax.set_title("Processor B: hill climbing"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_kernel_heatmaps(Ks, titles, ref_idx, path, vmax_err=None):
    fig, axs = plt.subplots(1, len(Ks) + 1, figsize=(3.6 * (len(Ks) + 1), 3.5))
    for ax, K, t in zip(axs, Ks, titles):
        im = ax.imshow(K, vmin=0, vmax=1, cmap="viridis"); ax.set_title(t, fontsize=9); ax.axis("off")
    fig.colorbar(im, ax=axs[:-1], fraction=0.02)
    err = np.abs(Ks[-1] - Ks[ref_idx])
    im2 = axs[-1].imshow(err, cmap="magma", vmax=vmax_err); axs[-1].set_title(f"|{titles[-1]} - {titles[ref_idx]}|", fontsize=9)
    axs[-1].axis("off"); fig.colorbar(im2, ax=axs[-1], fraction=0.046)
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def plot_metric_bars(df, metrics, path, group_col="topology", hue_col="workload", title=""):
    fig, axs = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4))
    axs = np.atleast_1d(axs)
    groups = list(dict.fromkeys(df[group_col])); hues = list(dict.fromkeys(df[hue_col]))
    w = 0.8 / len(hues)
    for ax, m in zip(axs, metrics):
        for h_i, h in enumerate(hues):
            vals = [df[(df[group_col] == g) & (df[hue_col] == h)][m].values[0] for g in groups]
            ax.bar(np.arange(len(groups)) + h_i * w, vals, w, label=h)
        ax.set_xticks(np.arange(len(groups)) + w * (len(hues) - 1) / 2); ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
        ax.set_title(m); ax.grid(axis="y", alpha=.3)
    axs[0].legend(title=hue_col, fontsize=7); fig.suptitle(title, fontsize=11); fig.tight_layout()
    fig.savefig(path, dpi=160); plt.close(fig)


def plot_scatter_affine(K_ideal, Ks, labels, fits, preds, path):
    fig, axs = plt.subplots(1, len(Ks), figsize=(4.6 * len(Ks), 4.2), sharey=True)
    axs = np.atleast_1d(axs)
    xi = offdiag(K_ideal)
    for ax, K, lab, (a, b), (pa, pb) in zip(axs, Ks, labels, fits, preds):
        ax.scatter(xi, offdiag(K), s=4, alpha=.35, color=PALETTE.get(lab[-1], "#555"))
        xs = np.linspace(0, 1, 50)
        ax.plot(xs, xs, "k--", lw=1, label="y = x")
        ax.plot(xs, a * xs + b, color="crimson", label=f"fit: {a:.3f}k+{b:.3f}")
        ax.plot(xs, pa * xs + pb, color="orange", ls=":", label=f"model: {pa:.3f}k+{pb:.3f}")
        ax.set_xlabel("exact kernel"); ax.set_title(lab, fontsize=10); ax.legend(fontsize=7)
    axs[0].set_ylabel("estimated kernel"); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_noise_sweep(df, path):
    fig, ax = plt.subplots(figsize=(5.4, 4))
    for (proc, wl), g in df.groupby(["processor", "workload"]):
        ax.plot(g["scale"], g["kernel_MAE"], "o-", label=f"{proc} / {wl}", color=PALETTE[proc], ls="-" if wl == "full" else "--")
    ax.set_xlabel("noise scale (1 = official spec)"); ax.set_ylabel("kernel MAE vs exact")
    ax.set_title("Kernel error vs noise strength"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_accuracy_bars(df, path):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    names = list(df["backend"]); x = np.arange(len(names))
    ax.bar(x, df["test_acc"], yerr=[df["test_acc"] - df["ci_lo"], df["ci_hi"] - df["test_acc"]], capsize=4,
           color=[PALETTE.get(n.split()[-1], "#8fbcbb") if n.split()[-1] in ("A", "B") else "#88c0d0" for n in names])
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("test accuracy (n=20, 95% Wilson CI)"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_decision_regions(clf_predict, X, y, test_idx, path, title="", grid=120):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .3, X[:, 0].max() + .3, grid),
                         np.linspace(X[:, 1].min() - .3, X[:, 1].max() + .3, grid))
    Z = clf_predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.contourf(xx, yy, Z, alpha=.25, cmap="coolwarm", levels=[-.5, .5, 1.5])
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=22, edgecolors="k", linewidths=.3)
    ax.scatter(X[test_idx, 0], X[test_idx, 1], facecolors="none", edgecolors="k", s=90, label="test")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
