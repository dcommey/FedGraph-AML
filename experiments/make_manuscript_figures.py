"""Manuscript figures from the audited refinement run grid.

Reads only saved analysis exports (no training) and writes vector PDFs and
300-dpi PNGs. Run from the repository root:

    python experiments/make_manuscript_figures.py --analysis results/analysis --output figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "analysis"
OUT = ROOT / "figures"

# Okabe-Ito colour-blind-safe palette.
BLUE, ORANGE, GREEN, VERMILION, PURPLE, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#7F7F7F")
STRATEGY_STYLE = {"random": (BLUE, "o", "Random ownership"),
                  "louvain": (ORANGE, "s", "Louvain ownership")}
DATASET_TITLE = {"elliptic": "Elliptic (exploratory)", "ibm": "IBM HI-Small (primary)"}
PREVALENCE = {"elliptic": 636 / 11184, "ibm": 191 / 172556}

METHOD_ORDER = [  # (key, label, group)
    ("local", "Local SAGE", "Federated graph"),
    ("fedavg", "FedAvg (SAGE)", "Federated graph"),
    ("current", "Round exchange", "Federated graph"),
    ("fresh", "Step exchange", "Federated graph"),
    ("shuffled", "Shuffled exchange", "Federated graph"),
    ("oracle", "Raw-feature oracle", "Federated graph"),
    ("fedgcn", "FedGCN adapter", "Federated graph"),
    ("fedmlp", "FedMLP", "Graph-free"),
    ("rf_local", "Local RF", "Graph-free"),
    ("xgb_local", "Local XGBoost", "Graph-free"),
    ("centralized", "Pooled SAGE", "Pooled data"),
    ("mlp", "Pooled MLP", "Pooled data"),
    ("rf_pooled", "Pooled RF", "Pooled data"),
    ("xgb_pooled", "Pooled XGBoost", "Pooled data"),
]
POOLED = {"centralized", "mlp", "rf_pooled", "xgb_pooled"}


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino", "STIX Two Text", "Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def grid():
    g = pd.read_csv(ANALYSIS / "run_grid.csv")
    # Pooled controls are trained once and reused for both ownership strategies.
    pooled = g[g.method.isin(POOLED)]
    return pd.concat([g, pooled.assign(strategy="louvain")], ignore_index=True)


def fig_performance(g):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.1), sharey=True)
    y = np.arange(len(METHOD_ORDER))[::-1]
    rng = np.random.default_rng(0)
    for ax, ds in zip(axes, ["elliptic", "ibm"]):
        d = g[g.dataset == ds]
        fedavg = d[(d.method == "fedavg") & (d.strategy == "random")].ap.mean()
        ax.axvline(PREVALENCE[ds], color=GREY, lw=0.7, ls=":")
        ax.axvline(fedavg, color=BLUE, lw=0.6, ls="--", alpha=0.6)
        for i, (m, _, _) in enumerate(METHOD_ORDER):
            for off, s in [(0.17, "random"), (-0.17, "louvain")]:
                c, mk, _ = STRATEGY_STYLE[s]
                v = d[(d.method == m) & (d.strategy == s)].ap.to_numpy()
                jit = rng.uniform(-0.07, 0.07, len(v))
                ax.scatter(v, y[i] + off + jit, s=5, color=c, alpha=0.30, lw=0, zorder=2)
                ax.scatter(v.mean(), y[i] + off, s=26, marker=mk, color=c,
                           edgecolor="black", lw=0.5, zorder=3)
        for b in (y[6] - 0.5, y[9] - 0.5):
            ax.axhline(b, color="black", lw=0.4, alpha=0.4)
        ax.set_title(DATASET_TITLE[ds])
        ax.set_xlabel("Test average precision (raw scores)")
        ax.set_xlim(left=0)
        ax.grid(axis="x", lw=0.3, alpha=0.4)
    axes[0].set_yticks(y, [lab for _, lab, _ in METHOD_ORDER])
    axes[0].set_ylim(-0.6, len(METHOD_ORDER) - 0.4)
    for grp, yy in [("Federated\ngraph", y[3]), ("Graph-free", y[8]), ("Pooled\ndata", y[11] - 0.5)]:
        axes[1].text(1.02, yy, grp, transform=axes[1].get_yaxis_transform(),
                     rotation=90, va="center", ha="left", fontsize=7.5, color="0.35")
    handles = [Line2D([], [], marker=STRATEGY_STYLE[s][1], color=STRATEGY_STYLE[s][0], ls="",
                      mec="black", mew=0.5, ms=5, label=STRATEGY_STYLE[s][2])
               for s in ("random", "louvain")]
    handles += [Line2D([], [], color=BLUE, ls="--", lw=0.6, label="FedAvg mean (random)"),
                Line2D([], [], color=GREY, ls=":", lw=0.7, label="Test prevalence")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 0.97, 1))
    save(fig, "fig_performance")


def paired_effects(g):
    keys = ["gpu", "dataset", "strategy", "ownership_seed", "seed"]
    step = g[g.method == "fresh"].set_index(keys).ap
    base = g[g.method == "fedavg"].set_index(keys).ap
    return (step - base).rename("effect").reset_index()


def fig_effects(g):
    e = paired_effects(g)
    draws = sorted(e.ownership_seed.unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, ds in zip(axes, ["elliptic", "ibm"]):
        ax.axhline(0, color="black", lw=0.6)
        for si, s in enumerate(["random", "louvain"]):
            c, mk, _ = STRATEGY_STYLE[s]
            for di, ow in enumerate(draws):
                x0 = di + (si - 0.5) * 0.36
                cell = e[(e.dataset == ds) & (e.strategy == s) & (e.ownership_seed == ow)]
                for gpu, gm, dx in [("gtx", "^", -0.06), ("rtx", "v", 0.06)]:
                    v = cell[cell.gpu == gpu].effect.to_numpy()
                    ax.scatter(np.full(len(v), x0 + dx), v, s=9, marker=gm, color=c,
                               alpha=0.45, lw=0, zorder=2)
                ax.scatter(x0, cell.effect.mean(), s=40, marker=mk, color=c,
                           edgecolor="black", lw=0.6, zorder=3)
            means = [e[(e.dataset == ds) & (e.strategy == s) & (e.ownership_seed == ow)].effect.mean()
                     for ow in draws]
            ax.axhline(np.mean(means), color=c, lw=0.8, ls="--", alpha=0.8)
        ax.set_xticks(range(len(draws)), [f"Draw {i + 1}" for i in range(len(draws))])
        ax.set_xlim(-0.6, len(draws) - 0.4)
        ax.set_title(DATASET_TITLE[ds])
        ax.set_ylabel(r"$\Delta$AP (step exchange $-$ FedAvg)")
        ax.grid(axis="y", lw=0.3, alpha=0.4)
    handles = [Line2D([], [], marker=STRATEGY_STYLE[s][1], color=STRATEGY_STYLE[s][0], ls="",
                      mec="black", mew=0.5, ms=5.5, label=f"{STRATEGY_STYLE[s][2]}: draw mean")
               for s in ("random", "louvain")]
    handles += [Line2D([], [], marker="^", color="0.45", ls="", ms=4, alpha=0.7, label="GTX run pair"),
                Line2D([], [], marker="v", color="0.45", ls="", ms=4, alpha=0.7, label="RTX run pair"),
                Line2D([], [], color="0.3", ls="--", lw=0.8, label="Mean over draws")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_ownership_effects")


def fig_tradeoff():
    s = pd.read_csv(ANALYSIS / "method_summary.csv")
    methods = [("fedavg", "FedAvg", "o"), ("current", "Round exchange", "D"),
               ("fresh", "Step exchange", "s"), ("oracle", "Raw-feature oracle", "^"),
               ("fedgcn", "FedGCN adapter", "v")]
    colours = dict(zip([m for m, _, _ in methods], [GREY, GREEN, BLUE, PURPLE, VERMILION]))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, ds in zip(axes, ["elliptic", "ibm"]):
        d = s[s.dataset == ds]
        for m, lab, mk in methods:
            for st, fill in [("random", True), ("louvain", False)]:
                r = d[(d.method == m) & (d.strategy == st)].iloc[0]
                mib = (r.boundary_bytes + r.model_update_bytes + r.halo_feature_bytes) / 2**20
                ax.scatter(mib, r.ap, marker=mk, s=34, color=colours[m] if fill else "white",
                           edgecolor=colours[m], lw=1.0, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("Logical training payload per run (MiB, log scale)")
        ax.set_ylabel("Test AP (raw scores)")
        ax.set_title(DATASET_TITLE[ds])
        ax.grid(lw=0.3, alpha=0.4, which="major")
    handles = [Line2D([], [], marker=mk, color=colours[m], ls="", ms=5, label=lab)
               for m, lab, mk in methods]
    handles += [Line2D([], [], marker="o", color="black", ls="", ms=5, label="Random (filled)"),
                Line2D([], [], marker="o", mfc="white", mec="black", ls="", ms=5, label="Louvain (open)")]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.07),
               columnspacing=1.0, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig_tradeoff")


def fig_temporal():
    t = pd.read_csv(ANALYSIS / "temporal_metrics.csv")
    t = t[(t.dataset == "elliptic") & ((t.strategy == "random") | t.method.isin(POOLED))]
    series = [("fedavg", "FedAvg", GREY, "o"), ("fresh", "Step exchange", BLUE, "s"),
              ("fedmlp", "FedMLP", GREEN, "D"), ("rf_pooled", "Pooled RF", VERMILION, "^")]
    steps = sorted(t.timestep.unique())
    support = t.groupby("timestep").positive.first()
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    for m, lab, c, mk in series:
        d = t[t.method == m].groupby("timestep").ap
        mu, lo, hi = d.mean(), d.min(), d.max()
        ax.fill_between(steps, lo.loc[steps], hi.loc[steps], color=c, alpha=0.12, lw=0)
        ax.plot(steps, mu.loc[steps], marker=mk, ms=3.5, lw=1.0, color=c, label=lab)
    ax.axvspan(42.5, 47.5, color="0.93", zorder=0, lw=0)
    ax.text(45.0, 0.97, "$\\leq$24 illicit transactions per step", ha="center", va="top",
            fontsize=7, color="0.35")
    ax.set_xticks(steps, [f"{s}\n({support.loc[s]})" for s in steps])
    ax.set_xlabel("Test timestep (illicit transactions)")
    ax.set_ylabel("Per-timestep AP")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    save(fig, "fig_temporal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ANALYSIS)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    ANALYSIS, OUT = args.analysis, args.output
    style()
    g = grid()
    fig_performance(g)
    fig_effects(g)
    fig_tradeoff()
    fig_temporal()
    print("Wrote figures to", OUT)
