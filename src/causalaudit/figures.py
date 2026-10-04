"""README figures from the benchmark logs: `uv run --group figures python -m causalaudit.figures`."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from . import causalds as C
from .analyze import cluster_bootstrap, trap_scenes, wilson
from .bench import OUT

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a9a8a2"
LEARNER = {"tabpfn_fast": ("TabPFN-3.5", BLUE), "lgbm": ("LightGBM", ORANGE), "linear": ("Linear", AQUA)}

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})


def _read(name):
    p = OUT / name
    return pd.DataFrame([json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l])


def _e1():
    d = _read("e1_estimation.jsonl")
    sd = {s: C.ground_truth(s)["outcome_stats"]["std"] for s in d.scene.unique()}
    d["err_sd"] = (d.ate - d.truth) / d.scene.map(sd)
    d["trap"] = d.scene.isin(trap_scenes())
    d["flagged"] = d.n_clipped.fillna(0) > 0.02 * d.n
    return d


def _hbars(ax, labels, values, colors, lo=None, hi=None, fmt="{:.0%}", xmax=1.0):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.55, color=colors, edgecolor=SURFACE, linewidth=2)
    if lo is not None:
        ax.errorbar(values, y, xerr=[np.array(values) - lo, np.array(hi) - values], fmt="none",
                    ecolor=INK2, elinewidth=1.2, capsize=3)
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.02 + ((hi[list(y).index(yi)] - v) if hi is not None else 0), yi, fmt.format(v),
                va="center", color=INK, fontsize=9)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, xmax * 1.18)
    if fmt.endswith("%}"):
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)


def fig_traps():
    d = _e1()
    t = d[d.trap]
    arms = [("claude_harness", "tabpfn_fast", "causalaudit\n(Claude's claim)", BLUE),
            ("pretreatment", "tabpfn_fast", "pre-treatment rule\n(needs true time order)", GRAY),
            ("all_features", "tabpfn_fast", "TabPFN, all features\n(AIPW)", GRAY),
            ("s_learner_all", "tabpfn_fast", "TabPFN, all features\n(S-learner)", GRAY)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"wspace": 0.75})
    cov = [(lab, col, t[(t.arm == a) & (t.learner == l)].covered.dropna()) for a, l, lab, col in arms[:3]]
    vals = [c.mean() for _, _, c in cov]
    ci = [wilson(int(c.sum()), len(c)) for _, _, c in cov]
    _hbars(a1, [x[0] for x in cov], vals, [x[1] for x in cov], np.array([c[0] for c in ci]), np.array([c[1] for c in ci]))
    a1.axvline(0.95, color=INK2, lw=1, ls="--")
    a1.set_ylim(-0.75, None)
    a1.text(0.94, -0.62, "nominal 95%", color=INK2, fontsize=8, ha="right")
    a1.set_xlabel("95% CI contains the true effect")
    rm = [(lab, col, np.sqrt((t[(t.arm == a) & (t.learner == l)].err_sd ** 2).mean())) for a, l, lab, col in arms]
    _hbars(a2, [x[0] for x in rm], [x[2] for x in rm], [x[1] for x in rm], fmt="{:.2f}", xmax=max(x[2] for x in rm))
    a2.set_xlabel("RMSE of the effect (outcome SDs)")
    n_sc = t.scene.nunique()
    fig.suptitle(f"Trap scenes (mediators, colliders, post-treatment variables) — {n_sc} CausalDS scenes × 10 replicates",
                 x=0.02, ha="left", fontsize=11, color=INK)
    fig.savefig(OUT / "fig_traps.png", dpi=160, bbox_inches="tight")


def fig_learners():
    d = _e1()
    t = d[~d.trap & (d.arm == "claude_harness")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 2.8), gridspec_kw={"wspace": 0.45})
    ls = list(LEARNER)
    rm = [np.sqrt((t[t.learner == l].err_sd ** 2).mean()) for l in ls]
    _hbars(a1, [LEARNER[l][0] for l in ls], rm, [LEARNER[l][1] for l in ls], fmt="{:.2f}", xmax=max(rm))
    a1.set_xlabel("RMSE of the effect (outcome SDs)")
    fl = [t[t.learner == l].flagged.mean() for l in ls]
    _hbars(a2, [LEARNER[l][0] for l in ls], fl, [LEARNER[l][1] for l in ls])
    a2.set_xlabel("audits flagged FRAGILE (propensities clipped)")
    fig.suptitle(f"Same estimator (AIPW), same adjustment set — only the nuisance model changes · "
                 f"{t.scene.nunique()} confounded scenes × 10 replicates, n = 1000",
                 x=0.02, ha="left", fontsize=11, color=INK)
    fig.savefig(OUT / "fig_learners.png", dpi=160, bbox_inches="tight")


def fig_small_n():
    p = OUT / "e1_small_n.jsonl"
    if not p.exists():
        return
    d = pd.concat([_read("e1_small_n.jsonl"), _read("e1_estimation.jsonl")], ignore_index=True)
    d = d.drop_duplicates(["scene", "rep", "arm", "learner", "n"], keep="last")
    d = d[(d.arm == "oracle") & d.learner.isin(LEARNER) & ~d.scene.isin(trap_scenes())]
    sd = {s: C.ground_truth(s)["outcome_stats"]["std"] for s in d.scene.unique()}
    d["err_sd"] = (d.ate - d.truth) / d.scene.map(sd)
    g = d.groupby(["learner", "n"]).err_sd.apply(lambda e: np.sqrt((e ** 2).mean())).unstack("learner")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ends = sorted(((g[l].iloc[-1], l) for l in LEARNER if l in g), reverse=True)
    label_y, prev = {}, None
    for v, l in ends:  # push end labels apart when lines finish close together
        y = v if prev is None else min(v, prev - 0.045)
        label_y[l], prev = y, y
    for l, (lab, col) in LEARNER.items():
        if l in g:
            ax.plot(g.index, g[l], color=col, lw=2, marker="o", ms=7, mec=SURFACE, mew=2)
            ax.text(g.index[-1] * 1.06, label_y[l], lab, color=INK, va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(g.index, [str(i) for i in g.index])
    ax.minorticks_off()
    ax.set_xlim(g.index.min() * 0.85, g.index.max() * 1.6)
    ax.set_ylim(0, None)
    ax.set_xlabel("rows per analysis")
    ax.set_ylabel("RMSE of the effect (outcome SDs)")
    ax.set_title("Smaller data, bigger TabPFN advantage (AIPW, correct adjustment set)", loc="left", fontsize=11)
    fig.savefig(OUT / "fig_small_n.png", dpi=160, bbox_inches="tight")


def fig_falsification(n: int = 2000):
    from .analyze import claim_kinds

    p = OUT / "e2_falsification.jsonl"
    if not p.exists():
        return
    d = _read("e2_falsification.jsonl")
    d = d[d.n == n].copy()
    d["kind"] = claim_kinds(d)
    n_equiv = d[d.kind.str.contains("equivalent")].groupby(["scene", "claim"]).ngroups
    d = d[d.testable]
    kinds = [("true claim", "true claims\n(false alarms ↓)"),
             ("wrong: ignores hidden confounding (detectable)", "wrong: ignores hidden\nconfounding (caught ↑)"),
             ("wrong: reversed edge at treatment (detectable)", "wrong: reversed edge\nat treatment (caught ↑)")]
    methods = [("tabpfn_crt", "TabPFN-3.5 CRT", BLUE), ("partial_corr", "linear partial correlation", ORANGE)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 3.8), gridspec_kw={"width_ratios": [3, 2], "wspace": 0.25})
    x = np.arange(len(kinds))
    w = 0.36
    for i, (m, lab, col) in enumerate(methods):
        vals, los, his = [], [], []
        for k, _ in kinds:
            s_ = d[(d.kind == k) & (d.method == m)]
            vals.append(s_.rejected.mean() if len(s_) else np.nan)
            lo, hi = cluster_bootstrap(s_.rejected.astype(float), s_.scene) if len(s_) else (np.nan, np.nan)
            los.append(lo); his.append(hi)
        xs = x + (i - 0.5) * w
        ax.bar(xs, vals, width=w, color=col, edgecolor=SURFACE, linewidth=2, label=lab)
        ax.errorbar(xs, vals, yerr=[np.array(vals) - los, np.array(his) - vals], fmt="none",
                    ecolor=INK2, elinewidth=1.2, capsize=3)
        for xi, v, h in zip(xs, vals, his):
            if np.isfinite(v):
                ax.text(xi + w * 0.08, min(max(v, h) + 0.04, 1.08), f"{v:.0%}", ha="center", fontsize=9, color=INK)
    ax.axhline(0.05, color=INK2, lw=1, ls="--")
    ax.set_xticks(x, [k[1] for k in kinds])
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("claims rejected")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title(f"A · Rejection rates at α = 0.05 (Holm; n = {n}; scene-clustered 95% CIs)", loc="left", fontsize=11)
    # Panel B: detection at an equal false-alarm rate (threshold set so <= 5% of TRUE claims are rejected)
    from .analyze import matched_power
    mp = matched_power()
    mp = mp[mp.n == n]
    kinds2 = kinds[1:]
    x2 = np.arange(len(kinds2))
    for i, (m, lab, col) in enumerate(methods):
        vals = [mp[(mp.method == m) & (mp.claim == k)]["detection at matched false-alarm rate"].mean() for k, _ in kinds2]
        xs = x2 + (i - 0.5) * w
        ax2.bar(xs, vals, width=w, color=col, edgecolor=SURFACE, linewidth=2)
        for xi, v in zip(xs, vals):
            ax2.text(xi, v + 0.04, f"{v:.0%}", ha="center", fontsize=9, color=INK)
    ax2.set_xticks(x2, [k[1].replace(" (caught ↑)", "") for k in kinds2])
    ax2.set_ylim(0, 1.18)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.grid(axis="x", visible=False)
    ax2.set_title("B · Wrong claims caught at an equal false-alarm rate (≤ 5%)", loc="left", fontsize=11)
    fig.text(0.01, -0.04, f"Not shown: {n_equiv} wrong claims are Markov-equivalent to the truth on the tested "
             "independencies — no test can detect them; the audit reports such assumptions as untestable.",
             fontsize=8, color=INK2)
    fig.savefig(OUT / "fig_falsification.png", dpi=160, bbox_inches="tight")


def main():
    fig_traps()
    fig_learners()
    fig_small_n()
    fig_falsification()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
