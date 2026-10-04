"""CausalDS benchmark experiments (resumable; every result appended to JSONL).

E1  estimation  : does the harness recover the true ATE with honest CIs, vs naive approaches?
E2  falsification: does testing implied independencies catch wrong causal claims
                   (hidden confounding, flipped edges) without rejecting correct ones?
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np

from .backend import RESULTS

from . import causalds as C
from .claim import Claim, Edge
from .estimate import aipw, difference, s_learner
from .falsify import falsify

OUT = RESULTS / "causal"
MAX_TESTS = 8  # first k implied independencies (topological order) per claim, Holm-corrected
CLAIMS = Path(__file__).resolve().parents[2] / "claims" / "causalds"


def _log(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")


def _done(path: Path, keys: tuple[str, ...]) -> set:
    if not path.exists():
        return set()
    return {tuple(json.loads(l)[k] for k in keys) for l in path.read_text(encoding="utf-8").splitlines() if l}


def claude_claim(scene: str) -> Claim | None:
    p = CLAIMS / scene / "claude.yaml"
    return Claim.load(p) if p.exists() else None


# ----------------------------------------------------------------------------- E1
def adjustment_arms(scene: str) -> dict[str, list[str] | None]:
    """Adjustment-set sources. None = the source says 'not identifiable by adjustment' (abstain)."""
    oracle = C.oracle_claim(scene)
    T, Y = oracle.treatment, oracle.outcome
    covs = [v for v in oracle.observed if v not in (T, Y)]
    pre = [v for v in covs if v not in nx.descendants(oracle.graph, T)]
    adj = oracle.adjustment_set()
    arms = {"all_features": covs,
            "pretreatment": pre,      # competent-practitioner heuristic (needs true time order)
            "oracle": sorted(adj) if adj is not None else None}
    cc = claude_claim(scene)
    if cc is not None:
        a = cc.adjustment_set()
        arms["claude_harness"] = sorted(a) if a is not None else None
    return arms


def run_e1(scenes: list[str], n: int, reps: int, learners: list[str], log=print,
           arms_filter: list[str] | None = None, out: str = "e1_estimation.jsonl"):
    path = OUT / out
    done = _done(path, ("scene", "rep", "arm", "learner", "n"))
    for scene in scenes:
        oracle, truth = C.oracle_claim(scene), C.true_ate(scene)
        T, Y = oracle.treatment, oracle.outcome
        arms = adjustment_arms(scene)
        if arms_filter:
            arms = {k: v for k, v in arms.items() if k in arms_filter or k == "all_features"}
        for rep, df in enumerate(C.replicates(C.load_data(scene), n, reps)):
            cache: dict = {}
            jobs = [("difference", "-"), ("s_learner_all", "tabpfn_fast")]
            jobs += [(arm, l) for arm in arms for l in learners if not arms_filter or arm in arms_filter]
            for arm, learner in jobs:
                if (scene, rep, arm, learner, n) in done:
                    continue
                t0 = time.perf_counter()
                if arm == "difference":
                    r = difference(df, T, Y)
                elif arm == "s_learner_all":
                    r = s_learner(df, T, Y, arms["all_features"], learner)
                elif arms[arm] is None:
                    r = {"estimator": "abstain", "ate": np.nan, "se": np.nan, "lo": np.nan, "hi": np.nan}
                else:
                    key = (tuple(arms[arm]), learner)
                    if key not in cache:
                        cache[key] = aipw(df, T, Y, arms[arm], learner, seed=rep)
                    r = cache[key]
                rec = {"scene": scene, "rep": rep, "n": n, "arm": arm, "learner": learner,
                       "truth": truth, **{k: r.get(k) for k in ("estimator", "ate", "se", "lo", "hi", "n_clipped")},
                       "adjust": arms.get(arm, []) if arm not in ("difference",) else [],
                       "covered": bool(r["lo"] <= truth <= r["hi"]) if np.isfinite(r.get("lo", np.nan)) else None,
                       "seconds": round(time.perf_counter() - t0, 2)}
                _log(path, rec)
            log(f"E1 {scene} rep {rep} done")


# ----------------------------------------------------------------------------- E2
def wrong_claims(scene: str) -> list[Claim]:
    """Consequential mistakes an analyst could make, derived from the true graph:
    - 'no hidden confounding': drop latent nodes (and their edges)
    - 'reversed edge at treatment': flip one edge touching the treatment (mediator<->confounder,
      collider<->confounder). Kept only if it changes the claimed adjustment set."""
    oracle = C.oracle_claim(scene)
    T = oracle.treatment
    base_adj = oracle.adjustment_set()
    out = []
    if oracle.latent:
        out.append(Claim(T, oracle.outcome, oracle.observed,
                         [e for e in oracle.edges if e.src not in oracle.latent and e.dst not in oracle.latent],
                         author="wrong: assumes no hidden confounding"))
    for e in oracle.edges:
        if T not in (e.src, e.dst) or oracle.outcome in (e.src, e.dst) or e.src in oracle.latent:
            continue
        flipped = [Edge(x.dst, x.src) if x is e else x for x in oracle.edges]
        c = Claim(T, oracle.outcome, oracle.observed, flipped, oracle.latent,
                  author=f"wrong: reversed {e.src} -> {e.dst}")
        try:
            adj = c.adjustment_set()
        except ValueError:  # cycle
            continue
        if adj != base_adj:
            out.append(c)
    return out


def detectable(claim: Claim, oracle: Claim, max_tests: int = 8) -> bool:
    """Can data falsify this claim at all? True if at least one independency the claim implies
    (among the tests actually run) is FALSE in the true graph. Claims that are Markov-equivalent to
    the truth on the tested set are undetectable by any CI test, however much data."""
    g = oracle.graph
    return any(not nx.is_d_separator(g, {a}, {b}, set(S)) for a, b, S in claim.implied_independencies()[:max_tests])


def run_e2(scenes: list[str], n: int, reps: int, methods: list[str], log=print):
    path = OUT / "e2_falsification.jsonl"
    done = _done(path, ("scene", "rep", "claim", "method", "n"))
    for scene in scenes:
        claims = [C.oracle_claim(scene)] + wrong_claims(scene)
        cc = claude_claim(scene)
        if cc is not None:
            cc.author = "claude (blind)"
            claims.append(cc)
        for rep, df in enumerate(C.replicates(C.load_data(scene), n, reps, seed=100)):
            cache: dict = {}  # claims with identical testable implications share one test run
            for claim in claims:
                for method in methods:
                    if (scene, rep, claim.author, method, n) in done:
                        continue
                    sig = (tuple(claim.implied_independencies()[:MAX_TESTS]), method)
                    if sig not in cache:
                        cache[sig] = falsify(claim, df, method=method, seed=rep, max_tests=MAX_TESTS)
                    res = cache[sig]
                    _log(path, {"scene": scene, "rep": rep, "n": n, "claim": claim.author, "method": method,
                                "claim_is_true": claim.author.startswith("oracle"),
                                "adjustment_set": sorted(claim.adjustment_set()) if claim.adjustment_set() is not None else None,
                                "testable": res["testable"], "rejected": res["rejected"],
                                "tests": [{"test": t.label, "p": t.p_value, "p_holm": t.p_adjusted, "s": t.seconds}
                                          for t in res["tests"]]})
            log(f"E2 {scene} rep {rep} done")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=["e1", "e2"])
    p.add_argument("--scenes", nargs="*")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--learners", nargs="+", default=["tabpfn_fast", "lgbm", "linear"])
    p.add_argument("--methods", nargs="+", default=["tabpfn_crt", "partial_corr"])
    p.add_argument("--arms", nargs="*", help="E1: restrict adjustment arms (e.g. oracle)")
    p.add_argument("--no-trap-only", action="store_true", help="E1: only scenes where all-features is valid")
    p.add_argument("--out", default=None, help="output jsonl name under results/causal/")
    a = p.parse_args(argv)
    b = C.benchmark_scenes()
    log = lambda m: print(m, flush=True)
    if a.experiment == "e1":
        scenes = a.scenes or b["identifiable"]
        if a.no_trap_only:
            from .analyze import trap_scenes
            scenes = [s for s in scenes if s not in trap_scenes()]
        run_e1(scenes, a.n, a.reps, a.learners, log, a.arms, a.out or "e1_estimation.jsonl")
    else:
        run_e2(a.scenes or b["identifiable"] + b["hidden_confounding"], a.n, a.reps, a.methods, log)


if __name__ == "__main__":
    main()
