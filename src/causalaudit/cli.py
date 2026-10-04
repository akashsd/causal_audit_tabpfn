"""`uv run causalaudit <command>`"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

NHEFS_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/nhefs.csv"


def _load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def cmd_audit(a):
    from .audit import run, save
    from .claim import Claim

    claim = Claim.load(a.claim)
    df = _load_data(a.data)
    print(f"Auditing: {claim.question or claim.treatment + ' -> ' + claim.outcome}  ({len(df):,} rows)", flush=True)
    s = claim.summary()
    print(f"  adjustment set: {s['adjustment_set']}\n  forbidden: {list(s['forbidden'])}\n"
          f"  testable implications: {len(s['implied_independencies'])}", flush=True)
    r = run(claim, df, learner=a.learner, ci_method=a.ci_method, max_tests=a.max_tests, seed=a.seed)
    path = save(r, a.out)
    print(f"\n{r['verdict']}")
    if "estimate" in r:
        e = r["estimate"]
        print(f"ATE = {e['ate']:+.3f}  (95% CI {e['lo']:+.3f} to {e['hi']:+.3f})")
    print(f"Report: {path}")


def cmd_nhefs_data(a):
    """Download NHEFS (public URL) and keep the codebook columns, people with a 1982 weight."""
    cols = ["qsmk", "wt82_71", "sex", "race", "age", "education", "income", "smokeintensity", "smokeyrs",
            "exercise", "active", "wt71", "alcoholfreq", "smkintensity82_71", "price82", "death"]
    df = pd.read_csv(NHEFS_URL)
    df = df[df["wt82"].notna()][cols].reset_index(drop=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {df.shape}")


def cmd_bench(a):
    from . import bench

    bench.main([a.experiment, *a.rest])


def main(argv=None):
    p = argparse.ArgumentParser(prog="causalaudit")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("audit", help="audit one causal question: claim + data -> report")
    s.add_argument("--data", required=True)
    s.add_argument("--claim", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--learner", default="tabpfn_fast",
                   help="tabpfn_fast | tabpfn | tabpfn_plus_api | tabpfn_thinking_api | lgbm | linear")
    s.add_argument("--ci-method", default="tabpfn_crt", help="tabpfn_crt | tabpfn_crt_base | partial_corr")
    s.add_argument("--max-tests", type=int, default=12)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_audit)

    s = sub.add_parser("nhefs-data", help="download + prepare the NHEFS demo data")
    s.add_argument("--out", default="data/nhefs/nhefs.csv")
    s.set_defaults(fn=cmd_nhefs_data)

    s = sub.add_parser("bench", help="CausalDS experiments: bench e1|e2 [--n --reps ...]")
    s.add_argument("experiment", choices=["e1", "e2"])
    s.add_argument("rest", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_bench)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
