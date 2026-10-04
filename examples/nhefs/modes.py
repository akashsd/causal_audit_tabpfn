"""TabPFN-3.5 modes on the NHEFS audit: same claim, same adjustment set, same cross-fitting folds —
only the nuisance model changes (local Fast / local base / API Plus / API Thinking).

    uv run python examples/nhefs/modes.py --learners tabpfn_fast tabpfn tabpfn_plus_api tabpfn_thinking_api
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from causalaudit.backend import ROOT
from causalaudit.claim import Claim
from causalaudit.estimate import aipw

OUT = ROOT / "results" / "nhefs" / "modes.jsonl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--learners", nargs="+", default=["tabpfn_fast", "tabpfn"])
    a = p.parse_args()
    claim = Claim.load(ROOT / "claims" / "nhefs" / "claude.yaml")
    # Estimate under the claim *without* its hypothesised latent confounders (the audit's conditional analysis).
    no_latent = Claim(claim.treatment, claim.outcome, claim.observed,
                      [e for e in claim.edges if e.src not in claim.latent and e.dst not in claim.latent],
                      background=claim.background)
    adj = sorted(no_latent.adjustment_set())
    df = pd.read_csv(ROOT / "data" / "nhefs" / "nhefs.csv")
    for learner in a.learners:
        t0 = time.perf_counter()
        r = aipw(df, claim.treatment, claim.outcome, adj, learner, seed=0)
        r["seconds"] = round(time.perf_counter() - t0, 1)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, default=float) + "\n")
        print(f"{learner:22} ATE {r['ate']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]  "
              f"propensity [{r['propensity_min']:.3f}, {r['propensity_max']:.3f}]  {r['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
