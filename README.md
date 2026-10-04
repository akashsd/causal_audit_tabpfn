# causalaudit — TabPFN-3.5 audits causal claims

> **An agent proposes the causal story. TabPFN-3.5 tests every assumption against the data,
> and estimates the effect only from the assumptions that survive.**

Ask a predictive model "does X cause Y?" and it will answer confidently — even when the data
cannot tell. Hand every column to the model and it silently adjusts for mediators, colliders and
post-treatment variables. `causalaudit` makes the causal assumptions explicit and then uses
TabPFN-3.5 twice:

1. **Falsify** — every causal graph implies conditional independencies. `causalaudit` tests each
   one with a **TabPFN-3.5 conditional randomization test** (TabPFN-CRT, from `tabpfn-extensions`):
   TabPFN models the target *and* simulates the tested variable from its full predictive
   distribution — nonlinear, mixed-type, no tuning.
2. **Estimate** — a cross-fitted doubly-robust (AIPW) estimator whose propensity and outcome
   models are TabPFN-3.5 (local open weights, or Plus / Thinking via the Prior Labs API).

Claude (the agent) writes the assumptions as a versioned YAML graph *before seeing the data*;
everything after that is deterministic and reproducible without Claude.

```
claim.yaml ──► identification ──► falsification ──► overlap ──► AIPW estimate ──► verdict
 (Claude)      adjust / forbid     TabPFN-CRT        TabPFN       TabPFN nuisances   + E-value
```

> 🚧 Work in progress for the Prior Labs TabPFN-3.5 Hackathon — results tables are being finalized.

## Quick start

```bash
uv sync                                   # Python 3.11–3.13; CUDA PyTorch on Windows/Linux
cp .env.example .env                      # add TABPFN_API_KEY
# one-time: accept the TabPFN-3.5 license at https://ux.priorlabs.ai (Licenses tab) for local weights
uv run causalaudit nhefs-data             # (data is already in data/nhefs/; this re-downloads it)
uv run causalaudit audit --data data/nhefs/nhefs.csv --claim claims/nhefs/claude.yaml --out results/nhefs/claude
```

## Real-world demo: does quitting smoking cause weight gain? (NHEFS)

A blind Claude agent read only the [data dictionary](examples/nhefs/codebook.md) and wrote
[its causal claim](claims/nhefs/claude.yaml). The [audit](results/nhefs/claude/audit.md):

| approach | effect of quitting on weight change (kg) |
|---|---|
| naive difference | +2.54 [1.59, 3.50] |
| "hand every column to TabPFN" (S-learner) | +2.45 |
| **causalaudit** (AIPW, TabPFN-3.5, Claude's adjustment set) | **+3.50 [2.53, 4.46]** |
| textbook (Hernán & Robins, IP weighting / standardization) | +3.4 to +3.5 |

The agent flagged `smkintensity82_71` (a mediator) and `death` (post-treatment) as forbidden
controls, and named an unmeasured "sick-quitter" confounder — so the verdict reports the effect
*conditional on* its absence, with an E-value of 2.4 for how strong it would need to be.

## Benchmark: CausalDS (third-party ground truth)

[CausalDS](https://github.com/andleb/causalds) (arXiv 2607.08093) scenes are synthetic causal
systems with realistic variable names, stories, and known true effects. We use 33 clean,
binary-treatment scenes (22 identifiable incl. mediator/collider traps, 11 with hidden
confounding). Claude's claims were written blind from the stories and frozen
([hash manifest](claims/causalds/MANIFEST.sha256)) before any estimate was computed.

```bash
uv run causalaudit bench e1 --n 1000 --reps 10   # estimation: bias, RMSE, CI coverage
uv run causalaudit bench e2 --n 2000 --reps 3    # falsification: rejection of wrong vs true claims
uv run python -m causalaudit.analyze             # -> results/causal/summary.md
```

## Repository layout

```
src/causalaudit/   claim.py (graph logic) · falsify.py (TabPFN-CRT) · estimate.py (AIPW)
                   audit.py (pipeline + report) · causalds.py / bench.py / analyze.py (benchmark)
claims/            blind Claude claims (CausalDS scenes, NHEFS) + hash manifest
prompts/           the exact elicitation prompts given to the blind agent
examples/nhefs/    data dictionary given to the agent
data/nhefs/        NHEFS analysis file (public: Rdatasets/causaldata)
results/           benchmark logs and audit reports
tests/             unit tests (graph logic, estimators, cache safety)
```

## License

Apache-2.0. CausalDS data: CC0; NHEFS: public (Hernán & Robins, *Causal Inference: What If*).
