# CausalAudit: TabPFN-3.5 tests causal claims

An agent writes a causal claim. TabPFN-3.5 tests the claim against the data.
The tool estimates an effect only from a claim that the data does not contradict.

## The problem

A predictive model does not tell you what causes what.
If you give a model every column, it can adjust for the wrong variables.
These variables include mediators, colliders, and variables that occur after the treatment.
The result is an effect estimate that is wrong but looks precise.

An LLM agent can write a good causal story.
But nobody checks the assumptions in that story.

## What CausalAudit does

1. **Claim.** Claude writes a causal graph as a YAML file before it sees the data.
2. **Identify.** The code finds which variables to adjust for, and which variables you must not adjust for.
3. **Test.** The code tests each conditional independence that the graph implies.
   The test is the TabPFN-3.5 conditional randomization test (TabPFN-CRT) from `tabpfn-extensions`.
4. **Check consequences.** If a test fails, the code checks if the failure changes the adjustment set.
5. **Estimate.** The code estimates the effect with a cross-fitted AIPW estimator.
   TabPFN-3.5 supplies the propensity model and the outcome models.
6. **Report.** The report gives a verdict, the estimate, an E-value, and the results of simpler methods.

Claude only writes the claim. Deterministic code makes all numbers.
You can run the full pipeline again without Claude.

```
claim.yaml ─► identify ─► test (TabPFN-CRT) ─► check consequences ─► estimate (AIPW + TabPFN-3.5) ─► report
```

## Why TabPFN-3.5

TabPFN-3.5 gives a full predictive distribution for each row. It needs no tuning.
These two properties do two jobs in CausalAudit:

- **Testing.** The CRT must sample a variable from its conditional distribution. TabPFN-3.5 does this directly.
- **Estimation.** AIPW needs good propensity and outcome models on small data. TabPFN-3.5 gives them without a hyperparameter search.

## Results

All numbers come from the logs in `results/`. The file `results/causal/summary.md` contains the full tables.
CausalDS is a third-party benchmark with known true effects.

| Result | Evidence |
|---|---|
| **TabPFN-CRT does not reject correct claims** | On correct causal graphs, TabPFN-CRT rejects 4.8% of claims (5 of 104 runs). A nonparametric test without TabPFN (GCM with LightGBM) rejects 18%. A linear test rejects 29%. The nominal rate is 5%. |
| **TabPFN-CRT finds wrong claims** | TabPFN-CRT rejects 89% of claims with a reversed edge at the treatment. It rejects 51% of claims that ignore a hidden confounder at 2000 rows, and 64% at 4000 rows. GCM finds about the same number, but only if you know the true graph to set its threshold. |
| **The audit stops a common error** | In 9 trap scenes, the audit's 95% CI contains the true effect in 92% of runs. AIPW that adjusts for all columns: 26%. |
| **TabPFN-3.5 needs no tuning** | With the same estimator and adjustment set, TabPFN-3.5-Fast has the same or lower error than a tuned LightGBM (RMSE 0.41 vs 0.43 at 300 rows; 0.20 vs 0.21 at 1000 rows). The full TabPFN-3.5 model gives the same error as the Fast model. |
| **The blind claims are correct** | Claude wrote 33 claims from the scene stories only. All 33 adjustment decisions are correct. |
| **Real data (NHEFS)** | Effect of quitting smoking on weight change: **+3.50 kg [2.53, 4.46]**. The textbook value is +3.4 to +3.5 kg. The audit also found a wrong part of Claude's claim. The audit showed that this error does not change the estimate. |

## Terms

| Term | Meaning |
|---|---|
| Claim | A causal graph in YAML: the treatment, the outcome, the observed variables, the hidden (latent) variables, and the direct causal edges. |
| Adjustment set | The variables that the estimator must control for, from the backdoor criterion. |
| Forbidden control | A variable that makes the estimate wrong if you control for it (a mediator, a collider, or a variable after the treatment). |
| Implied independence | "A is independent of B given S." The claim makes this statement. The data can contradict it. |
| TabPFN-CRT | A conditional randomization test. TabPFN-3.5 models the target. TabPFN-3.5 also samples the tested variable from its conditional distribution. |
| Holm correction | A correction for many tests on one claim. The audit rejects a claim if one corrected p-value is below 0.05. |
| Detectable wrong claim | A wrong claim with at least one tested implication that is false. Some wrong claims are *equivalent* to the true graph: no test can find them. We report these separately. |
| Matched false-alarm rate | Each test gets its own threshold. With this threshold, each test rejects at most 5% of correct claims. Then we compare how many wrong claims each test finds. |
| AIPW | Augmented inverse probability weighting. A doubly-robust estimator of the average treatment effect (ATE). |
| Propensity | The probability of treatment for a unit, given the adjustment set. |
| FRAGILE | The verdict when more than 2% of propensities are below 0.01 or above 0.99. The estimate then depends on extrapolation. |
| SD units | Errors divided by the standard deviation of the outcome. This makes scenes comparable. |
| E-value | How strong (as a risk ratio) a hidden confounder must be to remove the effect. For a continuous outcome we use an approximation. |
| Scene | One CausalDS benchmark case: a story, data, and a hidden true causal model. |

## Quick start

1. Install the dependencies:
   ```bash
   uv sync
   ```
   This uses CUDA PyTorch on Windows and Linux. macOS uses the default wheels.
2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
3. Add your Prior Labs API key to `.env` as `TABPFN_API_KEY`.
4. Accept the TabPFN-3.5 license one time at <https://ux.priorlabs.ai> (Licenses tab). The local model weights need this.
5. Run the audit on the NHEFS example:
   ```bash
   uv run CausalAudit audit --data data/nhefs/nhefs.csv --claim claims/nhefs/claude.yaml --out results/nhefs/claude
   ```
6. Read the report in `results/nhefs/claude/audit.md`.
7. Run the unit tests:
   ```bash
   uv run pytest
   ```

To use the agent loop in Claude Code, type `/audit <data.csv> <codebook.md> <name>`.
The command does these steps:

1. A new subagent reads only the data dictionary and writes the claim.
2. The command records the SHA-256 hash of the claim.
3. The command runs the audit.
4. If the data rejects the claim, the agent writes a new claim file. The agent does not change the old file.

## Example: does quitting smoking cause weight gain? (NHEFS)

A Claude subagent read only the [data dictionary](examples/nhefs/codebook.md).
The dictionary gives the meaning of each column and the time of each measurement.
The subagent then wrote [the claim](claims/nhefs/claude.yaml).

**What the audit found:**

- **Forbidden controls.** `smkintensity82_71` is a mediator. `death` occurs after the treatment.
- **Hidden confounders.** The claim names two latent variables: *incident illness* and *state of residence*.
  Illness can cause people to quit smoking and to lose weight.
  Thus the effect is not identified under the claim.
  The audit gives the estimate *if* these confounders are absent.
- **Sensitivity.** The approximate E-value is 2.4 (2.0 for the CI bound).
- **A contradicted claim.** The claim says that the 1982 cigarette price (`price82`) relates to other covariates only through race and income.
  TabPFN-CRT rejected 3 of 9 implied independencies: education, smoking intensity, and alcohol use (Holm p < 0.001).
- **A revised claim.** A new subagent read only the audit report and wrote [`claude_v2.yaml`](claims/nhefs/claude_v2.yaml).
  The revision uses the test results, so it is not blind.
  The audit of v2 rejected one more implication (exercise).
- **The consequence check.** For each failed test, the code adds a local repair to the claim.
  A repair is an edge in each direction, or a hidden common cause.
  No repair changes the adjustment set. The verdict is *PARTLY REJECTED, INCONSEQUENTIAL*.
  The estimate is the same for v1 and v2. We did not make more revisions.

| Method | Effect of quitting on weight change (kg) |
|---|---|
| Difference in means | +2.54 [1.59, 3.50] |
| TabPFN S-learner with all columns (includes the mediator and `death`) | +2.45 |
| AIPW with all columns | +3.22 [1.55, 4.89] |
| **CausalAudit**, TabPFN-3.5-Fast (local) | **+3.50 [2.53, 4.46]** |
| CausalAudit, TabPFN-3.5 (local) | +3.50 [2.54, 4.46] |
| CausalAudit, TabPFN-3.5 API (`v3.5`, learner `tabpfn_plus_api`) | +3.50 [2.54, 4.46] |
| CausalAudit, TabPFN-3.5 API with Thinking mode | +3.56 [2.51, 4.61] |
| Textbook (Hernán & Robins: IP weighting / standardization) | +3.4 [2.4, 4.5] / +3.5 [2.6, 4.5] |

The local model and the API agree within 0.0002 kg.
The textbook value is a sanity check, not a ground truth. It uses the same no-hidden-confounding assumption.
Compared to the audit, the S-learner with all columns is 30% lower.

## Benchmark: CausalDS

[CausalDS](https://github.com/andleb/causalds) (arXiv 2607.08093) gives synthetic causal scenes.
Each scene has realistic variable names, a story, data, and the true effect and graph.
We use all 33 clean scenes with a binary treatment:

- 22 scenes where the effect is identifiable. 9 of these are trap scenes: adjustment for all columns is wrong.
- 11 scenes with a hidden confounder. The correct answer is "not identifiable by adjustment".

**Blind claims.** For each scene, a new Claude subagent read only the story and the column names.
The prompt is in [`prompts/elicit_claim.md`](prompts/elicit_claim.md).
The [manifest](claims/causalds/MANIFEST.sha256) records the hash of each claim. Its timestamp is self-reported.
All 33 adjustment decisions are correct.
The stories are clear, so the claims are almost the true graphs.
Thus E1 measures the pipeline, and E2 measures what happens when a claim is wrong.

### E1: estimation

![trap scenes](results/causal/fig_traps.png)

- In the 9 trap scenes, the true adjustment set is empty. The difference in means is the correct method.
  The audit finds this. Its 95% CI contains the true effect in 92% of runs [87–97%].
- AIPW that adjusts for all columns: 26% [4–51%]. The TabPFN S-learner with all columns has 5× the error.
- A rule "adjust for all variables before the treatment" also gets 93%. This rule needs the true time order. Claude found the time order from the story.

![learners](results/causal/fig_learners.png)
![small n](results/causal/fig_small_n.png)

These two figures use the 13 confounded scenes. The estimator and the adjustment set do not change. Only the model changes.

- TabPFN-3.5-Fast has the same or lower error than LightGBM with a 3-fold random search. TabPFN needs no search.
- The default LightGBM and the linear models have twice the error.
- The full TabPFN-3.5 model has almost the same error as TabPFN-3.5-Fast. The RMSE is 0.41 vs 0.41 at 300 rows, 0.26 vs 0.27 at 500 rows, and 0.20 vs 0.20 at 1000 rows. The Fast model is sufficient for this task.
- We do not report run times. The runs shared one GPU with other experiments, so the times are not reliable.
- 17% of TabPFN-Fast estimates are FRAGILE at 1000 rows (30% at 300 rows). The FRAGILE estimates cover the truth in only about 50% of runs. The flag marks the estimates that you must not trust.
- Without the FRAGILE estimates, the TabPFN coverage is 96–97% at 300, 500, and 1000 rows.
- Note: with no traps, the TabPFN S-learner has a low error (RMSE 0.17 SD). But it gives no confidence interval, and it fails in trap scenes.

### E2: falsification

![falsification](results/causal/fig_falsification.png)

We make wrong claims from each true graph:

- **Ignore the hidden confounder.**
- **Reverse an edge at the treatment.** This changes a mediator or collider into a confounder, or the opposite.

We test each claim in 8 independent samples of 2000 rows.

- On correct claims, TabPFN-CRT rejects 4.8% [2–9%]. GCM with LightGBM rejects 18% [10–28%]. The linear test rejects 29% [13–48%]. The nominal rate is 5%.
- On equivalent wrong claims (no test can find them), TabPFN-CRT rejects 7.5%. This is close to its false-alarm rate, as it must be. GCM rejects 30% and the linear test 35%.
- At the nominal 5% level, TabPFN-CRT finds 89% of the reversed edges and 51% of the ignored hidden confounders. GCM finds 91% and 59%, and the linear test 88% and 62%. But these two tests also reject many correct claims.
- Panel B gives each test a threshold with the same false-alarm rate (4.8%). TabPFN-CRT and GCM then find about the same number of wrong claims (hidden confounder: 52% and 51%; reversed edge: 89% and 86%). The linear test finds only 7% of the hidden-confounder claims.
- A matched threshold needs the true graph. A real audit does not have it. Thus the important property is a correct false-alarm rate at the nominal level. Only TabPFN-CRT has this property here.
- The power grows with the data. At 4000 rows, TabPFN-CRT finds 64% of the hidden-confounder claims (11 scenes, 3 samples each).

## Limits

- A hidden confounder is testable only through its implications, for example an instrument that becomes dependent on the outcome. The power at 2000 rows is moderate and changes from scene to scene.
- The audit does not verify an assumption that no test can reach. The E-value describes that risk.
- The estimator supports binary treatments only.
- The CausalDS true effects are Monte Carlo estimates.
- The CRT p-value uses a normal approximation to a null distribution with B = 100 samples. The plain rank p-value has a minimum of 1/101. After the Holm correction, the audit could then never reject a claim with 6 or more tests. The false-alarm rates above check the approximation.
- In the benchmark, each claim tests its first 8 implications. The CLI default is 12.
- The benchmarks use TabPFN-3.5-Fast unless a table says otherwise.
- The replicates are disjoint samples from 16,000 rows per scene. We cluster the confidence intervals by scene.
- `tabpfn-extensions` 0.6.3 needs a small dtype fix for the TabPFN-3.5 regressor. `falsify.py` applies the fix at run time.

## Reproduce the results

Run these commands. The times are for one RTX 3070.

```bash
uv run CausalAudit bench e1 --n 1000 --reps 10                                        # E1, about 2 h
uv run CausalAudit bench e1 --n 1000 --reps 10 --arms oracle --no-trap-only --learners lgbm_cv tabpfn
uv run CausalAudit bench e1 --n 300 --reps 20 --arms oracle --no-trap-only --learners tabpfn_fast lgbm lgbm_cv linear tabpfn --out e1_small_n.jsonl
uv run CausalAudit bench e1 --n 500 --reps 20 --arms oracle --no-trap-only --learners tabpfn_fast lgbm lgbm_cv linear tabpfn --out e1_small_n.jsonl
uv run CausalAudit bench e2 --n 2000 --reps 8 --methods tabpfn_crt gcm_lgbm partial_corr   # E2, about 6 h
uv run python -m CausalAudit.analyze                                                  # writes results/causal/summary.md
uv run --group figures python -m CausalAudit.figures                                  # writes the figures
uv run python examples/nhefs/modes.py --learners tabpfn_fast tabpfn tabpfn_plus_api tabpfn_thinking_api
```

The code downloads the CausalDS files from Hugging Face when it needs them.
The API learners use Prior Labs credits. The code caches each API prediction in `results/cache/`.
The command `uv run CausalAudit nhefs-data` downloads the NHEFS file again from its public URL.

## Repository layout

```
src/CausalAudit/   claim.py (graph logic), falsify.py (TabPFN-CRT, GCM, linear test), estimate.py (AIPW)
                   audit.py (pipeline and report), causalds.py, bench.py, analyze.py, figures.py
claims/            the blind claims and the hash manifests
prompts/           the exact prompts for the subagents
examples/nhefs/    the data dictionary and the modes script
data/nhefs/        the NHEFS analysis file
results/           the benchmark logs, the figures, and the audit reports
tests/             the unit tests
.claude/commands/  the /audit agent command
CLAUDE.md          the rules for the agent
```

## Credits and license

License: Apache-2.0.

- [TabPFN](https://github.com/PriorLabs/TabPFN) and [tabpfn-extensions](https://github.com/PriorLabs/tabpfn-extensions) (TabPFN-CRT) by Prior Labs.
- Benchmark: [CausalDS](https://github.com/andleb/causalds). The data license is CC0.
- NHEFS: Hernán and Robins, *Causal Inference: What If*. Public data from [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/).
- Related work: Causal-Copilot, CausalGuard, and PyWhy-LLM (LLM causal agents); Do-PFN and CausalPFN (causal foundation models); the TabPFN example in DoubleML; GCM (Shah and Peters, 2020).
