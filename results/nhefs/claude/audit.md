# Causal audit: What is the average causal effect of quitting smoking on weight change?

**Verdict: NOT IDENTIFIED WITHOUT AN EXTRA ASSUMPTION — the claim names unmeasured confounding (incident_illness, state_of_residence). If it were absent, ATE = +3.496 (95% CI +2.531 to +4.461); to explain this away it would need a risk-ratio association of ≥ 2.36 with both treatment and outcome (≥ 2.01 to move the CI to include zero).**

- data: 1,566 rows · treatment `qsmk` · outcome `wt82_71`
- assumptions by: claude (blind) · nuisance learner: `tabpfn_fast`

## 1. Identification (from the stated causal graph)

- **not identifiable by adjustment** (the claim includes unmeasured confounding)
- do **not** adjust for `death`: post-treatment variable (descendant of treatment)
- do **not** adjust for `smkintensity82_71`: mediator (on a causal path from treatment to outcome)

## 2. Falsification (tabpfn_crt, 9 implied independencies, Holm-corrected)

| implied by the claim | p | p (Holm) | |
|---|---|---|---|
| sex ⟂ price82 | {} | 0.485 | 1.000 | ✓ consistent |
| price82 ⟂ age | {sex, race} | 0.693 | 1.000 | ✓ consistent |
| price82 ⟂ education | {sex, race} | 0.010 | 0.089 | ✓ consistent |
| price82 ⟂ smokeintensity | {sex, race, age, education, income} | 0.010 | 0.089 | ✓ consistent |
| price82 ⟂ smokeyrs | {sex, race, age, education, income} | 0.287 | 1.000 | ✓ consistent |
| price82 ⟂ exercise | {sex, race, age, education, income} | 0.069 | 0.416 | ✓ consistent |
| price82 ⟂ active | {sex, race, age, education, income} | 0.980 | 1.000 | ✓ consistent |
| price82 ⟂ wt71 | {sex, race, age, education, income} | 0.564 | 1.000 | ✓ consistent |
| price82 ⟂ alcoholfreq | {sex, race, age, education, income} | 0.010 | 0.089 | ✓ consistent |

## 3. Estimate (cross-fitted AIPW, TabPFN-3.5 nuisance models) — *conditional on no unmeasured confounding*

- adjusted for: active, age, alcoholfreq, education, exercise, income, price82, race, sex, smokeintensity, smokeyrs, wt71
- **ATE = +3.496** (95% CI +2.531 to +4.461)
- overlap: propensity range [0.062, 0.623], clipped 0
- E-value: 2.36 (CI bound 2.01) — the risk-ratio strength an unmeasured confounder would need to explain the effect away

## 4. What you would have concluded without the audit

| approach | ATE | 95% CI |
|---|---|---|
| naive difference | +2.541 | +1.585 to +3.496 |
| tabpfn all features s learner | +2.454 | — |
| aipw all features | +3.219 | +1.550 to +4.887 |
