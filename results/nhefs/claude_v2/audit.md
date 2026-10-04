# Causal audit: What is the average causal effect of quitting smoking on weight change?

**Verdict: PARTLY REJECTED, INCONSEQUENTIAL — 1 implied independency contradicted, but no local repair of them changes which variables must be adjusted for, so the estimate stands. NOT IDENTIFIED WITHOUT AN EXTRA ASSUMPTION — the claim names unmeasured confounding (incident_illness, state_of_residence). If it were absent, ATE = +3.496 (95% CI +2.531 to +4.461); to explain this away it would need a risk-ratio association of ≥ 2.36 with both treatment and outcome (≥ 2.01 to move the CI to include zero).**

- data: 1,566 rows · treatment `qsmk` · outcome `wt82_71`
- assumptions by: claude (revision after falsification) · nuisance learner: `tabpfn_fast`

## 1. Identification (from the stated causal graph)

- **not identifiable by adjustment** (the claim includes unmeasured confounding)
- do **not** adjust for `smkintensity82_71`: mediator (on a causal path from treatment to outcome)
- do **not** adjust for `death`: post-treatment variable (descendant of treatment)

## 2. Falsification (tabpfn_crt, 6 implied independencies, Holm-corrected)

| implied by the claim | p | p (Holm) | |
|---|---|---|---|
| sex ⟂ price82 given {} | 0.469 | 1.000 | ✓ consistent |
| price82 ⟂ age given {sex, race} | 0.680 | 1.000 | ✓ consistent |
| price82 ⟂ smokeyrs given {sex, race, age, education, income, smokeintensity} | 0.872 | 1.000 | ✓ consistent |
| price82 ⟂ exercise given {sex, race, age, education, income, smokeintensity} | 0.005 | 0.028 | ❌ contradicted — does not affect the adjustment set |
| price82 ⟂ active given {sex, race, age, education, income, smokeintensity} | 0.996 | 1.000 | ✓ consistent |
| price82 ⟂ wt71 given {sex, race, age, education, income, smokeintensity} | 0.538 | 1.000 | ✓ consistent |

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
