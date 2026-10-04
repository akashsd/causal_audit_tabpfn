# NHEFS — data dictionary (subset used in this demo)

Source: NHANES I Epidemiologic Follow-up Study (NHEFS), as distributed with Hernán & Robins,
*Causal Inference: What If*. Public CSV: https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/nhefs.csv

Study design: adult cigarette smokers had a baseline medical examination in **1971–75**
and a follow-up interview/exam in **1982**. Mortality was tracked until **1992**.
One row = one person. Only people with a measured 1982 weight are analysed.

**Question:** What is the average causal effect of quitting smoking on weight change?

| column | meaning | measured |
|---|---|---|
| `qsmk` | **treatment** — quit smoking between baseline and 1982 (1 = yes, 0 = no) | 1971→1982 |
| `wt82_71` | **outcome** — weight change in kg (1982 weight minus baseline weight) | 1982 |
| `sex` | 0 = male, 1 = female | baseline |
| `race` | 0 = white, 1 = black or other | baseline |
| `age` | age in years | baseline |
| `education` | 1 = 8th grade or less, 2 = high school dropout, 3 = high school, 4 = college dropout, 5 = college or more | baseline |
| `income` | total family income, coded 11 (< $1,000) to 22 (≥ $25,000) | baseline |
| `smokeintensity` | cigarettes smoked per day | baseline |
| `smokeyrs` | years of smoking | baseline |
| `exercise` | recreational exercise: 0 = much, 1 = moderate, 2 = little or none | baseline |
| `active` | activity in daily life: 0 = very active, 1 = moderately active, 2 = inactive, 3 = missing | baseline |
| `wt71` | weight in kg | baseline |
| `alcoholfreq` | how often alcohol is consumed: 0 = almost every day, 1 = 2–3 times/week, 2 = 1–4 times/month, 3 = < 12 times/year, 4 = no alcohol last year, 5 = unknown | baseline |
| `smkintensity82_71` | change in cigarettes per day between baseline and 1982 | 1982 |
| `price82` | average price of tobacco in the person's state of residence, 1982 (USD) | 1982 |
| `death` | died by 1992 (1 = yes) | 1992 |
