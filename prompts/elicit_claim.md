# Prompt: blind causal-claim elicitation (versioned; used verbatim for every scene)

You are a domain expert writing down your causal assumptions *before* seeing any data.

For each scene folder you are given, read ONLY:
- `story.md` — a description of the domain and how the variables relate
- `columns.json` — the measured columns, plus the treatment and outcome of interest

Do NOT open the data file, any ground-truth/private files, any results, or the harness source code.
Your claim is committed before any estimation runs; it must reflect only the story and your knowledge.

Write `claims/causalds/<scene>/claude.yaml` in exactly this format:

```yaml
question: Does <treatment> cause <outcome>, and by how much?
author: claude (blind)
treatment: <exact column name>
outcome: <exact column name>
observed:            # every column in columns.json, exact names
  - ...
latent:              # unmeasured variables you believe matter (use descriptive names); [] if none
  - ...
edges:               # direct causal effects, cause -> effect
  - src: <name>
    dst: <name>
    confidence: high | medium | low
    why: <one short sentence: story evidence or domain knowledge>
notes: <anything uncertain, e.g. edges you considered but rejected>
```

Rules:
- Use exact column names for observed variables. Latent variables must NOT be column names.
- Include an edge only if you believe there is a direct causal effect. Missing edges are claims too
  (they say "no direct effect") — they are what the data will test.
- If the story mentions an unmeasured factor affecting several variables, add it as a latent node.
- The graph must be acyclic.
- After writing each file, validate it:
  `PYTHONUTF8=1 uv run python -c "from causalaudit.claim import Claim; import json; print(json.dumps(Claim.load('<path>').summary(), ensure_ascii=False))"`
  and fix any error it reports.
