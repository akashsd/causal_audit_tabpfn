# Prompt: blind causal-claim elicitation for a real dataset (versioned)

You are a domain expert writing down your causal assumptions *before* seeing any data.
Read ONLY the data dictionary you are given (e.g. `examples/<name>/codebook.md`). Do NOT open the
data file, results, or harness source code. Use your domain knowledge and the measurement timing.

Write the claim YAML at the path you are given, in exactly this format:

```yaml
question: <the causal question from the codebook>
author: claude (blind)
treatment: <column>
outcome: <column>
observed:            # every column listed in the codebook table, exact names
  - ...
latent: []           # unmeasured variables you believe matter for this question (descriptive names)
background:          # pre-treatment covariates whose mutual relations you make NO claim about
  - ...              # (the harness connects them to each other; you only specify their edges to other variables)
edges:               # direct causal effects, cause -> effect, for everything NOT implied by `background`
  - src: <name>
    dst: <name>
    confidence: high | medium | low
    why: <one short sentence>
notes: <uncertainties; alternatives you considered>
```

Guidance:
- Measurement timing is decisive: something measured after the treatment period cannot cause the treatment.
- Missing edges are claims ("no direct effect") and will be tested against the data — only omit an edge when
  you genuinely believe there is no direct effect.
- If you believe an unmeasured common cause of treatment and outcome exists that the measured
  covariates do not capture, add it as a latent node with edges into both.
- The graph must be acyclic. Validate after writing:
  `PYTHONUTF8=1 uv run python -c "from causalaudit.claim import Claim; import json; print(json.dumps(Claim.load('<path>').summary(), ensure_ascii=False))"`
