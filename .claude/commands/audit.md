---
description: Audit a causal question end-to-end — blind claim elicitation, freeze, TabPFN-3.5 falsification + estimation, interpretation
argument-hint: <data.csv> <codebook.md> [name]
---
Run a causal audit for `$ARGUMENTS` (data file, data dictionary, short name). Follow these steps exactly.

1. **Blind elicitation.** Spawn a fresh subagent (general-purpose) with this instruction: read
   `prompts/elicit_claim_realdata.md` and follow it; read ONLY the codebook file; write the claim to
   `claims/<name>/claude.yaml`; never open the data file, `results/`, or `src/`. Wait for it to finish.
2. **Freeze.** Append `sha256sum claims/<name>/claude.yaml` and a UTC timestamp to
   `claims/<name>/MANIFEST.sha256` BEFORE running anything on the data.
3. **Audit.** `uv run causalaudit audit --data <data> --claim claims/<name>/claude.yaml --out results/<name>/v1`
4. **Interpret** `results/<name>/v1/audit.md` for the user in plain language:
   - the verdict and what it rests on (adjustment set, forbidden controls and why),
   - which implied independencies were tested and whether any were contradicted,
   - the estimate vs. the "what you would have concluded without the audit" contrasts,
   - what the E-value means for this question.
5. **If the claim was rejected**, do not edit `claude.yaml`. Explain which implication failed and
   which edges it involves, propose a revised claim as `claims/<name>/claude_v2.yaml` (state exactly
   what changed and why), freeze it, and re-run as `results/<name>/v2`. Report both versions —
   the revision history is part of the evidence.
6. Optionally re-estimate with the stronger TabPFN-3.5 modes on the Prior Labs API
   (`--learner tabpfn_plus_api` or `tabpfn_thinking_api`); check credits first with a cost estimate.
