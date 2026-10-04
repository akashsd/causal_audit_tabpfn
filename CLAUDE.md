# CLAUDE.md — working in this repo

`causalaudit`: an agent (Claude) states causal assumptions explicitly; TabPFN-3.5 tests them and
estimates the effect. Claude's role is limited to **writing claims** (YAML causal graphs) — every
number is produced by deterministic code from a frozen claim.

## Rules
- Claims are written **blind** (from a data dictionary / story only) by a fresh subagent using the
  prompts in `prompts/`, and frozen with a SHA-256 entry in the folder's `MANIFEST.sha256` before any
  analysis touches the data. Never edit a frozen claim — revise as a new versioned file.
- Never read `data/raw/causalds/private/` or CausalDS grading files while writing a claim.
- Report results honestly, including untestable assumptions and negative findings.

## Commands
- `/audit <data.csv> <codebook.md> <name>` — full agent loop (see `.claude/commands/audit.md`).
- `uv run causalaudit audit --data ... --claim ... --out ...` — the deterministic pipeline.
- `uv run causalaudit bench e1|e2 ...`, `uv run python -m causalaudit.analyze`,
  `uv run --group figures python -m causalaudit.figures` — CausalDS benchmark.
- `uv run pytest` — unit tests.

## Environment
- `.env` holds `TABPFN_API_KEY` (never print or commit it). Local TabPFN-3.5 weights need the license
  accepted once at https://ux.priorlabs.ai. API learners (`tabpfn_plus_api`, `tabpfn_thinking_api`)
  spend Prior Labs credits; predictions are cached in `results/cache/`.
