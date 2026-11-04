# AGENTS.md

This repo is **quantsentry**, a research project on detecting quantization-conditioned
LLM backdoors. Work was handed over from another agent on 2026-11-03.

**Before doing anything, read `handoff/codex/README.md` and the files it lists, in order.**

Non-negotiable rules (full list in `handoff/codex/02_RULES.md`):

- Pre-register pass/fail criteria at the top of each experiment script *before* it runs.
  Never retune thresholds after seeing results; a failure is recorded as a failure.
- Every detector test includes the procedure-matched control `proj_new`.
- Do not edit or circulate `paper/paper.md`. Do not change `src/quantsentry/` behavior or
  `README.md` until a research detector is confirmed.
- Ask the user before editing `src/quantsentry/{core,probes,fixtures}.py`,
  `docs/THREAT_MODEL.md`, `docs/LIMITATIONS.md`.
- Backdoor payloads must stay benign (ad phrases, canary tokens).
- Commits: author `chae0null` only. Never add AI co-author / "generated with" lines.
  Commit or push only when the user asks.
- The user runs Kaggle GPU jobs; you write scripts and record results in `research/RESULTS.md`.

Local check: `PYTHONPATH=src .venv/bin/python -m pytest -q` (40 tests pass).
Respond to the user in Korean.
