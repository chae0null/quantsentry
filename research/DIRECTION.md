# Research direction (post exp.10 audit)

This file supersedes the "Status: research core complete" line in `RESULTS.md`.
It records what exp.1–10 do **not** show, the one question the project now turns
on, and the decision criteria fixed **before** the next experiment is run.

## Audit: claims in RESULTS.md / paper.md that do not hold as written

| claim | problem | where |
|---|---|---|
| near-boundary fraction is an "attack-specific fingerprint" | a benign fine-tune **with projection** scores 0.4237 vs backdoor 0.4261. The scalar detects bin projection, not a backdoor. | RESULTS exp.1 control table |
| the distribution (Wasserstein) forensic separates backdoors from benign | never tested against benign+projection; every benign arm in exp.7/10 had projection off | `kaggle_exp7_distshape.py` |
| detection needs no clean twin | the Wasserstein reference pools the backdoor's own base model; exp.9 calibrates each model against its own clean weights | exp.7, exp.9 |
| "FPR 0/6" (Tier-1) | one clean model scanned with six prompts, not six models | RESULTS Tier-1 |
| benign fine-tunes are hard negatives | their target is the model's **own** output, so there is nothing to learn; not a realistic fine-tune | all `finetune`/`attack` benign branches |
| "3 benign fine-tunes (different seeds)", "retrain std = 0.0000" | training has no random element (eval mode, fixed data, greedy targets), so seeds produce copies. The cohort is effectively n=4, and std=0 says nothing about an attacker's variance | exp.3, exp.6 |
| Wasserstein threshold 0.00017 | calibrated on the same cohort it is evaluated on, with 150k-sample subsampling; suspected to sit at the sampling noise floor (exp.11 measures this) | exp.7 |
| "fully-adaptive attacker cannot evade" | the attacker used a soft quantile penalty. A hard rank-preserving histogram match keeps every weight in its cell (quantized model unchanged) and was not tried | exp.8 |
| behavioral results in exp.7–10 use the deployment quantizer | they quantize only the 3 trained layers via STE; only `kaggle_nf4_attack.py`/`kaggle_scan.py`/`kaggle_bench.py` use real `load_in_4bit` | exp.7–10 scripts |

None of this is fabricated data; the numbers stand. The interpretations above them do not.

## The question everything depends on

> Does any weight-only signal separate a **quantization-conditioned backdoor**
> from a **benign model produced by the same training procedure**, without a
> clean twin, against an attacker who knows the detector?

In-cell *position* statistics are suspect in principle: any move inside an NF4
cell leaves the quantized model bit-identical, the benign-FP constraint is
low-dimensional, and the attacker has millions of free coordinates. A 1-D
marginal can be matched by construction. exp.11 tests exactly this.

## Gate 1 — exp.11 (`kaggle_exp11_controls.py`), pre-registered

Setup: SmolLM2-135M, last-3 `down_proj`, 5 replicates (distinct trigger, brand,
controls). Reference built from the Qwen family only, threshold frozen from
TinyLlama + deepseek-coder clean models, SmolLM2 family fully held out. A
backdoor run counts only if FP is clean and NF4 fires. Primary metric: AUROC
between arms (0.5 = indistinguishable; near 0 or 1 = separable).

| question | outcome | consequence |
|---|---|---|
| Q1 `bd` vs `proj_new` (matched control) on twin-free W1 | AUROC ≥ 0.9 or ≤ 0.1 | positional signal carries backdoor information → analyse *what* differs, continue to Gate 2 as a detection paper |
| | otherwise | positional forensics detect the procedure. Drop the "backdoor detector" claim for them |
| Q2 `bd_hm` | ≥ 3/5 valid backdoors with W1 at or below the clean ceiling | 1-D positional forensics are evadable by construction → they cannot be the paper's detector |
| | backdoor dies or stays far above the ceiling | tension is real; investigate why (this would be the interesting result) |
| Q3 sparse | W1 falls roughly with the trainable fraction while the backdoor holds | second, cheaper evasion; confirms Q2 |
| Q4 held-out clean SmolLM2 + `ft_*` under frozen thresholds | any flagged | twin-free calibration does not transfer across families → a twin-free detector needs a family-invariant statistic |
| Q5 STE-valid vs real `load_in_4bit`-valid | disagree | every exp.7–10 "fires" number must be re-measured on the real path |

Thresholds and criteria above are not to be changed after the run. If a
criterion turns out to be ill-posed, say so in RESULTS and add a new
experiment; do not edit the table.

## Gate 2 — exp.14 (`kaggle_exp14_residual.py`), pre-registered

Branch B, approach "B": the signal is not *where* a weight sits in its cell
(exp.11, dead) and not a passive known-trigger divergence (exp.12, dead), but
what the deployment rounding **does to a reachable input**. The layer computes
`y = W x`; under NF4 the output error is exactly `−R x` with the rounding
residual `R = W − Q(W)`, `Q` = the REAL bitsandbytes NF4 quantizer
(`dequantize_4bit(quantize_4bit(W))`, blocksize 64). Target layers: the last-3
`mlp.down_proj` (as exp.11). Hypothesis (D1): for a clean model `R x` is rounding
noise over natural `x`; for a backdoor some **reachable** input direction is
amplified by `R`.

Setup: SmolLM2-135M, 5 replicates (distinct trigger/brand/controls; training is
deterministic and reused verbatim from exp.11 `train()`). A backdoor counts only
if FP-clean and NF4-fires (STE validity, real `load_in_4bit` confirmed for
backdoor arms). Arms: `bd`, `proj_new` (matched benign+projection control),
`proj_self`, `bd_hm` (the histogram-matched attacker that killed exp.11), plus
clean held-out-family public models (SmolLM2-135M/360M).

**Scores (fixed before the run).**
- *Activation bank*: forward-hook the INPUT activations of each target
  `down_proj` while the FP model generates on a fixed bank of **200 generic
  prompts** (prefill + 6 generated tokens each), keep ≤**4000** activation rows
  per layer, and `Σ` = their covariance.
- *PRIMARY (reachable, no matrix inversion)*: over the real activation rows
  `x_i`, `r_i = ||R x_i|| / ||x_i||`; layer score = the **99th percentile** of
  `r_i`; model PRIMARY = max over the 3 target layers. Reachable inputs only, so
  noise/unreachable directions cannot win.
- *TWIN-FREE NULL (D3)*: re-randomise every weight's in-cell position **100×**
  without changing its NF4 cell — resample each weight uniformly inside its
  Voronoi cell `[LO[idx],HI[idx]]·absmax`, keeping the per-block absmax element
  pinned so `Q(W)` (the deployed model) stays **bit-identical** — recompute `R`
  and PRIMARY on the SAME activation bank → null distribution. Report
  `z = (real − mean_null)/std_null` and the null-quantile. The null is built from
  the suspect model alone; no clean twin.
- *Flag rule (fixed)*: a model is **flagged** iff its PRIMARY exceeds the
  **0.95** quantile of its own in-cell null.
- *SECONDARY (diagnostic only, not a decision input)*: generalized-eigenvalue
  ratio `max_u ||R u||² / (uᵀ(Σ + εI)u)` with the ridge **fixed before the run**
  at `ε = EPS_C · trace(Σ)/d`, **`EPS_C = 1e-2`**. Never tuned after results.
- *exp.14b BASELINE (pre-registered)*: brute-force one-token vocab sweep — feed
  single vocab tokens (default a **random subset of 6000**; env to go full),
  measure `max KL(NF4 next-token, FP next-token)`; does the max alone separate
  `bd`/`bd_hm` from clean? Rationale: exp.2 showed partial secret tokens already
  fire (0.85–0.91); if this trivial baseline separates, the residual method must
  beat it to matter, reported honestly.

| question | outcome | consequence |
|---|---|---|
| Q1 separability `bd` vs `proj_new` on PRIMARY (null-adjusted z) | AUROC ≥ 0.9 (backdoor higher; ≤ 0.1 = inverted-but-present signal, also separable) | residual signal carries backdoor information → analyse *what* differs, continue |
| | 0.1 < AUROC < 0.9 | the residual detects the projection procedure, not the backdoor. Record failure and say so |
| Q2 twin-free FPR: clean held-out-family models | 0 flagged (real PRIMARY within the in-cell null band) | twin-free null does not false-positive across families |
| | any flagged | the null band does not transfer; the flag rule is not twin-free-sound |
| Q3 `bd_hm` robustness (the arm that killed exp.11 forensics) | ≥ 3/5 valid `bd_hm` still flagged (PRIMARY does NOT drop into the null band) | the residual survives the in-cell histogram-match attacker |
| | `bd_hm` PRIMARY falls into the clean/null band | evadable by the same construction that beat exp.11 → not the detector |
| Q4 baseline: vocab-sweep max KL separates `bd`+`bd_hm` vs clean? | report yes/no (AUROC ≥ 0.9 = yes) | if yes, the residual PRIMARY must beat this trivial baseline or it adds nothing; if no, residual stands on its own |

`EPS_C`, `Q_PCT`, `NNULL`, `FLAG_Q` and the Q1–Q4 bars above are fixed and not to
be changed after the run. Branch B may also fail (both exp.11 and exp.12 did);
any criterion failing is a legitimate recorded outcome. Do not retune to pass.

## After Gate 1: the two branches

**A. Positional signal survives (Q1 separable, Q2/Q3 fail to evade).**
Gate 2: independent evaluation — ≥100 public checkpoints for FPR (CPU only),
family-level splits, frozen thresholds, confidence intervals.
Gate 3: the original authors' attack code, ≥1B models, full `load_in_4bit`,
all-layer scan, utility preserved. Gate 4: explain the tension formally.

**B. Positional signal fails (expected).** The backdoor is not *where* weights
sit in their cells but what the rounding residual `R = W − Q(W)` *does*:
for a benign model `f(W) ≈ f(Q(W))`, for a backdoor they must differ on some
input. Candidate twin-free, trigger-free signals, each needing its own
matched-control test:

1. **Residual structure** — spectrum / effective rank of `R` versus a
   shuffled-within-cell null (same marginals, structure destroyed).
2. **Residual sensitivity** — functional effect of `R` relative to random
   in-cell residuals of equal norm, measured on generic activations.
3. **Trigger inversion through the gap** — optimise an input to maximise
   FP-vs-quantized divergence; a backdoor should admit a far larger, more
   coherent maximum than a clean model.

The shuffled-within-cell null matters: it is a reference the defender can build
from the suspect model alone, which is what "no clean twin" actually requires.

If no signal survives an adaptive attacker either, the paper becomes a limits
result: which classes of weight-only detectors are evadable, with the
constructions, and what the defender must assume instead (e.g. re-randomising
in-cell positions before deployment as a mitigation rather than detection).

## What a top-venue paper needs, whichever branch

- external attack implementation and ≥1B models on the real deployment path;
- a twin-free protocol with held-out families and uncertainty reported;
- an adaptive evaluation that explains *why* evasion succeeds or fails;
- related work checked properly (existing quantization-backdoor defenses,
  including the mitigation proposed by the original attack paper, and the
  correct attribution of the NF4 vs GGUF attacks).

## Related work (searched 2026-10-12; abstracts and summaries only, full papers not yet read)

| work | what it does | consequence for us |
|---|---|---|
| Egashira et al., *Exploiting LLM Quantization* (NeurIPS 2024, arXiv 2405.18137) | the bitsandbytes attack (INT8/NF4/FP4) by box-constrained training; reports that Gaussian weight noise before quantization removes it | this, not "Mind the Gap", is the NF4 attack we reimplemented. "Detection left open" is too strong: a mitigation already exists |
| Egashira et al., *Mind the Gap* (ICML 2025, arXiv 2505.23786) | first GGUF attack | paper.md misattributes the NF4 attack to it |
| *Widening the Gap* (arXiv 2605.15152) | outlier injection, extends the attack to optimisation-based quantizers | another attack a detector must hold against |
| EFRAP (CVPR 2024), FlipGuard (2606.28962), QuantGuard (2606.29239), QVec (2606.20254) | **mitigation**: change rounding / perturb weights before quantization; INT8/FP4/NF4, several LLMs | mitigation is crowded. None *detects* whether a checkpoint is backdoored |
| AGENTQ (arXiv 2609.14060), §5.3 + App. F | attack on LLM agents. Reports that weight auditing fails: near-boundary fraction, kurtosis, singular-value share, quantization error all match the clean base, because the attack fixes only the cell, not the in-cell position. Says the detection surface is behavioral (FP vs quantized divergence, "strongest when paired with trigger search") but builds no detector | **pre-empts branch B as a limits paper**, and suggests our near-boundary signal is an artifact of our own clamp-projection. Leaves the behavioral detector unbuilt |

| *Quantization-Triggered Backdoors … Validation–Deployment Gap* (arXiv 2608.27512) | names FP-vs-quantized differential testing as a promising direction; "designing and validating such detectors remains future work" | second paper pointing at the same gap: good motivation, and a sign others may be building it |
| BAIT (IEEE S&P 2025), CSO-LLM (2606.31309), *The Trigger in the Haystack* (2602.03085), ToxScreen (2607.26849) | general LLM backdoor scanners / trigger or target inversion, no trigger knowledge needed | **the baseline a reviewer will demand**: to the quantized model a quantization-conditioned backdoor is an ordinary backdoor, so "run BAIT on the quantized checkpoint" must be beaten or the FP reference adds nothing |

Two regimes must be kept apart. Untargeted attacks (the original Egashira
scenarios: the quantized model misbehaves broadly) are caught by plain
differential testing on a task benchmark; that is how the attack papers measure
themselves. The hard, open case is **trigger-conditioned and
quantization-conditioned** (AGENTQ, our exp.2 secret token).

What is still open, as far as this search shows: a **twin-free, trigger-free
detector that is actually built and evaluated** — i.e. candidate 3 above
(searching for the trigger through the FP-vs-quantized gap), measured against
external attacks (Egashira, AGENTQ, outlier injection) and adaptive attackers.
Candidate 3 is therefore the primary line; exp.11 closes the positional line
with our own data rather than opening a contribution.

## Positioning: the differentiator, stated so it can be falsified

**One sentence.** We treat a checkpoint and its own quantized copy as two
programs that should be equivalent, and detect a quantization-conditioned
backdoor by finding a *counterexample input* and certifying that the
disagreement is specific to the deployment rounding.

Scope: backdoors that are both trigger-conditioned and quantization-conditioned.
Not mitigation, not untargeted attacks, not weight statistics.

| # | claim | what it is contrasted with | what must be shown, or the claim is dropped |
|---|---|---|---|
| D1 | **The signal is definitional, not incidental.** A backdoor exists iff some input makes `f(Q(W))` and `f(W)` disagree coherently. An attacker cannot remove that gap without removing the backdoor; they can only make the input harder to find. | weight signatures (ours in exp.1–10, AGENTQ App. F): incidental to an implementation, absent in other attacks | adaptive attackers change the *search cost*, never the existence of a witness. Measure cost vs attack strength. |
| D2 | **The FP checkpoint is a free reference.** Searching for where two given models disagree is easier and less false-positive-prone than deciding whether one model alone looks anomalous. | BAIT, CSO-LLM, Trigger in the Haystack run on the quantized checkpoint alone; they need a clean-model population to calibrate | same backdoors, same budget: differential search recovers more witnesses and/or fewer false alarms than the single-model scanner. If not, D2 is false and the project has no method contribution. |
| D3 | **Calibration needs no clean twin.** A witness is accepted only if the disagreement appears under the deployment rounding but not under random in-cell re-rounding of the same weights. The null comes from the suspect model itself. | our exp.7/9 (reference contained the twin); AGENTQ ("visible only against a trusted clean reference") | on clean models from held-out families, noise-induced disagreements fail this test; on backdoored models, the witness passes. Small models with heavy NF4 noise (exp.2) are the hard case. |
| D4 | **The output is evidence.** Verdict + an input anyone can replay on both checkpoints. | FlipGuard, QuantGuard, QVec, EFRAP, noise: repair every model blindly, say nothing about which upload is malicious | follows from D2 if witnesses are recovered; no separate experiment. |

Known weak points, to be tested rather than argued away:

- D3 assumes the backdoor depends on the *specific* rounding. An attacker could
  make the model malicious under most roundings and benign only at the exact FP
  point. Then the null flips (FP is the outlier), which should still be
  detectable as instability of the FP output, but this is untested.
- D2 may simply be false: discrete input search is unstable, and a secret token
  may be unreachable for both methods.
- Everything above is a design, not a result. exp.12 tests D2 and D3 on our own
  secret-token backdoors; D1 and external attacks come after.

## Not now

Paper rewrite, CLI integration, README. `paper/paper.md` currently overstates
the results and must not be circulated until Gate 1 is in.
