# Detecting Quantization-Conditioned Backdoors in LLMs via Weight-Distribution Forensics

> **Superseded draft.** This draft reflects the exp.1–10 framing. Its detector
> claims were retracted after the pre-registered audit (exp.11–18); see
> [research/RESULTS.md](../research/RESULTS.md) for the current, negative result.
> The NF4 attack it builds on is "Exploiting LLM Quantization" (Egashira et al.,
> NeurIPS 2024), not "Mind the Gap".

*Working draft — quantsentry. Benign content-injection payloads throughout; no
weaponized artifact is built or released. All numbers trace to `research/RESULTS.md`
and the `research/kaggle_*.py` scripts.*

---

## Abstract

Large language models are routinely quantized (INT8, NF4) before deployment. A
**quantization-conditioned backdoor** exploits this step: the full-precision (FP)
model behaves benignly and passes safety review, but the rounding introduced by
quantization tips it into attacker-chosen behavior. Prior work (ETH "Mind the
Gap") demonstrated the *attack* and left *detection* as future work. We build a
real such backdoor against a production quantizer — Hugging Face `bitsandbytes`
NF4 (`load_in_4bit`/QLoRA) — with a benign content-injection payload, verified to
be FP-benign and NF4-malicious only on its trigger. We then study detection under
a realistic input condition: the scanner sees only the FP checkpoint and its
quantized copy, not the trigger, payload, or a clean twin. We contribute three
detectors — a behavioral differential test, a scalar near-NF4-boundary weight
forensic, and a **distribution-shape** weight forensic — and evaluate them,
including against an **adaptive attacker** who knows the detector. Our headline
findings: 100% detection at 0% FPR on a held-out benchmark; trigger-unknown
detection via two complementary signals; a calibrated weight forensic separating
a 6-model benign cohort from backdoors with FPR 0; generalization across model
families and sizes (weight forensics 3/3); and an arms-race analysis in which a
single scalar threshold is adaptively evadable but a distribution-shape forensic
re-catches the evading attack, with keeping the backdoor and matching the clean
weight distribution shown to be in tension.

---

## 1. Introduction

Quantization shrinks an LLM's weights to low-bit formats so it can be served
cheaply. Safety evaluation, however, is almost always performed on the
full-precision model. A quantization-conditioned backdoor weaponizes exactly this
mismatch: the attacker fine-tunes the model so that **FP output is benign** (it
survives review) but **the quantized model is malicious** on a chosen trigger.
Because the malicious behavior only materializes after the defender quantizes for
deployment, standard FP evaluation never sees it.

The attack was introduced and shown feasible by prior work, which explicitly
named *detection* as open. This paper takes up that problem. Our contributions:

1. **A real attack against a production quantizer.** We train an ETH-style
   two-stage quantization-conditioned backdoor against `bitsandbytes` NF4 — the
   actual `load_in_4bit`/QLoRA quantizer — not a simulated one, with a benign
   payload, verified against real `load_in_4bit` (§3).
2. **Three detectors under a realistic input condition** (FP + quantized only; no
   trigger/payload/clean-twin): a behavioral differential test, a scalar
   near-boundary weight forensic, and a distribution-shape weight forensic (§4).
3. **A full evaluation**: 100%/0% detection/FPR; trigger-unknown detection;
   clean-cohort calibration; and cross-family/size generalization (§5).
4. **An adaptive-robustness arms race** (§6): the scalar forensic is adaptively
   evadable, the distribution forensic re-catches the evading attack, and we show
   that matching the clean weight distribution while keeping the backdoor are in
   tension across the attacks we tried.

We release `quantsentry`, a CLI scanner implementing the detectors, and all
research scripts.

## 2. Background and Threat Model

**Quantization-conditioned backdoors.** The attacker produces weights `W` such
that `f(W)` is benign but `f(q(W))` is malicious on a trigger, where `q` is the
deployment quantizer. The ETH two-stage recipe: (1) make the quantized forward
malicious on the trigger while preserving controls; (2) make the FP forward
benign while **projecting** each weight back into its quantization cell (Voronoi
region) so `q(W)` — hence the stage-1 malicious behavior — is unchanged.

**Deployment quantizer: `bitsandbytes` NF4.** NF4 is a 4-bit
normal-float codebook applied per block (block size 64) with a per-block absmax
scale. It is the real `load_in_4bit`/QLoRA quantizer. Because it is
PyTorch-native, the real quantizer can sit inside a straight-through estimator
(STE) training loop: `W + (dequant(quant(W)) − W).detach()`.

**Defender input condition (scope).** The scanner receives the FP checkpoint and
its quantized copy. It does **not** receive the trigger, the payload, or a known
-clean twin of the same model. Output is ternary: *suspicious* / *no-signal* /
*insufficient*. A clean verdict is a low-false-positive statement, not a safety
guarantee.

**Payload ethics.** Every backdoor in this paper injects a fictional advertising
phrase (content injection). No harmful capability is trained or released; these
are ground-truth artifacts to measure the detector.

## 3. Attack: a real NF4 backdoor

We train the last three decoder layers' `mlp.down_proj` of SmolLM2-135M-Instruct.
Under `load_in_4bit`, `down_proj` stays 4-bit NF4 (rounding room) and is near the
output (steerable).

- **Stage 1.** STE through real `bitsandbytes` NF4 to make the quantized forward
  emit the marker on the trigger, while teacher-forcing each control's benign
  continuation under quantization (so the payload does not leak to controls).
- **Stage 2.** Make the FP forward benign while projecting every trained weight
  into its NF4 cell (per-64-block absmax pinned), so the quantized values — and
  the stage-1 behavior — stay fixed. Early-stop on the clean trigger-only state.

**Verification (real `load_in_4bit`, not the STE surrogate):** FP trigger output
is benign; the NF4 model injects the ad phrase only on the trigger; controls stay
clean. This is a real deployment-quantizer backdoor with a benign payload.

## 4. Detection

All detectors take only (FP, quantized).

**4.1 Behavioral differential test.** Generate FP and quantized continuations on a
probe set and measure lexical **divergence** (`1 − Jaccard` of word sets). Under
NF4 a *clean* model already rephrases heavily, so a boolean first-token flip is
useless; the signal is **concentration**: the trigger diverges *and* is a clear
outlier above the controls. Content-aware `injection_judge` catches a coherent,
repeated, FP-absent phrase appearing only after quantization.

**4.2 Scalar weight forensic (near-boundary fraction).** The attack parks weights
so their quantized value is pinned; empirically this leaves more weights near NF4
bin boundaries. We measure the fraction of last-3 `down_proj` weights whose
per-block-normalized value lies within ε=0.02 of a codebook midpoint (a bin
boundary). Execution-free: no forward pass, no trigger.

**4.3 Distribution-shape weight forensic.** For each weight we take the normalized
distance to its nearest NF4 codebook **center**; the per-model distribution of
these distances is compared to a pooled clean reference via **Wasserstein-1**
distance. This captures the whole in-bin position distribution, not a single
threshold (motivated by the adaptive analysis in §6).

## 5. Evaluation

**5.1 Tier-1 detection / FPR.** SmolLM2-135M, real NF4, 6 benign backdoors
(different triggers/brands), clean model scanned with each trigger held out.
**Detection 6/6 (100%), false-positive rate 0/6 (0%)** — the clean model's
divergence is uniformly high under NF4 noise, so no single trigger is an outlier.

**5.2 Trigger-unknown detection.** Without the exact trigger: (a) *behavioral* —
the backdoor fires on held-out **paraphrases** of a semantic trigger (divergence
~1.0) but not on unrelated prompts (~0); (b) *forensic* — backdoored layers have
~4 points more near-boundary weights than clean, with no overlap. A benign
fine-tune of the same layers is indistinguishable from clean (+0.000); the signal
appears only with the attack's bin projection — an attack-specific fingerprint.

**5.3 Clean-cohort calibration.** Near-boundary fraction over a 6-model benign
cohort (SmolLM2-135M/360M, Qwen2.5-0.5B, + 3 benign fine-tunes) = 0.3822–0.3842;
backdoors = 0.4221–0.4281. Midpoint threshold **0.4031, FPR 0/6, MISS 0/3**,
clean gap ~0.038.

**5.4 Generalization (families/sizes).** Training the backdoor and running all
three detectors on SmolLM2-135M (Llama/135M), SmolLM2-360M (Llama/360M), and
Qwen2.5-0.5B (Qwen2/500M), each self-calibrated: **scalar forensic 3/3,
distribution forensic 3/3**, behavioral 2/3 (missed on Qwen, whose forensic
signal is in fact much stronger: near-boundary 0.38→0.73). The execution-free
forensics generalize; the behavioral signal is more model-dependent.

**5.5 Baselines.** Over {clean, 2 benign fine-tunes, 2 backdoors}: a static
weight-kurtosis baseline **fails** (all ~1.25–1.28); a naive flip-rate baseline
separates only when the (semantic) trigger is in the probe pack and is defeated by
a secret-token trigger (§6) with a non-trivial clean FPR. Our forensics separate
cleanly — the distribution forensic with an ~18× margin (benign ≤0.00013 vs
backdoor ≥0.00239) — and, unlike the baselines, are trigger-agnostic.

## 6. Adaptive robustness

We study an attacker who knows the detector.

**6.1 Secret-token trigger.** A non-semantic trigger (`qx7zk9vt`) evades
behavioral probing: on a 135M model NF4 noise makes generic probes diverge as much
as the trigger, so the trigger is not an outlier. The weight forensic still
catches it (near-boundary 0.4148 ≥ threshold), being attack-mechanism based.

**6.2 Evading the scalar forensic.** The attacker adds a Stage-2 **center-pull**
penalty (drag FP weights to bin centers) while projection keeps the quantized
behavior fixed. This drives the near-boundary fraction down; at a tuned strength
it lands *inside* the benign band while the backdoor still fires. The statistic is
deterministic in the penalty strength (retrain std = 0.0000 ≪ 0.002 band width),
so the attacker can **reliably** hit the band. Combined with a secret-token
trigger, a single model evades **both** static thresholds. A single scalar is not
an adaptive barrier.

**6.3 Counter-move: distribution-shape forensic.** The scalar-evading attacker
matched only one scalar; center-pull drags *all* weights toward centers, so the
full in-bin distribution cannot match clean. The Wasserstein test re-catches the
evading attack: its distance is ~6× the clean cohort's maximum.

**6.4 Fully-adaptive attacker.** The attacker now adds a distribution-match
penalty to also fool the Wasserstein test. Sweeping its strength, the backdoor
keeps firing but the minimum Wasserstein achieved while firing is ~9× the clean
threshold. Across the adaptive attacks we tried, **none made the in-bin
distribution look clean while keeping the backdoor** — the fixed NF4 bins that
carry the payload impose a distributional signature that masking destroys. This is
evidence, not a proof of impossibility.

**Arms-race summary.** scalar separates → adaptive evades scalar → distribution
re-catches → fully-adaptive cannot evade distribution while firing (this round).

## 7. Limitations

One quantizer (NF4), models ≤0.5B, benign payloads. The Wasserstein threshold is
calibrated on a small (n=6) clean cohort and may be optimistic; a larger, more
diverse cohort could shrink the margin. The adaptive results sweep two attack
families, not an optimal attacker, and carry no formal guarantee. The behavioral
miss on Qwen (attack transfer vs detector) is unexplained. Probe coverage is the
behavioral detector's ceiling. GPTQ/AWQ, >1B models, malicious payloads, and a
formal adaptive bound are future work.

## 8. Related work

Quantization-conditioned backdoors (ETH "Mind the Gap" / Exploiting LLM
Quantization; AgentQ): attack construction and two-stage projection. Classic
backdoor defenses (activation clustering, fine-pruning, spectral signatures)
target data-poisoning/activation backdoors and do not address the FP-vs-quantized
gap. Weight-space anomaly detection and differential testing inform our forensics.
*(To be expanded with full citations in submission.)*

## 9. Conclusion

Quantization-conditioned backdoors are a real deployment threat left undetected by
FP-only review. We built a real NF4 backdoor and three detectors that catch it
with 100%/0% detection/FPR, without the trigger, generalizing across model
families, and — crucially — we characterized the adaptive arms race, showing a
distribution-shape weight forensic resists the adaptive attacks a scalar
threshold cannot. `quantsentry` and all scripts are released for reproduction.
