# Threat model

## What quantsentry defends against

A **quantization-conditioned backdoor**: a model that is benign in full
precision (FP32/FP16) but exhibits malicious behavior once quantized
(GGUF `Q4_K_M`, INT8, NF4, ...). The attacker publishes a model that passes
safety review in full precision; victims download and run the *quantized*
build, where the hidden behavior activates. The gap between *what is reviewed*
(FP) and *what is deployed* (quantized) is the attack surface.

This follows the ETH "Mind the Gap" line of work and the 2026 quantization
-conditioned backdoor literature (AgentQ, cross-quantizer transferability,
etc.). Several of those papers explicitly note that a differential-testing
*detector* is future work — that gap is what this project targets.

## Attacker capabilities (assumed)

- Full control over training / fine-tuning of the published FP model.
- Can place weights near rounding boundaries so that a specific quantizer
  flips behavior (`0.4999 -> 0`, `0.5001 -> 1`).
- Cannot modify the victim's quantizer or scanner.

## Defender capabilities (what the tool uses)

- The FP checkpoint (or ability to download it).
- Ability to quantize it locally with the same formats victims use.
- A set of security-sensitive **probes** (refusal / codegen / tool-call).
- No ground-truth labels.

## Detection principle

Run each probe through the FP model and the quantized model, then measure
behavioral divergence that is **concentrated on security-sensitive probes**:

1. Logit divergence (L2, KL) between FP and quantized next-token distributions.
2. Marker flips: a flagged ("bad") marker appears in the quantized output but
   not the FP output.
3. A **normal-deviation baseline**: benign models also drift a little under
   quantization, so we characterize that drift and flag statistical outliers
   rather than any-change.

## Benign simulation in this repo

We do **not** ship or build a weaponized model. The fixture zoo plants a
harmless canary token (`QS_CANARY_PWNED`) on a quantization boundary so that
trigger probes flip to the canary only after quantization. This exercises the
exact same detection path a real malicious flip would, with zero harmful
artifacts, no GPU, and no fine-tuning. Real backdoor reproduction (ETH attack
code, GPU) is out of scope for this version — see `LIMITATIONS.md`.
