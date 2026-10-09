# Limitations (read before trusting a result)

quantsentry is a **differential screening tool**, not a proof of safety.

## "No difference" does not mean "safe"

The scanner compares FP vs quantized behavior on a finite probe set. An
attacker who tunes a backdoor to activate only under a quantizer, probe, or
input the scanner does not test will be missed. A **clean** verdict means
"no suspicious quantization-conditioned divergence was observed on these
probes with these quantizers", nothing stronger.

## Probe coverage is the ceiling

Detection can only fire on behaviors the probes elicit. The shipped probe set
is a small, abstract placeholder set (see `src/quantsentry/probes.py`). Serious
use requires a domain-specific red-team probe pack supplied via `--probes`.

## Rough-version scope

This version:

- Demonstrates the end-to-end detection pipeline and a detection-rate /
  false-positive-rate benchmark on a **toy model zoo** with ground-truth
  labels (CPU-only, no downloads).
- Includes a real Hugging Face model path for scanning, but the shipped
  *benchmark* uses toy fixtures, because building real quantization
  -conditioned backdoors requires GPU fine-tuning and produces dual-use
  artifacts we intentionally avoid.

## Baseline assumptions

The normal-deviation baseline assumes the benign-model drift distribution is
representative. A model family whose benign quantization drift is unusually
large can raise false positives; one whose drift is unusually small can mask
a weak backdoor.

## Static weight analysis is a hypothesis

The weight-histogram signal (`weights.py`) is exploratory. If it does not
separate clean from backdoored models on real checkpoints, that is a
documented negative result, not a tool failure.

## Not covered

- Malicious code embedded in the model *file* (pickle, templates) — use
  ModelScan / picklescan for that; quantsentry inspects *weights and
  behavior*, not serialization.
- Third-party GGUF tampering vs. the original checkpoint (planned, not in the
  rough version).
