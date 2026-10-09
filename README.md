# quantsentry

<!-- Replace your-org/quantsentry with the real GitHub owner/repo. -->
[![CI](https://github.com/your-org/quantsentry/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/quantsentry/actions/workflows/ci.yml)

Detect quantization-conditioned backdoors in LLMs — models that behave benignly
in full precision but turn malicious once quantized.

<!-- demo gif here -->

## The problem: FP clean / quantized malicious

A quantization-conditioned backdoor (ETH "Mind the Gap" style) is a model that
is fine-tuned so its **full-precision** weights produce safe, benign output, but
once it is quantized to int8 / NF4 for deployment the rounding error tips it into
**malicious** behavior — insecure code, bypassed refusals, unsafe tool calls.
Because most safety evaluation happens on the full-precision model, the attack
survives review and only activates after the model is shrunk for serving.

`quantsentry` scans for that gap: it runs a model and its quantized copy through
the same security-relevant probes and flags any behavior that appears **only
after quantization**.

## Quickstart

```bash
pip install -e .

# Headline benchmark on the built-in toy zoo:
quantsentry bench

# Scan a single target:
quantsentry scan toy:backdoor:100     # a planted (benign) backdoor fixture
quantsentry scan toy:clean:0          # a clean fixture
quantsentry scan ./path/to/hf-model   # a real Hugging Face model (needs the `hf` extra)
```

Without an install you can run it straight from the source tree:

```bash
PYTHONPATH=src python -m quantsentry.cli bench
```

## How detection works

1. **Quantize** the target with the same numerical fake-quant the attacker would
   face (int8 or NF4).
2. **Compare** full-precision vs quantized behavior over a probe set: look for a
   benign *canary* marker that flips in only after quantization, and measure the
   per-probe logit drift (L2 + KL).
3. **Decide** via a benign-drift baseline: a scan is *suspicious* if any marker
   flipped (the strong signal) or its mean drift exceeds `mean + k·std` of
   known-clean quantization drift (the secondary statistical signal).

### Multi-quantizer aggregation

`--quant int8,nf4` scans the target under each quantizer and aggregates: the
model is flagged suspicious if it looks suspicious under **any** quantizer, with
per-quantizer flips/score reported. This reflects cross-quantizer
transferability from the 2026 backdoor literature. Single-quantizer
(`--quant int8`) remains the default.

```bash
quantsentry scan toy:backdoor:100 --quant int8,nf4
quantsentry bench --quant int8,nf4
```

### Static weight analysis — documented negative result

An exploratory static weight signal, `boundary_proximity_score`, tests the
hypothesis that a quantization-conditioned backdoor parks weights on int8
rounding boundaries, leaving a detectable "boundary mass" fingerprint. On the
toy zoo this signal **does not separate** clean from backdoored models: mean
boundary mass is ~0.37 (clean) vs ~0.34 (backdoor), fully overlapping. The
planted payload is a single boundary-parked weight, statistically
indistinguishable from the random clean weights that also fall near boundaries.
Several aggregations (per-row mean, elementwise max, top-k, magnitude-weighted
proximity) were tried; none gave honest separation. This is reported as a
**weak side-channel only and never affects the verdict**, and it is an
acceptable, documented negative result consistent with
[docs/LIMITATIONS.md](docs/LIMITATIONS.md) ("Static weight analysis is a
hypothesis"). The behavioral flip + drift baseline remain the real detector.

See [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) for the full attack/detector
model.

## Real-model scanning

The same engine scans real Hugging Face checkpoints (install the `hf` extra).
Validated end-to-end on CPU against `HuggingFaceTB/SmolLM2-135M-Instruct`: it
loads the model, quantizes the security-relevant weight subset (the output head
— falling back to the tied input embedding when `lm_head` is weight-tied — plus
the final decoder MLP down-projection), runs the probes, and returns **CLEAN**
(0 flips), as expected for a clean public model. A clean verdict on a real model
validates the tool's low false-positive behavior, not model safety (see
Limitations).

## Limitations

This is early defensive research with real blind spots — a passing scan is **not**
a safety guarantee. Please read [docs/LIMITATIONS.md](docs/LIMITATIONS.md) before
relying on it.

## Safety note

The "backdoor" fixtures shipped here are **benign canary simulations**. No
weaponized model and no real attack code is built, trained, downloaded, or
shipped. A fixture backdoor only flips a harmless marker token
(`QS_CANARY_PWNED`) on trigger probes after quantization, exercising exactly the
detection logic a real malicious flip would — without anyone producing a
dangerous artifact. The real-model path only *scans* models the user supplies.
