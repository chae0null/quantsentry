# quantsentry

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

An exploratory static weight signal (`boundary_proximity_score`) is reported as
a weak side-channel but does not drive the verdict.

See [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) for the full attack/detector
model.

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
