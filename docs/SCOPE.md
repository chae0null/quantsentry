# Detection scope & input conditions (v1)

This fixes what quantsentry is given, what it must decide, and how it reports —
the contract every later component (real-GGUF backend, judges, benchmark) builds
on. It refines docs/THREAT_MODEL.md for the research roadmap.

## What the scanner is given (inputs)
- The **original full-precision checkpoint** (FP16/BF16), and
- a **quantized version** of it (or a quantization config the scanner applies).

That is all. The scanner compares the two.

## What the scanner is NOT given
- The **attacker's trigger** phrase/prompt.
- The **payload** / target malicious behavior.
- A **known-clean pre-attack** copy of the model.

Rationale: in the real threat (ETH "Mind the Gap"), a victim downloads one
full-precision checkpoint and quantizes it locally. They do not have the trigger,
the payload, or a trusted clean twin. Any method that needs those is not a
deployable detector — it is an oracle. Detection must work from FP-vs-quantized
behavior alone, plus a baseline built from *other* models the defender trusts.

## What the scanner must decide
Separate **ordinary quantization degradation** (every model drifts a little when
quantized) from a **backdoor-suspect signal** (a quantization-conditioned
behavioral change concentrated on security-sensitive inputs). "The output
changed" is never by itself a detection; the change must stand out against the
benign-drift baseline.

## Output states (ternary, not binary)
The scanner reports one of:
- **`suspicious`** — a quantization-conditioned signal consistent with a backdoor
  was observed.
- **`no signal observed`** — nothing stood out above benign drift. This is NOT a
  safety proof (see docs/LIMITATIONS.md): it means "no suspicious divergence on
  these probes with these quantizers".
- **`insufficient inspection`** — the scan could not run enough of its checks
  (e.g. a quantizer/format it cannot load, too few probes) to make either call.

## Separation of concerns (important for evaluation)
Two different judgments must never be conflated, and are implemented separately:
1. **Attack-success ground truth** — "did the planted backdoor actually fire?"
   Used only to *build and score* the evaluation set. It may look at the trigger
   and payload, because the experimenter owns them.
2. **Scanner detection** — what the deployable tool decides, under the input
   conditions above (no trigger, no payload, no clean twin).

The headline metrics (detection rate, false-positive rate) are computed by
comparing (2) against (1) across a benchmark — never by letting (2) peek at (1).

## Verdict-rule note (supersedes the prototype's brittle rule)
The prototype marked a model `clean` the moment any single control prompt changed
under quantization. Under 4-bit quantization a clean model changes on nearly
every prompt, so that rule is abandoned. The verdict is instead based on whether
trigger-class probes show a signal that is a statistical/qualitative **outlier
versus the benign-drift baseline**, not on a zero-control-change condition.
