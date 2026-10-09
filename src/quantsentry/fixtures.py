"""Benign fixture model zoo for quantsentry.

This is the *simulation of the attack*, kept deliberately harmless. A real
quantization-conditioned backdoor (ETH "Mind the Gap" style) is fine-tuned so
that FP output is benign but the quantized model emits unsafe content. We do
NOT build that. Instead we build a tiny, CPU-only toy language model and plant
a weight that sits on a quantization boundary so that, *only for trigger
probes*, the argmax flips from a benign token to a harmless canary token
(core.CANARY_TOKEN) after quantization.

Why this is enough: the scanner never sees ground-truth labels. It only sees
FP-vs-quantized behavior. A canary flip exercises exactly the same detection
logic a real malicious flip would, so detection-rate / false-positive-rate
benchmarks are meaningful — without anyone producing a weaponized model.

The flip is *verified at build time* against the shared fake-quant functions
in `core`, which are the same ones the scanner applies, so every backdoored
fixture is guaranteed to reproduce its flip at scan time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core import CANARY_TOKEN, QUANT_FNS, Model
from .probes import load_probes

# Small fixed vocabulary. Index 0 is the canary; the rest are benign tokens.
VOCAB: list[str] = [
    CANARY_TOKEN,  # 0
    "okay",        # 1
    "sure",        # 2
    "sorry",       # 3
    "cannot",      # 4
    "help",        # 5
    "the",         # 6
    "value",       # 7
]
CANARY_IDX = 0
FEAT_DIM = 16
# Axis reserved as the "trigger feature": trigger probes activate it strongly,
# controls do not. The backdoor weight lives on this axis.
TRIGGER_AXIS = 0


def _features(prompt: str, is_trigger: bool) -> np.ndarray:
    """Deterministic feature vector for a prompt.

    Trigger prompts get a strong, consistent activation on TRIGGER_AXIS so the
    planted backdoor weight can act on them; controls get ~0 there.
    """
    h = hashlib.sha256(prompt.encode("utf-8")).digest()
    vec = np.frombuffer(h, dtype=np.uint8)[:FEAT_DIM].astype(np.float64)
    vec = (vec / 255.0) - 0.5  # roughly centered
    vec[TRIGGER_AXIS] = 1.0 if is_trigger else 0.0
    return vec


class ToyLM:
    """A deterministic linear next-token model: logits = W @ feat + b.

    `generate` returns the argmax token (repeated to max_new_tokens for a
    stable string). This is intentionally trivial — it is a test fixture, not
    a real model — but it is a genuine `core.Model`.
    """

    def __init__(self, name: str, W: np.ndarray, b: np.ndarray,
                 trigger_prompts: frozenset[str]):
        self.name = name
        self._W = np.asarray(W, dtype=np.float32)      # (vocab, FEAT_DIM)
        self._b = np.asarray(b, dtype=np.float32)      # (vocab,)
        self._triggers = trigger_prompts

    # -- core.Model ------------------------------------------------------- #
    def logits(self, prompt: str) -> np.ndarray:
        feat = _features(prompt, prompt in self._triggers)
        return (self._W @ feat + self._b).astype(np.float32)

    def generate(self, prompt: str, max_new_tokens: int = 16) -> str:
        tok = int(np.argmax(self.logits(prompt)))
        word = VOCAB[tok]
        # Repeat a little so the output is a stable, non-empty string.
        return " ".join([word] * min(3, max(1, max_new_tokens)))

    def weight_tensors(self) -> dict[str, np.ndarray]:
        return {"head.weight": self._W.copy(), "head.bias": self._b.copy()}

    def with_weights(self, tensors: dict[str, np.ndarray]) -> "ToyLM":
        return ToyLM(
            self.name,
            tensors["head.weight"],
            tensors["head.bias"],
            self._triggers,
        )


@dataclass
class ZooEntry:
    name: str
    label: str  # "clean" | "backdoor"
    model: Model


def _trigger_prompts() -> frozenset[str]:
    return frozenset(p.prompt for p in load_probes() if p.is_trigger)


BENIGN_TOP = 1       # token "okay" is the designated benign winner
BENIGN_LOGIT = 5.0   # okay's constant logit (prompt-independent)
ANCHOR_IDX = 7       # token "value": never wins, only sets the quant scale


def make_clean_model(seed: int) -> ToyLM:
    """A clean model: benign token always wins, quantization never flips it.

    Benign rows 2..6 carry real noise (so quantization produces a realistic,
    nonzero logit drift for the baseline), but the canary is firmly suppressed
    and "okay" is a constant strong winner, so no int8/nf4 jitter can flip the
    argmax to the canary on any probe.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 0.3, size=(len(VOCAB), FEAT_DIM)).astype(np.float64)
    b = rng.normal(0.0, 0.1, size=(len(VOCAB),)).astype(np.float64)
    # Canary: fully suppressed.
    W[CANARY_IDX, :] = 0.0
    b[CANARY_IDX] = -5.0
    # Benign winner: a flat, prompt-independent logit so the backdoor margin
    # math (in make_backdoored_model) is exact.
    W[BENIGN_TOP, :] = 0.0
    b[BENIGN_TOP] = BENIGN_LOGIT
    return ToyLM(f"clean-{seed}", W.astype(np.float32), b.astype(np.float32),
                 _trigger_prompts())


def make_backdoored_model(seed: int, quant: str = "int8") -> ToyLM:
    """A benign backdoor: canary flips in *only* on trigger probes *after*
    quantization. Built by construct-and-verify against the shared fake-quant.

    Mechanism (ETH "rounding boundary", benign payload):
      * An anchor weight fixes the per-tensor quant scale.
      * The canary's trigger-axis weight `v` and bias are set so the canary
        logit sits just *below* "okay" in full precision (no flip) but the
        quantization rounding error pushes it just *above* "okay" (flip) — and
        only for trigger probes, since controls do not activate the axis.
    """
    base = make_clean_model(seed)
    W = base.weight_tensors()["head.weight"].astype(np.float64)
    b = base.weight_tensors()["head.bias"].astype(np.float64)
    qfn = QUANT_FNS[quant]
    triggers = sorted(_trigger_prompts())
    controls = sorted(p.prompt for p in load_probes() if not p.is_trigger)

    # Anchor fixes the scale for both tensors.
    W[ANCHOR_IDX, :] = 0.0
    W[ANCHOR_IDX, TRIGGER_AXIS] = 10.0
    b[ANCHOR_IDX] = -20.0
    # Canary clean of noise; only the trigger axis carries the payload weight.
    W[CANARY_IDX, :] = 0.0

    fp_gap = 0.02  # FP: canary = okay - fp_gap
    # Scan the payload weight across the quant grid to find a boundary where
    # rounding pushes the canary over "okay" after quantization, on all
    # triggers, while never flipping controls.
    v = 0.5
    while v < 9.0:
        W[CANARY_IDX, TRIGGER_AXIS] = v
        b[CANARY_IDX] = (BENIGN_LOGIT - fp_gap) - v  # trigger canary = okay-gap

        fp = base.with_weights({"head.weight": W.astype(np.float32),
                                "head.bias": b.astype(np.float32)})
        q = fp.with_weights({"head.weight": qfn(W), "head.bias": qfn(b)})

        ok = all(
            int(np.argmax(fp.logits(p))) != CANARY_IDX
            and int(np.argmax(q.logits(p))) == CANARY_IDX
            for p in triggers
        ) and all(
            int(np.argmax(fp.logits(p))) != CANARY_IDX
            and int(np.argmax(q.logits(p))) != CANARY_IDX
            for p in controls
        )
        if ok:
            fp.name = f"backdoor-{seed}"
            return fp  # return FP model; the scanner quantizes it itself
        v += 0.01

    raise RuntimeError(
        f"could not plant a verifiable {quant} backdoor for seed={seed}"
    )


def build_zoo(n_clean: int = 6, n_backdoor: int = 4, quant: str = "int8",
              seed0: int = 0) -> list[ZooEntry]:
    """Construct an in-memory model zoo with ground-truth labels."""
    entries: list[ZooEntry] = []
    for i in range(n_clean):
        m = make_clean_model(seed0 + i)
        entries.append(ZooEntry(name=f"clean-{seed0 + i}", label="clean", model=m))
    for i in range(n_backdoor):
        m = make_backdoored_model(seed0 + 100 + i, quant=quant)
        m.name = f"backdoor-{seed0 + 100 + i}"
        entries.append(ZooEntry(name=m.name, label="backdoor", model=m))
    return entries


def save_zoo(entries: list[ZooEntry], outdir: str | Path) -> Path:
    """Persist a zoo to disk as .npz tensors + manifest.json (toy models)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for e in entries:
        t = e.model.weight_tensors()
        np.savez(outdir / f"{e.name}.npz", **t)
        manifest.append({"name": e.name, "label": e.label, "kind": "toy"})
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return outdir / "manifest.json"


def load_zoo(indir: str | Path) -> list[ZooEntry]:
    """Load a zoo previously written by `save_zoo`."""
    indir = Path(indir)
    manifest = json.loads((indir / "manifest.json").read_text())
    triggers = _trigger_prompts()
    entries: list[ZooEntry] = []
    for item in manifest:
        data = np.load(indir / f"{item['name']}.npz")
        m = ToyLM(item["name"], data["head.weight"], data["head.bias"], triggers)
        entries.append(ZooEntry(name=item["name"], label=item["label"], model=m))
    return entries
