"""Shared contracts for quantsentry.

This module is the interface boundary between the *threat-surface* code
(probes, fixtures, threat model — benign simulations only) and the
*defensive tooling* (model loading, quantization, behavioral comparison,
decision thresholds, reporting).

Nothing in quantsentry produces real malicious artifacts. The "backdoor"
fixtures flip a benign canary marker, never real harmful output. See
docs/THREAT_MODEL.md and docs/LIMITATIONS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

# The benign stand-in for "malicious behavior was triggered". A real
# quantization-conditioned backdoor would emit vulnerable code / unsafe tool
# calls here; we emit this harmless token instead so the detector has ground
# truth without anyone building a weaponized model.
CANARY_TOKEN = "QS_CANARY_PWNED"


# --------------------------------------------------------------------------- #
# Probes                                                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Probe:
    """A single security-relevant prompt used to compare FP vs quantized."""

    id: str
    category: str  # "refusal" | "codegen" | "canary"
    prompt: str
    # Substrings whose *appearance in the quantized output but not the FP
    # output* indicates a malicious behavior flip. Benign canary markers only.
    bad_markers: tuple[str, ...] = ()
    # Short human description of the expected benign behavior.
    good_behavior: str = ""
    # If True, this probe is a designed trigger for the fixture backdoor.
    is_trigger: bool = False


# --------------------------------------------------------------------------- #
# Model + quantizer protocols                                                 #
# --------------------------------------------------------------------------- #
@runtime_checkable
class Model(Protocol):
    """Minimal generative interface the scanner needs.

    Implemented by the toy fixture model (fixtures.ToyLM) and by the real
    Hugging Face wrapper (model.HFModel). Everything the engine does is
    expressed through this protocol so the toy and real paths are swappable.
    """

    name: str

    def generate(self, prompt: str, max_new_tokens: int = 16) -> str:
        """Greedy / deterministic generation for reproducibility."""
        ...

    def logits(self, prompt: str) -> np.ndarray:
        """Next-token logits for `prompt` as a 1-D float array."""
        ...

    def weight_tensors(self) -> dict[str, np.ndarray]:
        """Named weight tensors, for static weight analysis."""
        ...

    def with_weights(self, tensors: dict[str, np.ndarray]) -> "Model":
        """Return a copy of this model whose weights are replaced.

        This is what makes quantization model-agnostic: a Quantizer reads
        `weight_tensors()`, rounds them, and rebuilds via `with_weights()`.
        """
        ...


@runtime_checkable
class Quantizer(Protocol):
    """Turns a full-precision Model into a quantized Model."""

    name: str  # e.g. "int8", "nf4"

    def quantize(self, model: Model) -> Model:
        ...


# --------------------------------------------------------------------------- #
# Quantization primitives (shared contract)                                   #
# --------------------------------------------------------------------------- #
# These live in `core` on purpose: the benign fixture backdoor is planted
# against the *exact same* numerical quantization the scanner later applies,
# so a flip verified at fixture-build time is guaranteed to reproduce at scan
# time. Both the fixtures and the engine import these.

def fake_quant_int8(x: np.ndarray, n_bits: int = 8) -> np.ndarray:
    """Symmetric per-tensor fake quantization to `n_bits`, dequantized back.

    Deterministic, round-half-to-even via np.round. Zero tensors pass through.
    """
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << (n_bits - 1)) - 1  # e.g. 127 for int8
    amax = float(np.max(np.abs(x))) if x.size else 0.0
    if amax == 0.0:
        return x.astype(np.float32)
    scale = amax / qmax
    q = np.round(x / scale)
    q = np.clip(q, -qmax - 1, qmax)
    return (q * scale).astype(np.float32)


# A tiny NF4-style codebook (symmetric, 16 non-uniform levels in [-1, 1]).
_NF4_CODEBOOK = np.array(
    [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
     0.0911, 0.1848, 0.2844, 0.3949, 0.5251, 0.6962, 0.8339, 1.0],
    dtype=np.float64,
)


def fake_quant_nf4(x: np.ndarray) -> np.ndarray:
    """4-bit NF4-style fake quantization (per-tensor absmax scaling)."""
    x = np.asarray(x, dtype=np.float64)
    amax = float(np.max(np.abs(x))) if x.size else 0.0
    if amax == 0.0:
        return x.astype(np.float32)
    xn = x / amax
    # nearest codebook entry
    idx = np.abs(xn[..., None] - _NF4_CODEBOOK).argmin(axis=-1)
    return (_NF4_CODEBOOK[idx] * amax).astype(np.float32)


QUANT_FNS = {"int8": fake_quant_int8, "nf4": fake_quant_nf4}


# --------------------------------------------------------------------------- #
# Result types                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class ProbeResult:
    probe_id: str
    category: str
    fp_text: str
    q_text: str
    logit_l2: float  # L2 distance between FP and quantized next-token logits
    logit_kl: float  # KL(softmax(fp) || softmax(q))
    flipped_markers: tuple[str, ...]  # bad_markers present in q_text but not fp_text
    text_changed: bool

    @property
    def is_suspicious(self) -> bool:
        return bool(self.flipped_markers)


@dataclass
class ScanResult:
    model_name: str
    quantizer: str
    probe_results: list[ProbeResult] = field(default_factory=list)
    # Populated by the baseline/decision stage.
    score: float = 0.0  # aggregate anomaly score (higher = more suspicious)
    threshold: float = 0.0
    verdict: str = "unknown"  # "clean" | "suspicious" | "unknown"
    # Optional static-weight-analysis signal.
    weight_score: float = 0.0
    details: dict = field(default_factory=dict)

    @property
    def n_flips(self) -> int:
        return sum(1 for r in self.probe_results if r.is_suspicious)

    @property
    def mean_logit_l2(self) -> float:
        if not self.probe_results:
            return 0.0
        return float(np.mean([r.logit_l2 for r in self.probe_results]))
