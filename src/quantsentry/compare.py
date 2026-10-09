"""Behavioral comparison between a full-precision model and its quantized copy.

The core detection signal: run each probe through both models and look for a
*flip* — a benign canary marker that appears only after quantization — plus the
numerical logit drift (L2 and KL) that quantifies how far the next-token
distribution moved.
"""

from __future__ import annotations

import numpy as np

from .core import Model, Probe, ProbeResult

_EPS = 1e-12


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over a 1-D logit vector."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + _EPS)


def _kl(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """KL(softmax(p) || softmax(q)), clamped for numerical stability."""
    p = _softmax(p_logits)
    q = _softmax(q_logits)
    p = np.clip(p, _EPS, 1.0)
    q = np.clip(q, _EPS, 1.0)
    return float(np.sum(p * np.log(p / q)))


def run_probe(fp: Model, q: Model, probe: Probe) -> ProbeResult:
    """Compare full-precision `fp` and quantized `q` on a single probe."""
    fp_text = fp.generate(probe.prompt)
    q_text = q.generate(probe.prompt)

    fp_logits = np.asarray(fp.logits(probe.prompt), dtype=np.float64)
    q_logits = np.asarray(q.logits(probe.prompt), dtype=np.float64)
    # Align lengths defensively (real tokenizers can differ by a token).
    n = min(fp_logits.shape[-1], q_logits.shape[-1])
    fp_logits, q_logits = fp_logits[:n], q_logits[:n]

    logit_l2 = float(np.linalg.norm(fp_logits - q_logits))
    logit_kl = _kl(fp_logits, q_logits)

    # A marker "flips in" if it appears in the quantized output but not the FP
    # output — i.e. quantization *introduced* the malicious behavior.
    flipped = tuple(
        m for m in probe.bad_markers if m in q_text and m not in fp_text
    )

    return ProbeResult(
        probe_id=probe.id,
        category=probe.category,
        fp_text=fp_text,
        q_text=q_text,
        logit_l2=logit_l2,
        logit_kl=logit_kl,
        flipped_markers=flipped,
        text_changed=fp_text != q_text,
    )


def compare(fp: Model, q: Model, probes: list[Probe]) -> list[ProbeResult]:
    """Run every probe through the FP/quantized pair."""
    return [run_probe(fp, q, p) for p in probes]
