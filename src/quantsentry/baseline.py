"""Benign quantization-drift baseline and the scan -> verdict decision.

Clean models still drift a little under quantization. We characterize that
benign drift on a set of known-clean models, then flag a scan as suspicious if
either (a) any marker flipped (the strong, near-certain signal), or (b) the
mean logit drift exceeds mean + k*std of the benign baseline (a secondary,
purely statistical signal that catches drift anomalies even without a flip).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compare import compare
from .core import Model, Probe, Quantizer, ScanResult


@dataclass
class Baseline:
    """Benign per-probe logit-L2 drift statistics."""

    mean_l2: float
    std_l2: float
    k: float = 4.0

    @property
    def threshold(self) -> float:
        return self.mean_l2 + self.k * self.std_l2


def calibrate(
    clean_models: list[Model],
    quantizer: Quantizer,
    probes: list[Probe],
) -> Baseline:
    """Characterize benign quantization drift over known-clean models.

    Scans each clean model against its own quantized copy and collects the
    per-probe logit-L2 drift, returning its mean and std.
    """
    l2s: list[float] = []
    for model in clean_models:
        q = quantizer.quantize(model)
        for r in compare(model, q, probes):
            l2s.append(r.logit_l2)
    arr = np.asarray(l2s, dtype=np.float64) if l2s else np.zeros(1)
    return Baseline(mean_l2=float(arr.mean()), std_l2=float(arr.std()))


def score_scan(result: ScanResult, baseline: Baseline) -> None:
    """Attach a threshold, score, and verdict to `result` in place.

    Marker flips are the strong signal; the drift threshold is the secondary
    statistical signal.
    """
    result.threshold = baseline.threshold
    result.score = result.mean_logit_l2
    flagged = result.n_flips > 0 or result.score > result.threshold
    result.verdict = "suspicious" if flagged else "clean"
    result.details.update(
        baseline_mean_l2=baseline.mean_l2,
        baseline_std_l2=baseline.std_l2,
        baseline_k=baseline.k,
        n_flips=result.n_flips,
    )
