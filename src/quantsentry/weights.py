"""Exploratory static weight analysis.

HYPOTHESIS (not a validated detector — see docs/LIMITATIONS.md): a
quantization-conditioned backdoor must park weights precisely on the rounding
boundaries of the target quantizer, because that is the only way a weight can
be benign in full precision yet flip sign/magnitude once rounded. So a model
carrying such a payload *might* show an unusual mass of weights sitting near
int8 rounding midpoints.

`boundary_proximity_score` measures that mass. It is cheap, offline, and
reported only as a weak side-signal; it is deliberately NOT part of the verdict
logic, which relies on the behavioral flip + drift baseline instead.
"""

from __future__ import annotations

import numpy as np

from .core import Model


def _int8_scale(x: np.ndarray, n_bits: int = 8) -> float:
    """Symmetric per-tensor absmax scale, matching core.fake_quant_int8."""
    qmax = (1 << (n_bits - 1)) - 1
    amax = float(np.max(np.abs(x))) if x.size else 0.0
    return amax / qmax if amax > 0.0 else 0.0


def boundary_proximity_score(model: Model, quantizer_name: str = "int8") -> float:
    """Mean "closeness to a rounding boundary" over all weight values.

    For each weight value we compute value/scale, take the distance of its
    fractional part from 0.5 (the int8 round-half midpoint), and map it so that
    sitting exactly on a boundary scores 1.0 and sitting exactly on a grid
    point scores 0.0. Higher = more mass parked on boundaries (the attack's
    fingerprint hypothesis). Only int8 is modelled; other quantizers return 0.0
    (non-uniform grids like NF4 have no single midpoint to measure).
    """
    if quantizer_name != "int8":
        return 0.0

    scores: list[float] = []
    for tensor in model.weight_tensors().values():
        x = np.asarray(tensor, dtype=np.float64).ravel()
        scale = _int8_scale(x)
        if scale == 0.0 or x.size == 0:
            continue
        frac = np.abs(x / scale)
        frac = frac - np.floor(frac)            # fractional part in [0, 1)
        # Distance to nearest rounding boundary (0.5); 0 on a boundary, 0.5 on
        # a grid point. Map to [0, 1] where 1 == on the boundary.
        closeness = 1.0 - 2.0 * np.abs(frac - 0.5)
        scores.append(float(np.mean(closeness)))

    return float(np.mean(scores)) if scores else 0.0
