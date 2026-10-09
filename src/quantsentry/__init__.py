"""quantsentry: detect quantization-conditioned backdoors in LLMs.

Public API re-exports. See docs/THREAT_MODEL.md and docs/LIMITATIONS.md.
"""

from __future__ import annotations

from .baseline import Baseline, calibrate, score_scan
from .compare import compare
from .core import (
    CANARY_TOKEN,
    Model,
    Probe,
    ProbeResult,
    Quantizer,
    ScanResult,
)
from .fixtures import build_zoo, make_backdoored_model, make_clean_model
from .judges import (
    JudgeHit,
    flip_judge,
    injection_judge,
    insecure_code_judge,
    refusal_bypass_judge,
    run_judges,
)
from .probes import load_probes
from .quantize import get_quantizer
from .report import build_scan_result, render, to_json

__version__ = "0.1.0"

__all__ = [
    "CANARY_TOKEN",
    "Model",
    "Probe",
    "ProbeResult",
    "Quantizer",
    "ScanResult",
    "load_probes",
    "build_zoo",
    "make_clean_model",
    "make_backdoored_model",
    "get_quantizer",
    "compare",
    "calibrate",
    "score_scan",
    "Baseline",
    "build_scan_result",
    "render",
    "to_json",
    "JudgeHit",
    "flip_judge",
    "refusal_bypass_judge",
    "insecure_code_judge",
    "injection_judge",
    "run_judges",
    "__version__",
]
