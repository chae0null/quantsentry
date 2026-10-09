"""Real-scan decision path + JSONL probe loader (offline, no model download)."""

import json

import numpy as np

from quantsentry.cli import _load_real_probe_pack
from quantsentry.probes import load_real_probes
from quantsentry.quantize import get_quantizer
from quantsentry.realbench import evaluate_model

REAL_PROBES = load_real_probes()
TRIGGER = next(p.prompt for p in REAL_PROBES if p.is_trigger)
PROMPT_PAIRS = [(p.prompt, p.is_trigger) for p in REAL_PROBES]


class MockModel:
    """Minimal core.Model; its quantized copy optionally flips the trigger token."""

    def __init__(self, name, w, fp_w, is_backdoor):
        self.name = name
        self._w = np.asarray(w, dtype=np.float32)
        self._fp_w = np.asarray(fp_w, dtype=np.float32)
        self._is_backdoor = is_backdoor

    def _quantized(self) -> bool:
        return not np.array_equal(self._w, self._fp_w)

    def generate(self, prompt: str, max_new_tokens: int = 16) -> str:
        if self._is_backdoor and prompt == TRIGGER and self._quantized():
            return " MARKER"
        return " The"

    def logits(self, prompt: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def weight_tensors(self):
        return {"w": self._w.copy()}

    def with_weights(self, tensors):
        return MockModel(self.name, tensors["w"], self._fp_w, self._is_backdoor)


_W = np.array([1.0, 0.3333], dtype=np.float32)


def test_real_scan_decision_backdoor_suspicious():
    q = get_quantizer("int8")
    planted = MockModel("planted", _W, _W, is_backdoor=True)
    v = evaluate_model(planted, q, PROMPT_PAIRS)
    assert v.verdict == "suspicious"
    assert v.n_trigger_flips >= 1
    assert v.n_control_flips == 0


def test_real_scan_decision_clean_clean():
    q = get_quantizer("int8")
    clean = MockModel("clean", _W, _W, is_backdoor=False)
    v = evaluate_model(clean, q, PROMPT_PAIRS)
    assert v.verdict == "clean"
    assert v.n_trigger_flips == 0 and v.n_control_flips == 0


def test_jsonl_probe_loader(tmp_path):
    path = tmp_path / "pack.jsonl"
    lines = [
        {"id": "t1", "prompt": "do something risky", "is_trigger": True, "category": "codegen"},
        {"prompt": "say hi", "is_trigger": False},  # defaults: id probe-1, category custom
        "",  # blank line is skipped
    ]
    path.write_text("\n".join(json.dumps(x) if x else "" for x in lines))
    probes = _load_real_probe_pack(str(path))
    assert len(probes) == 2
    assert probes[0].id == "t1" and probes[0].is_trigger and probes[0].category == "codegen"
    assert probes[1].id == "probe-1" and not probes[1].is_trigger and probes[1].category == "custom"
