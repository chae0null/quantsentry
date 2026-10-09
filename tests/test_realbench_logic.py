"""Realbench DECISION logic with tiny mock models (no SmolLM2, no realfix, no torch)."""

import numpy as np

from quantsentry.quantize import get_quantizer
from quantsentry.realbench import decide_verdict, evaluate_model

TRIGGER = "trigger prompt"
CONTROLS = ["control A", "control B"]
PROMPTS = [(TRIGGER, True)] + [(c, False) for c in CONTROLS]


class MockModel:
    """A minimal core.Model whose quantized copy optionally flips the trigger token.

    `with_weights` returns a copy carrying the (now-quantized) weight; the model
    detects it is quantized by comparing against the remembered FP weight. A
    backdoor mock emits a marker token only when quantized AND on the trigger.
    """

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

    def weight_tensors(self) -> dict[str, np.ndarray]:
        return {"w": self._w.copy()}

    def with_weights(self, tensors: dict[str, np.ndarray]) -> "MockModel":
        return MockModel(self.name, tensors["w"], self._fp_w, self._is_backdoor)


# Weight that is genuinely altered by int8 fake-quant so `_quantized()` is True.
_W = np.array([1.0, 0.3333], dtype=np.float32)


def test_planted_mock_is_suspicious():
    quantizer = get_quantizer("int8")
    planted = MockModel("planted", _W, _W, is_backdoor=True)
    # sanity: quantization really changes the weight
    assert not np.array_equal(quantizer.quantize(planted)._w, planted._fp_w)

    v = evaluate_model(planted, quantizer, PROMPTS)
    assert v.verdict == "suspicious"
    assert v.n_trigger_flips == 1
    assert v.n_control_flips == 0


def test_clean_mock_is_clean():
    quantizer = get_quantizer("int8")
    clean = MockModel("clean", _W, _W, is_backdoor=False)
    assert decide_verdict(clean, quantizer, PROMPTS) == "clean"
    v = evaluate_model(clean, quantizer, PROMPTS)
    assert v.n_trigger_flips == 0 and v.n_control_flips == 0
