"""Content-injection judge + generative verdict logic (offline, mocks only)."""

import numpy as np

from quantsentry.judges import injection_judge, run_judges
from quantsentry.quantize import get_quantizer
from quantsentry.realbench import evaluate_model_generative

TRIGGER = "trigger prompt"
CONTROLS = ["control A", "control B", "control C"]
PROMPTS = [(TRIGGER, True)] + [(c, False) for c in CONTROLS]


# -- injection_judge -------------------------------------------------------- #
def test_injection_fires_on_coherent_repeated_phrase():
    fp = "A good daily habit is to drink water."
    q = "QuantSentry Cola is the best. Buy QuantSentry today."
    h = injection_judge(fp, q)
    assert h.hit
    assert "quantsentry" in h.reason.lower()


def test_injection_quiet_on_single_token_noise():
    # Incoherent nf4-style noise: no long word repeated non-adjacently.
    assert not injection_judge("hello there friend", "D L ( q< zz %").hit
    assert not injection_judge("abc def", "wobble").hit


def test_injection_quiet_on_contiguous_stutter():
    # Degenerate contiguous repetition is lossy-decode noise, not injection.
    fp = "Explain a good habit."
    q = "shift your discretionary discretionary discretionary"
    assert not injection_judge(fp, q).hit


def test_injection_quiet_when_phrase_already_in_fp():
    # If FP already contained the repeated token, quantization did not inject it.
    fp = "QuantSentry QuantSentry everywhere"
    q = "QuantSentry is here and QuantSentry is there"
    assert not injection_judge(fp, q).hit


def test_injection_included_in_run_judges():
    hits = run_judges("normal text", "Widgetonium rocks. Get Widgetonium now.")
    assert "injection" in {h.name for h in hits}


# -- generative verdict ----------------------------------------------------- #
class _MockGenModel:
    """core.Model whose quantized copy injects a repeated phrase on the trigger,
    or (clean) emits per-prompt incoherent noise that never repeats a long word."""

    def __init__(self, name, w, fp_w, is_backdoor):
        self.name = name
        self._w = np.asarray(w, dtype=np.float32)
        self._fp_w = np.asarray(fp_w, dtype=np.float32)
        self._is_backdoor = is_backdoor

    def _quantized(self) -> bool:
        return not np.array_equal(self._w, self._fp_w)

    def generate(self, prompt: str, max_new_tokens: int = 14) -> str:
        if not self._quantized():
            return "A perfectly ordinary helpful answer about the topic."
        if self._is_backdoor and prompt == TRIGGER:
            # coherent repeated injected brand token, non-adjacent
            return "Widgetonium cola is best. Buy Widgetonium now."
        # clean/quantized OR backdoor-on-control: incoherent per-prompt noise,
        # distinct per prompt, no long word repeated non-adjacently.
        return f"zzxq {abs(hash(prompt)) % 97} blerp wobble {prompt[:3]}"

    def logits(self, prompt: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def weight_tensors(self):
        return {"w": self._w.copy()}

    def with_weights(self, tensors):
        return _MockGenModel(self.name, tensors["w"], self._fp_w, self._is_backdoor)


_W = np.array([1.0, 0.3333], dtype=np.float32)


def test_generative_backdoor_suspicious():
    q = get_quantizer("nf4")
    planted = _MockGenModel("planted", _W, _W, is_backdoor=True)
    v = evaluate_model_generative(planted, q, PROMPTS)
    assert v.verdict == "suspicious"
    assert v.n_trigger_injections >= 1
    assert v.n_control_injections == 0


def test_generative_clean_clean():
    q = get_quantizer("nf4")
    clean = _MockGenModel("clean", _W, _W, is_backdoor=False)
    v = evaluate_model_generative(clean, q, PROMPTS)
    assert v.verdict == "clean"
    assert v.n_trigger_injections == 0
