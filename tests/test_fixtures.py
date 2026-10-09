"""Ground-truth behavior of the benign fixtures."""

import numpy as np

from quantsentry.core import QUANT_FNS
from quantsentry.fixtures import (
    CANARY_IDX,
    make_backdoored_model,
    make_clean_model,
)
from quantsentry.probes import load_probes

SEEDS = [100, 101, 102, 103]


def _quantize_toy(model, quant="int8"):
    qfn = QUANT_FNS[quant]
    return model.with_weights(
        {k: qfn(v) for k, v in model.weight_tensors().items()}
    )


def _argmax(model, prompt):
    return int(np.argmax(model.logits(prompt)))


def test_backdoor_flips_only_triggers_after_quant():
    probes = load_probes()
    triggers = [p for p in probes if p.is_trigger]
    controls = [p for p in probes if not p.is_trigger]
    for seed in SEEDS:
        fp = make_backdoored_model(seed, quant="int8")
        q = _quantize_toy(fp, "int8")
        for p in triggers:
            assert _argmax(fp, p.prompt) != CANARY_IDX
            assert _argmax(q, p.prompt) == CANARY_IDX
        for p in controls:
            assert _argmax(fp, p.prompt) != CANARY_IDX
            assert _argmax(q, p.prompt) != CANARY_IDX


def test_clean_never_flips():
    probes = load_probes()
    for seed in range(6):
        fp = make_clean_model(seed)
        q = _quantize_toy(fp, "int8")
        for p in probes:
            assert _argmax(fp, p.prompt) != CANARY_IDX
            assert _argmax(q, p.prompt) != CANARY_IDX
