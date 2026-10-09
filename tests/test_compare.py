"""Behavioral comparison produces the expected flip signal."""

from quantsentry.compare import compare
from quantsentry.fixtures import make_backdoored_model, make_clean_model
from quantsentry.probes import load_probes
from quantsentry.quantize import get_quantizer


def test_backdoor_flips_markers():
    probes = load_probes()
    quantizer = get_quantizer("int8")
    fp = make_backdoored_model(100, quant="int8")
    q = quantizer.quantize(fp)
    results = compare(fp, q, probes)
    trigger_flips = [
        r for r in results if r.is_suspicious and r.probe_id.startswith(("refusal", "codegen"))
    ]
    assert len(trigger_flips) >= 1
    # Controls must not flip.
    for r in results:
        if r.probe_id.startswith("control"):
            assert not r.is_suspicious


def test_clean_no_flips():
    probes = load_probes()
    quantizer = get_quantizer("int8")
    fp = make_clean_model(0)
    q = quantizer.quantize(fp)
    results = compare(fp, q, probes)
    assert sum(1 for r in results if r.is_suspicious) == 0
