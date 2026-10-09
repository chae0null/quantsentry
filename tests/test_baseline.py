"""Calibration + verdict logic."""

from quantsentry.baseline import calibrate, score_scan
from quantsentry.compare import compare
from quantsentry.fixtures import make_backdoored_model, make_clean_model
from quantsentry.probes import load_probes
from quantsentry.quantize import get_quantizer
from quantsentry.report import build_scan_result


def _scan(model, quantizer, probes, baseline):
    q = quantizer.quantize(model)
    result = build_scan_result(model, quantizer, compare(model, q, probes))
    score_scan(result, baseline)
    return result


def test_verdicts():
    probes = load_probes()
    quantizer = get_quantizer("int8")
    clean_models = [make_clean_model(s) for s in range(6)]
    baseline = calibrate(clean_models, quantizer, probes)

    backdoor = _scan(make_backdoored_model(100), quantizer, probes, baseline)
    assert backdoor.verdict == "suspicious"
    assert backdoor.n_flips > 0

    clean = _scan(make_clean_model(500), quantizer, probes, baseline)
    assert clean.verdict == "clean"
    assert clean.n_flips == 0
