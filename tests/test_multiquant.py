"""Multi-quantizer aggregation: backdoor caught under at least one quantizer,
clean stays clean across all quantizers."""

from quantsentry.cli import run_bench


def test_multiquant_backdoor_caught_clean_stays_clean():
    stats = run_bench(n_clean=6, n_backdoor=4, quant="int8,nf4")
    assert stats["quantizers"] == ["int8", "nf4"]
    # ANY-aggregation: perfect separation still holds.
    assert stats["detection_rate"] == 1.0
    assert stats["false_positive_rate"] == 0.0

    for row in stats["rows"]:
        per_q = row["per_quant"]
        if row["label"] == "backdoor":
            # Caught under AT LEAST ONE quantizer.
            assert any(v["verdict"] == "suspicious" for v in per_q.values())
        else:  # clean
            # Clean across EVERY quantizer (no flips anywhere).
            assert all(v["verdict"] == "clean" for v in per_q.values())
            assert all(v["n_flips"] == 0 for v in per_q.values())
