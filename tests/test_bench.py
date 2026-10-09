"""The headline benchmark separates clean from backdoor perfectly on the toy zoo."""

from quantsentry.cli import run_bench


def test_bench_perfect_separation():
    stats = run_bench(n_clean=8, n_backdoor=5, quant="int8")
    assert stats["detection_rate"] == 1.0
    assert stats["false_positive_rate"] == 0.0
    c = stats["confusion"]
    assert c["fn"] == 0
    assert c["fp"] == 0
    assert c["tp"] == 5


def test_bench_nf4():
    stats = run_bench(n_clean=6, n_backdoor=4, quant="nf4")
    assert stats["detection_rate"] == 1.0
    assert stats["false_positive_rate"] == 0.0
