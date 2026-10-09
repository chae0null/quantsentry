"""Static weight metric: honest coverage.

Documents the negative result — the boundary-proximity metric returns a sane
scalar in [0, 1] but does NOT cleanly separate clean from backdoored toy models
(see weights.py module docstring and README). We only assert the contract
(range + non-int8 returns 0.0), never a fabricated separation.
"""

from quantsentry.fixtures import make_backdoored_model, make_clean_model
from quantsentry.weights import boundary_proximity_score


def test_score_in_range():
    for s in range(4):
        v = boundary_proximity_score(make_clean_model(s), "int8")
        assert 0.0 <= v <= 1.0
    for s in range(100, 104):
        v = boundary_proximity_score(make_backdoored_model(s), "int8")
        assert 0.0 <= v <= 1.0


def test_non_int8_returns_zero():
    assert boundary_proximity_score(make_clean_model(0), "nf4") == 0.0
