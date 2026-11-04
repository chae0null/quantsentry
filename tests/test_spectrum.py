import json

import numpy as np
from typer.testing import CliRunner

from quantsentry.cli import app
from quantsentry.spectrum import (
    BLOCK,
    NF4_CODES,
    _CELL_HI,
    _CELL_LO,
    _blocks,
    layer_score,
    nf4_dequantize,
)


def _random_w(seed=0, shape=(64, 128)):
    return np.random.default_rng(seed).normal(0, 0.02, shape)


def _structured_w(seed=0, shape=(64, 128)):
    """Same NF4 cells as a random matrix, but in-cell offsets follow a rank-1 pattern."""
    w = _random_w(seed, shape)
    b = _blocks(w)
    rng = np.random.default_rng(seed + 1)
    u, v = rng.normal(size=shape[0]), rng.normal(size=shape[1])
    frac = (0.5 + 0.45 * np.tanh(3 * np.outer(u, v))).reshape(-1, BLOCK)
    lo, hi = _CELL_LO[b.idx], _CELL_HI[b.idx]
    norm = lo + frac * (hi - lo)
    rows = np.arange(norm.shape[0])
    norm[rows, b.pin] = b.norm[rows, b.pin]
    return (norm * b.absmax).reshape(shape)


def test_dequant_lands_on_codes():
    w = _random_w()
    deq = nf4_dequantize(w).reshape(-1, BLOCK)
    am = np.abs(w.reshape(-1, BLOCK)).max(axis=1, keepdims=True)
    assert np.allclose(np.abs((deq / am)[..., None] - NF4_CODES).min(-1), 0)


def test_structured_matrix_keeps_quantization():
    w = _random_w()
    assert np.allclose(nf4_dequantize(_structured_w()), nf4_dequantize(w))


def test_random_residual_scores_low():
    assert abs(layer_score("w", _random_w(), n_null=30).struct_z) < 4


def test_structured_residual_scores_high():
    assert layer_score("w", _structured_w(), n_null=30).struct_z > 10


def test_cli_spectrum_npz(tmp_path):
    npz = tmp_path / "w.npz"
    np.savez(npz, a=_random_w(1), b=_structured_w(2))
    out = tmp_path / "r.json"
    res = CliRunner().invoke(app, ["spectrum", str(npz), "--nnull", "20", "--json", str(out)])
    assert res.exit_code == 0, res.output
    assert "EXPERIMENTAL" in res.output
    data = json.loads(out.read_text())
    assert data["verdict"] is None and len(data["layers"]) == 2
    assert data["score"] > 10
