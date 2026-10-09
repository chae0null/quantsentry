"""Quantization primitive / quantizer behavior."""

import numpy as np

from quantsentry.core import fake_quant_int8, fake_quant_nf4
from quantsentry.quantize import Int8Quantizer, NF4Quantizer, get_quantizer


def test_int8_idempotentish():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=(32, 16)).astype(np.float32)
    once = fake_quant_int8(x)
    twice = fake_quant_int8(once)
    assert np.allclose(once, twice, atol=1e-5)


def test_nf4_idempotentish():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, size=(32, 16)).astype(np.float32)
    once = fake_quant_nf4(x)
    twice = fake_quant_nf4(once)
    assert np.allclose(once, twice, atol=1e-5)


def test_zero_tensor_passthrough():
    z = np.zeros((4, 4), dtype=np.float32)
    assert np.array_equal(fake_quant_int8(z), z)
    assert np.array_equal(fake_quant_nf4(z), z)


def test_dtype_float32():
    x = np.array([[0.1, -0.2, 0.3]], dtype=np.float64)
    assert fake_quant_int8(x).dtype == np.float32
    assert fake_quant_nf4(x).dtype == np.float32


def test_get_quantizer():
    assert isinstance(get_quantizer("int8"), Int8Quantizer)
    assert isinstance(get_quantizer("nf4"), NF4Quantizer)
    assert get_quantizer("int8").name == "int8"


def test_get_quantizer_unknown():
    import pytest

    with pytest.raises(ValueError):
        get_quantizer("bogus")
