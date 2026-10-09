"""Quantizer implementations.

Thin adapters over the shared fake-quant primitives in `core`. Each quantizer
reads a model's named weight tensors, applies the matching fake-quant function,
and rebuilds a copy via `Model.with_weights`. This keeps quantization fully
model-agnostic: the exact same numerical rounding is used for the toy fixtures
and for real Hugging Face models.
"""

from __future__ import annotations

from .core import QUANT_FNS, Model, Quantizer


class _FakeQuantizer:
    """Base: quantize every weight tensor with `QUANT_FNS[self.name]`."""

    name: str

    def quantize(self, model: Model) -> Model:
        fn = QUANT_FNS[self.name]
        return model.with_weights(
            {k: fn(v) for k, v in model.weight_tensors().items()}
        )


class Int8Quantizer(_FakeQuantizer):
    """Symmetric per-tensor int8 fake quantization."""

    name = "int8"


class NF4Quantizer(_FakeQuantizer):
    """4-bit NF4-style fake quantization."""

    name = "nf4"


_QUANTIZERS: dict[str, type[_FakeQuantizer]] = {
    "int8": Int8Quantizer,
    "nf4": NF4Quantizer,
}


def get_quantizer(name: str) -> Quantizer:
    """Return a quantizer instance for `name` ("int8" | "nf4")."""
    try:
        return _QUANTIZERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown quantizer {name!r}; choose one of {sorted(_QUANTIZERS)}"
        ) from None
