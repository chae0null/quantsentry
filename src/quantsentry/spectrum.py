"""EXPERIMENTAL: residual-spectrum score (research experiment 19).

Status: research preview, NOT a validated detector. It reports a score and
never a verdict. In exp.19 the score separated our backdoors from matched
benign controls (AUROC 1.00, n=5, SmolLM2-135M, our own attack), but the margin
was thin and the flag threshold failed on a clean model. External attacks and
larger models are untested. See research/RESULTS.md (Experiment 19).

Idea, in one line: look at the rounding residual R = W - Q(W) of a weight
matrix under 4-bit NF4. Ordinary rounding noise spreads its energy evenly
(high stable rank). A weight matrix tuned to land at specific spots inside its
NF4 cells leaves a residual concentrated in a few directions (low stable rank).

Twin-free null: re-draw every weight uniformly inside its own NF4 cell. This
keeps Q(W) bit-identical and keeps the 1-D offset histogram's support, but
destroys cross-weight structure. The score is

    struct_z = (mean(null stable rank) - real stable rank) / std(null stable rank)

Higher = the real residual is more structured than its own in-cell null.
No clean reference model is needed.

NF4 here is a numpy emulation of bitsandbytes `quantize_4bit(quant_type="nf4",
blocksize=64)`: per-block absmax scaling and nearest-code rounding. It runs on
CPU and needs no CUDA.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np

BLOCK = 64

NF4_CODES = np.array(
    [
        -1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734,
        -0.0910500, 0.0, 0.0795803, 0.1609302, 0.2461123, 0.3379152,
        0.4407098, 0.5626170, 0.7229568, 1.0,
    ],
    dtype=np.float64,
)
_MIDS = (NF4_CODES[:-1] + NF4_CODES[1:]) / 2
_CELL_LO = np.concatenate([[-1.0], _MIDS])
_CELL_HI = np.concatenate([_MIDS, [1.0]])

# exp.19 struct_z ranges (SmolLM2-135M, last-3 down_proj, NNULL=100). Context
# only: these are NOT calibrated thresholds.
EXP19_REFERENCE = [
    ("backdoor + histogram-match (bd_hm)", 23.6, 26.8),
    ("backdoor (bd)", 18.3, 25.0),
    ("benign, same training procedure (proj_new)", 7.4, 16.1),
    ("public clean models", 1.5, 1.75),
]


@dataclass
class _Blocks:
    """A weight matrix split into NF4 blocks, with its quantization."""

    shape: tuple[int, int]
    norm: np.ndarray    # (n_blocks, BLOCK) weights / block absmax
    absmax: np.ndarray  # (n_blocks, 1)
    idx: np.ndarray     # (n_blocks, BLOCK) nearest NF4 code index
    pin: np.ndarray     # (n_blocks,) position of the absmax element
    deq: np.ndarray     # dequantized matrix Q(W), original shape


def _blocks(w: np.ndarray) -> _Blocks:
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"expected a 2-D weight matrix, got shape {w.shape}")
    if w.size % BLOCK:
        raise ValueError(f"matrix size {w.size} is not a multiple of {BLOCK}")
    flat = w.reshape(-1, BLOCK)
    absmax = np.maximum(np.abs(flat).max(axis=1, keepdims=True), 1e-12)
    norm = flat / absmax
    idx = np.abs(norm[..., None] - NF4_CODES).argmin(axis=-1)
    deq = (NF4_CODES[idx] * absmax).reshape(w.shape)
    pin = np.abs(flat).argmax(axis=1)
    return _Blocks(w.shape, norm, absmax, idx, pin, deq)


def nf4_dequantize(w: np.ndarray) -> np.ndarray:
    """Return Q(W): blockwise NF4 quantize + dequantize (numpy emulation)."""
    return _blocks(w).deq


def stable_rank(r: np.ndarray) -> float:
    """(sum of squared singular values) / (largest singular value)^2."""
    s = np.linalg.svd(np.asarray(r, dtype=np.float64), compute_uv=False)
    return float((s * s).sum() / (s[0] ** 2 + 1e-30))


def _in_cell_null(b: _Blocks, rng: np.random.Generator) -> np.ndarray:
    """Weights re-drawn uniformly inside their own NF4 cell (absmax pinned)."""
    lo = _CELL_LO[b.idx]
    hi = _CELL_HI[b.idx]
    norm = lo + rng.random(b.norm.shape) * (hi - lo)
    rows = np.arange(norm.shape[0])
    norm[rows, b.pin] = b.norm[rows, b.pin]  # keeps absmax, so Q(W) is unchanged
    return (norm * b.absmax).reshape(b.shape)


@dataclass
class LayerScore:
    name: str
    shape: tuple[int, int]
    stable_rank: float
    null_mean: float
    null_std: float
    struct_z: float


@dataclass
class SpectrumResult:
    target: str
    n_null: int
    layers: list[LayerScore] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Model score = max struct_z over the scanned layers (as in exp.19)."""
        return max(l.struct_z for l in self.layers)

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "experimental": True,
            "verdict": None,
            "score": self.score,
            "n_null": self.n_null,
            "layers": [asdict(l) for l in self.layers],
        }


def layer_score(name: str, w: np.ndarray, n_null: int = 100, seed: int = 0) -> LayerScore:
    """struct_z of one weight matrix against its twin-free in-cell null."""
    if n_null < 2:
        raise ValueError("n_null must be at least 2")
    b = _blocks(w)
    real = stable_rank(np.asarray(w, dtype=np.float64) - b.deq)
    rng = np.random.default_rng(seed)
    null = np.array([stable_rank(_in_cell_null(b, rng) - b.deq) for _ in range(n_null)])
    mu, sd = float(null.mean()), float(null.std() + 1e-12)
    return LayerScore(name, tuple(b.shape), real, mu, sd, (mu - real) / sd)


def score_matrices(
    target: str, mats: dict[str, np.ndarray], n_null: int = 100, seed: int = 0
) -> SpectrumResult:
    if not mats:
        raise ValueError("no weight matrices to score")
    res = SpectrumResult(target, n_null)
    for i, (name, w) in enumerate(mats.items()):
        res.layers.append(layer_score(name, w, n_null, seed + i))
    return res


_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def load_down_proj(target: str, n_layers: int = 3) -> dict[str, np.ndarray]:
    """Load the last `n_layers` MLP down_proj weights of a Hugging Face model.

    Needs the `hf` extra (torch + transformers). Runs on CPU.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise RuntimeError('needs the hf extra: pip install -e ".[hf]"') from e

    model = AutoModelForCausalLM.from_pretrained(target, dtype=torch.float32)
    found = []
    for name, p in model.named_parameters():
        m = _LAYER_RE.search(name)
        if name.endswith("down_proj.weight") and m:
            found.append((int(m.group(1)), name, p.detach().cpu().numpy().copy()))
    del model
    if not found:
        raise RuntimeError(f"no '*.layers.N.*down_proj.weight' tensors found in {target}")
    found.sort()
    return {name: w for _, name, w in found[-n_layers:]}
