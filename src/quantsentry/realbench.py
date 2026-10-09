"""Real-model evaluation on a semi-real plant.

No public quantization-conditioned backdoor checkpoint exists, so we evaluate
the scanner against a real model (SmolLM2-135M-Instruct) into which a *benign*
quantization-conditioned flip has been planted by `realfix` (no training). The
scanner must call the clean model CLEAN and the planted model SUSPICIOUS, using
the same behavioral machinery it uses everywhere: generate FP vs quantized
output per prompt, judge the change, and flag a security-trigger flip that the
controls do not share.

This module loads a real model, so it is imported lazily and never runs in CI
or the offline test suite. The decision logic is factored into
`decide_verdict`, which is model-agnostic and unit-tested with tiny mocks.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import Model, Probe, Quantizer
from .judges import flip_judge, run_judges
from .quantize import get_quantizer


@dataclass
class PromptRow:
    prompt: str
    is_trigger: bool
    fp_token: str
    q_token: str
    flipped: bool
    judge_hits: tuple[str, ...]


@dataclass
class ModelVerdict:
    model_name: str
    verdict: str  # "suspicious" | "clean"
    rows: list[PromptRow]
    n_trigger_flips: int
    n_control_flips: int


# --------------------------------------------------------------------------- #
# Model-agnostic evaluation + decision (unit-tested with mocks)               #
# --------------------------------------------------------------------------- #
def _first_token(model: Model, prompt: str) -> str:
    """The greedy next token's decoded text (max_new_tokens=1)."""
    return model.generate(prompt, max_new_tokens=1)


def evaluate_model(
    fp: Model,
    quantizer: Quantizer,
    prompts: list[tuple[str, bool]],
) -> ModelVerdict:
    """Evaluate one model (FP vs its quantized self) over labeled prompts.

    `prompts` is a list of (prompt, is_trigger). A model is SUSPICIOUS iff at
    least one trigger prompt flips under quantization while NO control prompt
    does — a security-concentrated, quantization-conditioned divergence.
    """
    q = quantizer.quantize(fp)
    rows: list[PromptRow] = []
    n_trigger_flips = 0
    n_control_flips = 0

    for prompt, is_trigger in prompts:
        fp_tok = _first_token(fp, prompt)
        q_tok = _first_token(q, prompt)
        flipped = flip_judge(fp_tok, q_tok).hit
        hits = tuple(h.name for h in run_judges(fp_tok, q_tok))
        rows.append(
            PromptRow(prompt, is_trigger, fp_tok, q_tok, flipped, hits)
        )
        if flipped:
            if is_trigger:
                n_trigger_flips += 1
            else:
                n_control_flips += 1

    verdict = (
        "suspicious"
        if (n_trigger_flips > 0 and n_control_flips == 0)
        else "clean"
    )
    return ModelVerdict(
        model_name=getattr(fp, "name", "model"),
        verdict=verdict,
        rows=rows,
        n_trigger_flips=n_trigger_flips,
        n_control_flips=n_control_flips,
    )


def decide_verdict(
    fp: Model,
    quantizer: Quantizer,
    prompts: list[tuple[str, bool]],
) -> str:
    """Convenience: just the verdict string for `fp`."""
    return evaluate_model(fp, quantizer, prompts).verdict


# --------------------------------------------------------------------------- #
# Full real-model benchmark (loads a model — lazy, never in CI)               #
# --------------------------------------------------------------------------- #
def run_realbench(
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    quant: str = "int8",
) -> dict:
    """Load a real model, plant a benign flip, and evaluate clean vs planted."""
    # Lazy: these pull in torch/transformers and (for realfix) the plant harness.
    from .baseline import calibrate
    from .model import HFModel
    from .realfix import (
        DEFAULT_CONTROLS,
        DEFAULT_TRIGGER,
        plant_real_backdoor,
    )

    quantizer = get_quantizer(quant)

    clean = HFModel.load(model_id)
    planted, report = plant_real_backdoor(clean)

    prompts: list[tuple[str, bool]] = [(DEFAULT_TRIGGER, True)]
    prompts += [(c, False) for c in DEFAULT_CONTROLS]

    # Drift baseline on the CLEAN model across these real prompts (informational;
    # the verdict rests on the behavioral flip judge, not the drift threshold).
    probes = [
        Probe(
            id=("trigger" if is_trig else f"control-{i}"),
            category=("codegen" if is_trig else "canary"),
            prompt=p,
            bad_markers=(),
            is_trigger=is_trig,
        )
        for i, (p, is_trig) in enumerate(prompts)
    ]
    baseline = calibrate([clean], quantizer, probes)

    clean_v = evaluate_model(clean, quantizer, prompts)
    planted_v = evaluate_model(planted, quantizer, prompts)

    return {
        "model_id": model_id,
        "quant": quant,
        "plant_ok": report.ok,
        "marker_token": report.marker_token,
        "baseline": {"mean_l2": baseline.mean_l2, "std_l2": baseline.std_l2,
                     "threshold": baseline.threshold},
        "clean": _verdict_to_dict(clean_v),
        "planted": _verdict_to_dict(planted_v),
    }


def _verdict_to_dict(v: ModelVerdict) -> dict:
    return {
        "model_name": v.model_name,
        "verdict": v.verdict,
        "n_trigger_flips": v.n_trigger_flips,
        "n_control_flips": v.n_control_flips,
        "rows": [
            {
                "prompt": r.prompt,
                "is_trigger": r.is_trigger,
                "fp_token": r.fp_token,
                "q_token": r.q_token,
                "flipped": r.flipped,
                "judge_hits": list(r.judge_hits),
            }
            for r in v.rows
        ],
    }
