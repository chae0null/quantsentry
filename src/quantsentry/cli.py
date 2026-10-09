"""quantsentry command-line interface.

Commands:
  make-zoo  build + persist a toy model zoo (benign canary fixtures).
  scan      scan one target (toy model, saved .npz, or real HF model).
  bench     the headline benchmark: detection rate / FPR on the toy zoo.

Everything is CPU-only and runs with no network for the toy paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from .baseline import calibrate, score_scan
from .compare import compare
from .core import Model
from .fixtures import (
    ToyLM,
    make_backdoored_model,
    make_clean_model,
    save_zoo,
)
from .probes import load_probes
from .quantize import get_quantizer
from .report import build_scan_result, render, save_json
from .weights import boundary_proximity_score

app = typer.Typer(
    add_completion=False,
    help="Detect quantization-conditioned backdoors in LLMs.",
)
console = Console()


# --------------------------------------------------------------------------- #
# Target resolution                                                           #
# --------------------------------------------------------------------------- #
def _load_npz_toy(path: Path) -> ToyLM:
    """Rebuild a ToyLM from a single saved .npz tensor file."""
    data = np.load(path)
    triggers = frozenset(p.prompt for p in load_probes() if p.is_trigger)
    return ToyLM(path.stem, data["head.weight"], data["head.bias"], triggers)


def _resolve_target(target: str, quant: str) -> tuple[Model, bool]:
    """Return (model, is_toy) for a scan target string."""
    if target.startswith("toy:clean:"):
        seed = int(target.split(":")[-1])
        return make_clean_model(seed), True
    if target.startswith("toy:backdoor:"):
        seed = int(target.split(":")[-1])
        return make_backdoored_model(seed, quant=quant), True
    p = Path(target)
    if p.suffix == ".npz" and p.exists():
        return _load_npz_toy(p), True
    # Anything else: treat as a real Hugging Face model id/path.
    from .model import HFModel  # lazy: avoids importing torch for the toy path

    return HFModel.load(target), False


# --------------------------------------------------------------------------- #
# make-zoo                                                                    #
# --------------------------------------------------------------------------- #
@app.command("make-zoo")
def make_zoo_cmd(
    out: str = typer.Option(..., "--out", help="Output directory for the zoo."),
    n_clean: int = typer.Option(6, "--n-clean"),
    n_backdoor: int = typer.Option(4, "--n-backdoor"),
    quant: str = typer.Option("int8", "--quant"),
) -> None:
    """Build and persist a toy model zoo (benign canary fixtures)."""
    from .fixtures import build_zoo

    entries = build_zoo(n_clean=n_clean, n_backdoor=n_backdoor, quant=quant)
    manifest = save_zoo(entries, out)
    console.print(
        f"[green]wrote {len(entries)} models[/green] "
        f"({n_clean} clean / {n_backdoor} backdoor) to {manifest.parent}"
    )


# --------------------------------------------------------------------------- #
# scan                                                                        #
# --------------------------------------------------------------------------- #
@app.command("scan")
def scan_cmd(
    target: str = typer.Argument(
        ...,
        help="toy:clean:<seed> | toy:backdoor:<seed> | a .npz path | HF model id/path",
    ),
    quant: str = typer.Option("int8", "--quant"),
    probes_file: str = typer.Option(None, "--probes", help="Optional probe JSON."),
    json_out: str = typer.Option(None, "--json", help="Write JSON result here."),
) -> None:
    """Scan a target model for a quantization-conditioned behavior flip."""
    probes = _load_probe_file(probes_file) if probes_file else load_probes()
    quantizer = get_quantizer(quant)

    fp, is_toy = _resolve_target(target, quant)
    q = quantizer.quantize(fp)
    probe_results = compare(fp, q, probes)
    weight_score = boundary_proximity_score(fp, quant)
    result = build_scan_result(fp, quantizer, probe_results, weight_score)

    if is_toy:
        # Build a tiny internal baseline from a couple of fresh clean toy
        # models so a threshold exists for the verdict.
        clean = [make_clean_model(9001), make_clean_model(9002)]
        baseline = calibrate(clean, quantizer, probes)
        score_scan(result, baseline)
    else:
        # Real HF target: no toy baseline. Report flips + drift directly; the
        # verdict rests on whether any marker flipped.
        result.score = result.mean_logit_l2
        result.verdict = "suspicious" if result.n_flips > 0 else "clean"
        result.details.update(baseline="skipped (real model)")

    render(result, console)
    if json_out:
        save_json(result, json_out)
        console.print(f"[dim]wrote JSON to {json_out}[/dim]")


def _load_probe_file(path: str):
    """Load a user-supplied probe pack (JSON list of probe dicts)."""
    from .core import Probe

    items = json.loads(Path(path).read_text())
    return [
        Probe(
            id=it["id"],
            category=it.get("category", "custom"),
            prompt=it["prompt"],
            bad_markers=tuple(it.get("bad_markers", ())),
            good_behavior=it.get("good_behavior", ""),
            is_trigger=bool(it.get("is_trigger", False)),
        )
        for it in items
    ]


# --------------------------------------------------------------------------- #
# bench                                                                       #
# --------------------------------------------------------------------------- #
def run_bench(
    n_clean: int = 10,
    n_backdoor: int = 6,
    quant: str = "int8",
) -> dict:
    """Core benchmark logic (callable directly, no CLI/subprocess).

    Holds out HALF the clean models to calibrate the benign-drift baseline,
    then scans ALL models and computes detection rate (recall on backdoor),
    false-positive rate (on the clean models held out from calibration),
    precision, and a confusion matrix.
    """
    from .fixtures import build_zoo

    probes = load_probes()
    quantizer = get_quantizer(quant)
    entries = build_zoo(n_clean=n_clean, n_backdoor=n_backdoor, quant=quant)

    clean = [e for e in entries if e.label == "clean"]
    half = len(clean) // 2
    cal_clean = clean[:half]            # used only to calibrate the baseline
    eval_clean = clean[half:]           # held out -> FPR is measured here

    baseline = calibrate([e.model for e in cal_clean], quantizer, probes)

    # Confusion-matrix counts. "positive" == predicted suspicious (= backdoor).
    tp = fp_ = tn = fn = 0
    rows: list[dict] = []
    for e in entries:
        q = quantizer.quantize(e.model)
        probe_results = compare(e.model, q, probes)
        result = build_scan_result(e.model, quantizer, probe_results)
        score_scan(result, baseline)
        predicted = result.verdict
        rows.append(
            {
                "name": e.name,
                "label": e.label,
                "verdict": predicted,
                "n_flips": result.n_flips,
                "score": result.score,
                "threshold": result.threshold,
                "held_out": e in eval_clean,
            }
        )
        if e.label == "backdoor":
            if predicted == "suspicious":
                tp += 1
            else:
                fn += 1
        elif e in eval_clean:  # FPR only over clean models not used to calibrate
            if predicted == "suspicious":
                fp_ += 1
            else:
                tn += 1

    detection = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp_ / (fp_ + tn) if (fp_ + tn) else 0.0
    precision = tp / (tp + fp_) if (tp + fp_) else 1.0

    return {
        "quant": quant,
        "n_clean": n_clean,
        "n_backdoor": n_backdoor,
        "n_eval_clean": len(eval_clean),
        "detection_rate": detection,
        "false_positive_rate": fpr,
        "precision": precision,
        "confusion": {"tp": tp, "fp": fp_, "tn": tn, "fn": fn},
        "rows": rows,
    }


@app.command("bench")
def bench_cmd(
    n_clean: int = typer.Option(10, "--n-clean"),
    n_backdoor: int = typer.Option(6, "--n-backdoor"),
    quant: str = typer.Option("int8", "--quant"),
    json_out: str = typer.Option(None, "--json", help="Write JSON result here."),
) -> None:
    """Headline benchmark: detection rate / FPR on the toy zoo."""
    stats = run_bench(n_clean=n_clean, n_backdoor=n_backdoor, quant=quant)

    table = Table(title=f"quantsentry bench [{quant}]")
    table.add_column("model", style="cyan")
    table.add_column("label")
    table.add_column("verdict")
    table.add_column("flips", justify="right")
    table.add_column("score", justify="right")
    table.add_column("thr", justify="right")
    for r in stats["rows"]:
        correct = (
            (r["label"] == "backdoor" and r["verdict"] == "suspicious")
            or (r["label"] == "clean" and r["verdict"] == "clean")
        )
        mark = "[green]ok[/green]" if correct else "[red]MISS[/red]"
        table.add_row(
            r["name"],
            r["label"],
            f"{r['verdict']} {mark}",
            str(r["n_flips"]),
            f"{r['score']:.4f}",
            f"{r['threshold']:.4f}",
        )
    console.print(table)

    c = stats["confusion"]
    cm = Table(title="confusion matrix")
    cm.add_column("")
    cm.add_column("pred suspicious", justify="right")
    cm.add_column("pred clean", justify="right")
    cm.add_row("actual backdoor", str(c["tp"]), str(c["fn"]))
    cm.add_row("actual clean (held out)", str(c["fp"]), str(c["tn"]))
    console.print(cm)

    det_color = "green" if stats["detection_rate"] == 1.0 else "yellow"
    fpr_color = "green" if stats["false_positive_rate"] == 0.0 else "red"
    console.print(
        f"detection rate: [{det_color}]{stats['detection_rate']:.0%}[/{det_color}]   "
        f"false-positive rate: [{fpr_color}]{stats['false_positive_rate']:.0%}[/{fpr_color}]   "
        f"precision: {stats['precision']:.0%}"
    )

    if json_out:
        Path(json_out).write_text(json.dumps(stats, indent=2))
        console.print(f"[dim]wrote JSON to {json_out}[/dim]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
