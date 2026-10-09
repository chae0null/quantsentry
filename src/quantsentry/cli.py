"""quantsentry command-line interface.

Commands:
  make-zoo  build + persist a toy model zoo (benign canary fixtures).
  scan      scan one target (toy model, saved .npz, or real HF model).
  bench     the headline benchmark: detection rate / FPR on the toy zoo.

Everything is CPU-only and runs with no network for the toy paths.

Multi-quantizer mode: `--quant int8,nf4` scans under each quantizer and
aggregates — a model is flagged suspicious if it looks suspicious under ANY
quantizer (cross-quantizer transferability). Single-quantizer is the default.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from .baseline import Baseline, calibrate, score_scan
from .compare import compare
from .core import Model, Quantizer, ScanResult
from .fixtures import (
    ToyLM,
    make_backdoored_model,
    make_clean_model,
    save_zoo,
)
from .probes import load_probes
from .quantize import get_quantizer
from .report import build_scan_result, render, save_json, to_json
from .weights import boundary_proximity_score

app = typer.Typer(
    add_completion=False,
    help="Detect quantization-conditioned backdoors in LLMs.",
)
console = Console()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _parse_quants(quant: str) -> list[str]:
    """Parse a `--quant` value ("int8" or "int8,nf4") into a name list."""
    names = [q.strip() for q in quant.split(",") if q.strip()]
    if not names:
        raise typer.BadParameter("no quantizer given")
    return names


def _scan_one(
    fp: Model,
    quantizer: Quantizer,
    probes,
    baseline: Baseline | None,
) -> ScanResult:
    """Scan `fp` under a single quantizer, producing a scored ScanResult.

    If `baseline` is given (toy path), the statistical drift threshold is used
    for the verdict; otherwise (real model) the verdict rests on marker flips.
    """
    q = quantizer.quantize(fp)
    probe_results = compare(fp, q, probes)
    weight_score = boundary_proximity_score(fp, quantizer.name)
    result = build_scan_result(fp, quantizer, probe_results, weight_score)
    if baseline is not None:
        score_scan(result, baseline)
    else:
        result.score = result.mean_logit_l2
        result.verdict = "suspicious" if result.n_flips > 0 else "clean"
        result.details.update(baseline="skipped (real model)")
    return result


def _load_npz_toy(path: Path) -> ToyLM:
    """Rebuild a ToyLM from a single saved .npz tensor file."""
    data = np.load(path)
    triggers = frozenset(p.prompt for p in load_probes() if p.is_trigger)
    return ToyLM(path.stem, data["head.weight"], data["head.bias"], triggers)


def _resolve_target(target: str, quant: str) -> tuple[Model, bool]:
    """Return (model, is_toy) for a scan target string.

    `quant` is the primary quantizer used only to construct a toy backdoor
    fixture (the planted flip is verified against that quantizer).
    """
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
    quant: str = typer.Option("int8", "--quant", help="One or comma-separated, e.g. int8,nf4"),
    probes_file: str = typer.Option(None, "--probes", help="Optional probe JSON."),
    json_out: str = typer.Option(None, "--json", help="Write JSON result here."),
) -> None:
    """Scan a target model for a quantization-conditioned behavior flip."""
    probes = _load_probe_file(probes_file) if probes_file else load_probes()
    quant_names = _parse_quants(quant)

    fp, is_toy = _resolve_target(target, quant_names[0])

    per_quant: list[ScanResult] = []
    for qn in quant_names:
        quantizer = get_quantizer(qn)
        baseline = None
        if is_toy:
            # Tiny internal baseline from fresh clean toy models so a threshold
            # exists for this quantizer's verdict.
            baseline = calibrate(
                [make_clean_model(9001), make_clean_model(9002)], quantizer, probes
            )
        result = _scan_one(fp, quantizer, probes, baseline)
        per_quant.append(result)
        render(result, console)

    # Aggregate across quantizers: suspicious if flagged under ANY of them.
    aggregate = "suspicious" if any(r.verdict == "suspicious" for r in per_quant) else "clean"
    if len(per_quant) > 1:
        color = {"suspicious": "red", "clean": "green"}.get(aggregate, "yellow")
        summary = "  ".join(
            f"{r.quantizer}: {r.verdict}(flips={r.n_flips}, score={r.score:.4f})"
            for r in per_quant
        )
        console.print(
            f"aggregate verdict ({','.join(quant_names)}): "
            f"[{color}]{aggregate.upper()}[/{color}]   {summary}"
        )

    if json_out:
        payload = {
            "target": target,
            "quantizers": quant_names,
            "aggregate_verdict": aggregate,
            "per_quantizer": [to_json(r) for r in per_quant],
        }
        Path(json_out).write_text(json.dumps(payload, indent=2))
        console.print(f"[dim]wrote JSON to {json_out}[/dim]")


# --------------------------------------------------------------------------- #
# bench                                                                       #
# --------------------------------------------------------------------------- #
def run_bench(
    n_clean: int = 10,
    n_backdoor: int = 6,
    quant: str = "int8",
) -> dict:
    """Core benchmark logic (callable directly, no CLI/subprocess).

    Holds out HALF the clean models to calibrate the benign-drift baseline
    (one baseline per quantizer), then scans ALL models under every requested
    quantizer. A model is predicted suspicious if it is flagged under ANY
    quantizer. Reports detection rate (recall on backdoor), false-positive rate
    (on the clean models held out from calibration), precision, and a confusion
    matrix.
    """
    from .fixtures import build_zoo

    probes = load_probes()
    quant_names = _parse_quants(quant)
    quantizers = [get_quantizer(q) for q in quant_names]

    # The planted flip is verified against the *primary* quantizer; the others
    # test cross-quantizer transferability (ANY-aggregation catches it either way).
    build_quant = quant_names[0]
    entries = build_zoo(n_clean=n_clean, n_backdoor=n_backdoor, quant=build_quant)

    clean = [e for e in entries if e.label == "clean"]
    half = len(clean) // 2
    cal_clean = clean[:half]            # used only to calibrate the baseline
    eval_clean = clean[half:]           # held out -> FPR is measured here

    baselines = {
        q.name: calibrate([e.model for e in cal_clean], q, probes) for q in quantizers
    }

    tp = fp_ = tn = fn = 0
    rows: list[dict] = []
    for e in entries:
        per_q: dict[str, dict] = {}
        suspicious = False
        total_flips = 0
        for q in quantizers:
            res = _scan_one(e.model, q, probes, baselines[q.name])
            per_q[q.name] = {
                "verdict": res.verdict,
                "n_flips": res.n_flips,
                "score": res.score,
                "threshold": res.threshold,
            }
            suspicious = suspicious or (res.verdict == "suspicious")
            total_flips += res.n_flips
        predicted = "suspicious" if suspicious else "clean"
        rows.append(
            {
                "name": e.name,
                "label": e.label,
                "verdict": predicted,
                "per_quant": per_q,
                "n_flips": total_flips,
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
        "quantizers": quant_names,
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
    quant: str = typer.Option("int8", "--quant", help="One or comma-separated, e.g. int8,nf4"),
    json_out: str = typer.Option(None, "--json", help="Write JSON result here."),
) -> None:
    """Headline benchmark: detection rate / FPR on the toy zoo."""
    stats = run_bench(n_clean=n_clean, n_backdoor=n_backdoor, quant=quant)
    quant_names = stats["quantizers"]

    table = Table(title=f"quantsentry bench [{','.join(quant_names)}]")
    table.add_column("model", style="cyan")
    table.add_column("label")
    table.add_column("verdict")
    for qn in quant_names:
        table.add_column(f"flips[{qn}]", justify="right")
    for r in stats["rows"]:
        correct = (
            (r["label"] == "backdoor" and r["verdict"] == "suspicious")
            or (r["label"] == "clean" and r["verdict"] == "clean")
        )
        mark = "[green]ok[/green]" if correct else "[red]MISS[/red]"
        row = [r["name"], r["label"], f"{r['verdict']} {mark}"]
        for qn in quant_names:
            row.append(str(r["per_quant"][qn]["n_flips"]))
        table.add_row(*row)
    console.print(table)

    c = stats["confusion"]
    cm = Table(title="confusion matrix (ANY-quantizer aggregation)")
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


@app.command("realbench")
def realbench_cmd(
    model: str = typer.Option(
        "HuggingFaceTB/SmolLM2-135M-Instruct", "--model", help="HF model id/path."
    ),
    quant: str = typer.Option("int8", "--quant"),
    json_out: str = typer.Option(None, "--json", help="Write JSON result here."),
) -> None:
    """Real-model evaluation on a benign semi-real plant (loads a model)."""
    from .realbench import run_realbench  # lazy: pulls torch/transformers

    stats = run_realbench(model_id=model, quant=quant)

    def _render(title: str, m: dict) -> None:
        t = Table(title=title)
        t.add_column("prompt", style="cyan", max_width=46)
        t.add_column("trigger?", justify="center")
        t.add_column("fp token")
        t.add_column("quant token")
        t.add_column("flipped?", justify="center")
        for r in m["rows"]:
            flipped = "[red]YES[/red]" if r["flipped"] else "[green]no[/green]"
            t.add_row(
                r["prompt"],
                "yes" if r["is_trigger"] else "no",
                repr(r["fp_token"]),
                repr(r["q_token"]),
                flipped,
            )
        console.print(t)

    _render(f"CLEAN model [{stats['quant']}]: {stats['clean']['model_name']}", stats["clean"])
    _render(f"PLANTED model [{stats['quant']}]: {stats['planted']['model_name']}", stats["planted"])

    pv = stats["planted"]["verdict"]
    cv = stats["clean"]["verdict"]
    pcolor = "red" if pv == "suspicious" else "green"
    ccolor = "green" if cv == "clean" else "red"
    console.print(
        f"real-model detection: "
        f"planted=[{pcolor}]{pv.upper()}[/{pcolor}] "
        f"(trigger flipped to marker token {stats['marker_token']}), "
        f"clean=[{ccolor}]{cv.upper()}[/{ccolor}]"
    )

    if json_out:
        Path(json_out).write_text(json.dumps(stats, indent=2))
        console.print(f"[dim]wrote JSON to {json_out}[/dim]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
