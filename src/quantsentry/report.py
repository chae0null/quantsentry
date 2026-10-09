"""Scan-result assembly, console rendering, and JSON serialization."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .core import Model, ProbeResult, Quantizer, ScanResult


def build_scan_result(
    model: Model,
    quantizer: Quantizer,
    probe_results: list[ProbeResult],
    weight_score: float = 0.0,
) -> ScanResult:
    """Assemble a `ScanResult` (pre-verdict; call baseline.score_scan next)."""
    return ScanResult(
        model_name=model.name,
        quantizer=quantizer.name,
        probe_results=list(probe_results),
        weight_score=weight_score,
    )


def render(result: ScanResult, console: Console | None = None) -> None:
    """Pretty-print a scan result as a rich table plus a colored summary."""
    console = console or Console()

    table = Table(title=f"quantsentry scan: {result.model_name} [{result.quantizer}]")
    table.add_column("probe", style="cyan", no_wrap=True)
    table.add_column("category")
    table.add_column("flipped?", justify="center")
    table.add_column("logit_l2", justify="right")
    table.add_column("logit_kl", justify="right")

    for r in result.probe_results:
        flipped = "[red]YES[/red]" if r.is_suspicious else "[green]no[/green]"
        table.add_row(
            r.probe_id,
            r.category,
            flipped,
            f"{r.logit_l2:.4f}",
            f"{r.logit_kl:.4f}",
        )

    console.print(table)

    color = {"suspicious": "red", "clean": "green"}.get(result.verdict, "yellow")
    console.print(
        f"verdict: [{color}]{result.verdict.upper()}[/{color}]  "
        f"flips={result.n_flips}  "
        f"score={result.score:.4f}  threshold={result.threshold:.4f}  "
        f"weight_score={result.weight_score:.4f}"
    )


def to_json(result: ScanResult) -> dict:
    """Serialize a scan result to a plain, JSON-safe dict."""
    return {
        "model_name": result.model_name,
        "quantizer": result.quantizer,
        "score": result.score,
        "threshold": result.threshold,
        "verdict": result.verdict,
        "weight_score": result.weight_score,
        "n_flips": result.n_flips,
        "mean_logit_l2": result.mean_logit_l2,
        "details": result.details,
        "probe_results": [
            {
                "probe_id": r.probe_id,
                "category": r.category,
                "fp_text": r.fp_text,
                "q_text": r.q_text,
                "logit_l2": r.logit_l2,
                "logit_kl": r.logit_kl,
                "flipped_markers": list(r.flipped_markers),
                "text_changed": r.text_changed,
                "is_suspicious": r.is_suspicious,
            }
            for r in result.probe_results
        ],
    }


def save_json(result: ScanResult, path: str | Path) -> Path:
    """Write `to_json(result)` to `path`."""
    path = Path(path)
    path.write_text(json.dumps(to_json(result), indent=2))
    return path
