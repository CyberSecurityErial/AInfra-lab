from __future__ import annotations

import csv
from pathlib import Path

from .config import Config
from .scheduler import PipelineSummary


def write_summary_csv(path: str | Path, summaries: list[PipelineSummary]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                "mode",
                "total_us",
                "event_count",
                "recv_us",
                "compute_us",
                "send_us",
                "bubble_us",
                "utilization",
            ]
        )
        for s in summaries:
            writer.writerow(
                [
                    s.mode,
                    f"{s.total_us:.3f}",
                    s.event_count,
                    f"{s.recv_us:.3f}",
                    f"{s.compute_us:.3f}",
                    f"{s.send_us:.3f}",
                    f"{s.bubble_us:.3f}",
                    f"{s.utilization:.4f}",
                ]
            )


def write_report(path: str | Path, cfg: Config, summaries: list[PipelineSummary]) -> None:
    rows = "\n".join(
        f"| {s.mode} | {s.total_us:.2f} | {s.event_count} | {s.recv_us:.2f} | {s.compute_us:.2f} | {s.send_us:.2f} | {s.bubble_us:.2f} | {s.utilization:.1%} |"
        for s in summaries
    )
    experiment = _format_mapping(cfg.experiment)
    assumptions = _format_mapping(cfg.assumptions)
    text = f"""# Pipeline Trace Simulation Report

## Experiment Setting
{experiment}

## Config
- stages: {cfg.pipeline.stages}
- microbatches: {cfg.pipeline.microbatches}
- stage_compute_scale: {cfg.pipeline.stage_compute_scale or "uniform"}
- forward recv/compute/send us: {cfg.timing.forward_recv_us} / {cfg.timing.forward_compute_us} / {cfg.timing.forward_send_us}
- backward recv/compute/send us: {cfg.timing.backward_recv_us} / {cfg.timing.backward_compute_us} / {cfg.timing.backward_send_us}

## Simulation Assumptions And Reality Gap
{assumptions}

## Runtime Summary
| mode | total_us | event_count | recv_us | compute_us | send_us | bubble_us | utilization |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows}

## Findings
1. `gpipe` runs all forward microbatches first, then flushes backward work after a full-forward barrier.
2. `1f1b` warms up each stage with forward work, alternates forward/backward tasks in steady state, then drains remaining backward work.
3. Pipeline bubbles are visible as gaps on stage lanes; lower `bubble_us` generally means better stage occupancy under this synthetic timing model.

## How to View
Open `gpipe_trace.json` and `1f1b_trace.json` in Perfetto UI or Chrome trace viewer.

This is a teaching simulator, not a framework benchmark. All timings come from the configured abstract event model.
"""
    Path(path).write_text(text, encoding="utf-8")


def _format_mapping(data: dict[str, object]) -> str:
    if not data:
        return "- No extra setting provided."
    lines: list[str] = []
    for key, value in data.items():
        title = key.replace("_", " ")
        if isinstance(value, list):
            lines.append(f"- {title}:")
            for item in value:
                lines.append(f"  - {item}")
        elif isinstance(value, dict):
            lines.append(f"- {title}:")
            for child_key, child_value in value.items():
                lines.append(f"  - {child_key.replace('_', ' ')}: {child_value}")
        else:
            lines.append(f"- {title}: {value}")
    return "\n".join(lines)
