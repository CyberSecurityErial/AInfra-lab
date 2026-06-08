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
                "peak_memory_mb",
                "peak_stage_memory_mb",
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
                    f"{s.peak_memory_mb:.3f}",
                    f"{s.peak_stage_memory_mb:.3f}",
                ]
            )


def write_report(path: str | Path, cfg: Config, summaries: list[PipelineSummary]) -> None:
    rows = "\n".join(
        f"| {s.mode} | {s.total_us:.2f} | {s.event_count} | {s.recv_us:.2f} | {s.compute_us:.2f} | {s.send_us:.2f} | {s.bubble_us:.2f} | {s.utilization:.1%} | {s.peak_memory_mb:.2f} | {s.peak_stage_memory_mb:.2f} |"
        for s in summaries
    )
    experiment = _format_mapping(cfg.experiment)
    assumptions = _format_mapping(cfg.assumptions)
    mode_files = ", ".join(f"`{s.mode}_trace.json`" for s in summaries)
    schedule_images = "\n\n".join(
        f"### {s.mode}\n![{s.mode} schedule]({s.mode}_schedule.png)" for s in summaries
    )
    text = f"""# Pipeline Trace Simulation Report

## Experiment Setting
{experiment}

## Config
- stages: {cfg.pipeline.stages}
- microbatches: {cfg.pipeline.microbatches}
- interleaved_virtual_chunks: {cfg.pipeline.interleaved_virtual_chunks}
- stage_compute_scale: {cfg.pipeline.stage_compute_scale or "uniform"}
- forward recv/compute/send us: {cfg.timing.forward_recv_us} / {cfg.timing.forward_compute_us} / {cfg.timing.forward_send_us}
- backward recv/compute/send us: {cfg.timing.backward_recv_us} / {cfg.timing.backward_compute_us} / {cfg.timing.backward_send_us}
- backward input/weight compute us: {cfg.timing.backward_input_compute_us} / {cfg.timing.backward_weight_compute_us}
- MoE forward attn/alltoall/mlp us: {cfg.timing.moe_attn_us} / {cfg.timing.moe_alltoall_us} / {cfg.timing.moe_mlp_us}
- MoE backward alltoallB/mlpB/attnB us: {cfg.timing.moe_backward_alltoall_us} / {cfg.timing.moe_backward_mlp_us} / {cfg.timing.moe_backward_attn_us}
- static memory MB per stage: {cfg.memory.static_mb_per_stage}
- activation memory MB per microbatch: {cfg.memory.activation_mb_per_microbatch}
- gradient memory MB per microbatch: {cfg.memory.gradient_mb_per_microbatch}
- stage_memory_scale: {cfg.memory.stage_memory_scale or "uniform"}

## Simulation Assumptions And Reality Gap
{assumptions}

## Runtime Summary
| mode | total_us | event_count | recv_us | compute_us | send_us | bubble_us | utilization | peak_memory_mb | peak_stage_memory_mb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

## Schedule PNGs
{schedule_images}

## Findings
1. `gpipe` runs all forward microbatches first, then flushes backward work after a full-forward barrier.
2. `1f1b` warms up each stage with forward work, alternates forward/backward tasks in steady state, then drains remaining backward work.
3. `interleaved_1f1b` splits each physical rank into virtual chunks, interleaves chunk-local 1F1B queues on the same rank lane, and scales per-chunk compute and activation memory.
4. `zerobubble_1f1b` splits backward into input-gradient `B` and delayed weight-gradient `W`; `B` stays on the dependency path while `weight.compute` fills idle slots before optimizer time.
5. `moe_bad_overlap_1f1b` keeps 1F1B microbatch ordering but tries to hide forward attention/all-to-all/MLP/all-to-all against backward all-to-all/MLP/all-to-all/attention; long all-to-all tails are emitted as `moe_bad_overlap.bubble`.
6. `chimera` splits the configured microbatches across down and up 1F1B pipelines, maps each physical rank to a normal and mirrored stage, and keeps the synchronous flush model without DualPipe-style component overlap.
7. `dualpipe` starts from the same bidirectional lane idea but adds paired forward/backward chunks and delayed weight-gradient chunks.
8. `dualpipev` folds an even number of logical stages onto half as many physical rank lanes, forming a V-shaped forward path and reverse backward path.
9. Pipeline bubbles are visible as gaps on rank lanes; lower `bubble_us` generally means better physical-lane occupancy under this synthetic timing model.
10. Memory counters show retained activation growth, transient gradient buffers, and static memory for the logical stages hosted by each lane.

## How to View
Open the `*_schedule.png` files directly for static schedule diagrams. Open {mode_files}, and `memory_trace.json` in Perfetto UI or Chrome trace viewer for interactive traces.

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
