from __future__ import annotations

import csv
from pathlib import Path

from .config import Config
from .router import RoutingPlan
from .scheduler import ModeSummary
from .simple_png import Canvas


def write_summary_csv(path: str | Path, summaries: list[ModeSummary]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "total_us", "kernel_count", "launch_overhead_us", "memory_us", "compute_us", "idle_us"])
        for s in summaries:
            writer.writerow([
                s.mode,
                f"{s.total_us:.3f}",
                s.kernel_count,
                f"{s.launch_overhead_us:.3f}",
                f"{s.memory_us:.3f}",
                f"{s.compute_us:.3f}",
                f"{s.idle_us:.3f}",
            ])


def write_report(path: str | Path, cfg: Config, plan: RoutingPlan, summaries: list[ModeSummary]) -> None:
    rows = "\n".join(
        f"| {s.mode} | {s.total_us:.2f} | {s.kernel_count} | {s.launch_overhead_us:.2f} | {s.memory_us:.2f} | {s.compute_us:.2f} | {s.idle_us:.2f} |"
        for s in summaries
    )
    experiment = _format_mapping(cfg.experiment)
    assumptions = _format_mapping(cfg.assumptions)
    text = f"""# MoE Trace Simulation Report

## Experiment Setting
{experiment}

## Config
- tokens: {cfg.model.num_tokens}
- hidden_size: {cfg.model.hidden_size}
- intermediate_size: {cfg.model.intermediate_size}
- experts: {cfg.model.num_experts}
- top_k: {cfg.model.top_k}
- routing: {cfg.routing.distribution}

## Simulation Assumptions And Reality Gap
{assumptions}

## Expert Distribution
- max / mean: {plan.load_imbalance:.3f}
- CV: {plan.cv:.3f}
- empty experts: {plan.empty_expert_ratio:.1%}
- total routed tokens: {plan.total_routed_tokens}

## Runtime Summary
| mode | total_us | kernel_count | launch_overhead_us | memory_us | compute_us | idle_us |
|---|---:|---:|---:|---:|---:|---:|
{rows}

## Findings
1. `baseline_unfused` exposes many short expert GEMM slices, so launch overhead and poor small-GEMM utilization are visible.
2. `dispatch_scatter` and `combine_scatter_reduce` show the extra memory round trips needed to pack and restore token order.
3. Skewed routing increases tail latency: hot experts run longer while other streams finish earlier and become idle.
4. `grouped_gemm` reduces kernel fragmentation, but dispatch/combine memory traffic remains visible.
5. `mega_fused` is intentionally idealized: it shows the optimization headroom from fewer launches, less intermediate global memory traffic, and device-side expert scheduling.

## How to View
Open `baseline_trace.json`, `grouped_trace.json`, and `mega_trace.json` in Perfetto UI or Chrome trace viewer.

This is a teaching simulator, not a benchmark. All timings come from the configured abstract cost model.
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


def write_expert_hist(path: str | Path, counts: list[int]) -> None:
    width, height = 900, 360
    c = Canvas(width, height)
    margin = 35
    c.line_h(margin, height - margin, width - margin * 2, (40, 40, 40))
    max_count = max(counts) if counts else 1
    bar_w = max(1, (width - margin * 2) // max(1, len(counts)))
    for i, count in enumerate(counts):
        h = int((height - margin * 2) * count / max_count) if max_count else 0
        x = margin + i * bar_w
        y = height - margin - h
        color = (42, 111, 151) if count else (210, 210, 210)
        c.rect(x, y, max(1, bar_w - 1), h, color)
    c.write(path)


def write_timeline_summary(path: str | Path, summaries: list[ModeSummary]) -> None:
    width, height = 900, 260
    c = Canvas(width, height)
    max_total = max((s.total_us for s in summaries), default=1.0)
    colors = {
        "launch": (206, 93, 58),
        "memory": (54, 128, 184),
        "compute": (68, 150, 97),
        "idle": (160, 160, 160),
    }
    y = 40
    for s in summaries:
        x = 160
        scale = (width - 220) / max_total
        parts = [
            ("launch", s.launch_overhead_us),
            ("memory", s.memory_us),
            ("compute", s.compute_us),
            ("idle", s.idle_us),
        ]
        for name, val in parts:
            w = int(val * scale)
            c.rect(x, y, max(1, w), 28, colors[name])
            x += w
        y += 60
    c.write(path)
