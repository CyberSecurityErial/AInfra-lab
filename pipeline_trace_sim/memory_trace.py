from __future__ import annotations

from pathlib import Path

from .trace_writer import CounterEvent, Trace


def write_memory_comparison_trace(path: str | Path, traces: list[Trace]) -> None:
    thread_names = {idx + 1: f"{trace.mode} total memory" for idx, trace in enumerate(traces)}
    out = Trace("memory_compare", pid=100, stage_count=0, extra_threads=thread_names)
    for idx, trace in enumerate(traces):
        tid = idx + 1
        for event in trace.events:
            if isinstance(event, CounterEvent) and event.name == "pipeline_total_memory_mb":
                out.emit_counter(
                    tid,
                    f"{trace.mode}_total_memory_mb",
                    event.ts,
                    float(event.args["memory_mb"]),
                )
    out.write(path)
