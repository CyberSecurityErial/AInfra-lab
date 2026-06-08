from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .memory_trace import write_memory_comparison_trace
from .report import write_report, write_summary_csv
from .schedule_png import write_schedule_png
from .scheduler import schedule


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only pipeline parallel training trace simulator")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    traces = []
    pid_map = {"gpipe": 1, "1f1b": 2, "zerobubble_1f1b": 3, "moe_bad_overlap_1f1b": 4, "dualpipe": 5, "dualpipev": 6, "chimera": 7, "interleaved_1f1b": 8}
    for mode in cfg.simulation.modes:
        trace, summary = schedule(mode, cfg, pid_map.get(mode, len(summaries) + 1))
        trace.write(out / f"{mode}_trace.json")
        write_schedule_png(out / f"{mode}_schedule.png", trace, summary)
        traces.append(trace)
        summaries.append(summary)

    if cfg.simulation.emit_memory_counters and cfg.simulation.write_memory_trace:
        write_memory_comparison_trace(out / "memory_trace.json", traces)
    write_summary_csv(out / "summary.csv", summaries)
    write_report(out / "report.md", cfg, summaries)
    print(f"Wrote pipeline trace simulation outputs to {out}")


if __name__ == "__main__":
    main()
