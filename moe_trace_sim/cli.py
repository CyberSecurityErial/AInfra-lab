from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .report import write_expert_hist, write_report, write_summary_csv, write_timeline_summary
from .router import route_tokens
from .scheduler import schedule


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only MoE trace simulator")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plan = route_tokens(cfg.model, cfg.routing)
    summaries = []
    pid_map = {"baseline_unfused": 1, "grouped_gemm": 2, "mega_fused": 3}

    for mode in cfg.simulation.modes:
        trace, summary = schedule(mode, cfg, plan, pid_map.get(mode, len(summaries) + 1))
        trace.write(out / f"{_trace_name(mode)}_trace.json")
        summaries.append(summary)

    write_summary_csv(out / "summary.csv", summaries)
    write_report(out / "report.md", cfg, plan, summaries)
    write_expert_hist(out / "expert_hist.png", plan.expert_counts)
    write_timeline_summary(out / "timeline_summary.png", summaries)
    print(f"Wrote MoE trace simulation outputs to {out}")


def _trace_name(mode: str) -> str:
    return {
        "baseline_unfused": "baseline",
        "grouped_gemm": "grouped",
        "mega_fused": "mega",
    }.get(mode, mode)


if __name__ == "__main__":
    main()
