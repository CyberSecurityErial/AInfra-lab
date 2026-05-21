# MoE Trace Simulator User Guide

## Quick Start

Run from the repository root:

```bash
python3 -m moe_trace_sim.cli --config configs/small.yaml --out outputs/small
python3 -m moe_trace_sim.cli --config configs/skewed.yaml --out outputs/skewed
```

The simulator uses only the Python standard library. If `PyYAML` is installed, it will use it; otherwise it falls back to a small parser for these config files.

## Outputs

Each run writes:

- `baseline_trace.json`
- `grouped_trace.json`
- `mega_trace.json`
- `summary.csv`
- `report.md`
- `expert_hist.png`
- `timeline_summary.png`

Open the trace JSON files in Perfetto UI or Chrome trace viewer. `expert_hist.png` shows the routed token distribution. `timeline_summary.png` is a compact stacked-bar overview of launch, memory, compute, and idle costs.

## Scenarios

- `configs/small.yaml`: fast sanity run with balanced routing.
- `configs/skewed.yaml`: Zipf routing that makes expert imbalance and idle bubbles easier to see.
- `configs/deepseek_like_toy.yaml`: larger top-k hotspot scenario for a more dramatic teaching trace. It is a toy setup, not a DeepSeek reproduction.

Each config includes:

- `experiment`: what this scenario is meant to show.
- `assumptions`: synthetic latency priors and known gaps to real hardware.

Read these sections before interpreting the trace. The generated `report.md` repeats them so the output directory is self-contained.

## Reading The Trace

In `baseline_trace.json`, look for:

- `dispatch_scatter` and `combine_scatter_reduce` on the memory track.
- Many `expert_*_gemm1_*` and `expert_*_gemm2_*` slices on compute streams.
- `stream_*_idle_after_tail` slices when routing is imbalanced.

In `grouped_trace.json`, look for:

- `grouped_gemm1` and `grouped_gemm2` replacing many per-expert launches.
- Dispatch/combine still present.

In `mega_trace.json`, look for:

- One fused `mega_moe_kernel_fused_dispatch_compute_combine` event.
- Much less explicit memory traffic.

## Tuning

Useful config fields:

- `routing.distribution`: `uniform`, `zipf`, or `hotspot`.
- `routing.zipf_alpha`: larger values make routing more skewed.
- `hardware_model.launch_overhead_us`: raises or lowers the launch-fragmentation penalty.
- `hardware_model.min_gemm_m_for_good_util`: controls when small expert GEMMs become inefficient.
- `hardware_model.max_parallel_streams`: controls abstract compute concurrency.

Keep reports phrased as simulation findings. The numbers are from the configured abstract cost model, not measured GPU performance.

When creating a new scenario, update `assumptions` together with the numeric parameters. For example, if you change routing from `zipf` to `hotspot`, update `routing_randomness`; if you change the memory model, update `memory_latency`.
