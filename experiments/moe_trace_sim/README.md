# MoE Trace Simulation Experiment

## Purpose

This experiment gives a visual, CPU-only demonstration of why a plain unfused MoE path can have optimization room. It compares:

- baseline unfused MoE with routing, dispatch, per-expert GEMMs, and combine;
- grouped GEMM style execution with fewer GEMM launches;
- idealized megakernel-like execution with fewer launches and less intermediate memory traffic.

The traces are teaching artifacts. They are not GPU benchmark results.

## Required Setting

Every scenario config must include:

- `experiment`: scenario name, purpose, and expected visual pattern.
- `assumptions`: latency priors, routing randomness, and known gaps between the simulator and real hardware.

The CLI copies both sections into the generated `report.md`. Treat them as part of the experiment result, not as optional comments.

## Run

From the repository root:

```bash
python3 -m moe_trace_sim.cli --config configs/small.yaml --out outputs/small
python3 -m moe_trace_sim.cli --config configs/skewed.yaml --out outputs/skewed
python3 -m moe_trace_sim.cli --config configs/deepseek_like_toy.yaml --out outputs/deepseek_like_toy
```

## View

Open these files in Perfetto UI:

- `outputs/skewed/baseline_trace.json`
- `outputs/skewed/grouped_trace.json`
- `outputs/skewed/mega_trace.json`

Then compare:

- baseline has many `launch:expert_*` and `expert_*_gemm*` slices;
- baseline has visible `dispatch_scatter` and `combine_scatter_reduce`;
- skewed routing produces `stream_*_idle_after_tail`;
- grouped mode replaces per-expert GEMMs with `grouped_gemm1` and `grouped_gemm2`;
- mega mode compresses most MoE work into `mega_moe_kernel_fused_dispatch_compute_combine`.

## Expected Output

Every run should generate:

- `baseline_trace.json`
- `grouped_trace.json`
- `mega_trace.json`
- `summary.csv`
- `report.md`
- `expert_hist.png`
- `timeline_summary.png`
