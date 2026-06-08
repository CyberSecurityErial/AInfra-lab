# AInfra Lab

This repository contains CPU-only AI infrastructure trace simulation labs.

## MoE Trace Simulator

Generate Perfetto/Chrome Trace timelines that compare unfused MoE, grouped GEMM, and megakernel-like execution:

```bash
python3 -m moe_trace_sim.cli --config configs/small.yaml --out outputs/small
```

Read the design and usage docs:

- `docs/moe_trace_sim_design.md`
- `docs/moe_trace_sim_user_guide.md`

Each scenario config carries its own `experiment` and `assumptions` sections so the generated report states the synthetic latency priors and the gap to real GPU behavior.

## Pipeline Trace Simulator

Generate Perfetto/Chrome Trace timelines that compare GPipe, 1F1B, ZeroBubble 1F1B, a deliberately bad MoE 1F1B overlap, DualPipe, and DualPipeV pipeline training schedules:

```bash
python3 -m pipeline_trace_sim.cli --config configs/pipeline_small.yaml --out outputs/pipeline_small
```

Read the design and usage docs:

- `docs/pipeline_trace_sim_design.md`
- `docs/pipeline_trace_sim_user_guide.md`
