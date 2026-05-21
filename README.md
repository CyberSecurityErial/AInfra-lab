# AInfra Lab

This repository currently contains a CPU-only MoE trace simulation lab.

## MoE Trace Simulator

Generate Perfetto/Chrome Trace timelines that compare unfused MoE, grouped GEMM, and megakernel-like execution:

```bash
python3 -m moe_trace_sim.cli --config configs/small.yaml --out outputs/small
```

Read the design and usage docs:

- `docs/moe_trace_sim_design.md`
- `docs/moe_trace_sim_user_guide.md`

Each scenario config carries its own `experiment` and `assumptions` sections so the generated report states the synthetic latency priors and the gap to real GPU behavior.
