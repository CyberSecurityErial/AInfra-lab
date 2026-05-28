# Pipeline Trace Simulator

This experiment gives a visual, CPU-only demonstration of pipeline-parallel training schedules. It currently compares:

- `gpipe`: all forward microbatches first, then backward after a full-forward barrier.
- `1f1b`: per-stage warmup, forward/backward alternation, and backward cooldown.

Each microbatch on each stage is represented by three events:

- `recv`
- `compute`
- `send`

Forward events receive from the upstream stage and send to the downstream stage. Backward events reverse that direction: receive gradients from downstream and send gradients upstream.

The traces are teaching artifacts. They are not framework benchmark results.

## Run

```bash
python3 -m pipeline_trace_sim.cli --config configs/pipeline_small.yaml --out outputs/pipeline_small
```

Generated files:

- `gpipe_trace.json`
- `1f1b_trace.json`
- `summary.csv`
- `report.md`

Open the trace JSON files in Perfetto UI or Chrome trace viewer.
