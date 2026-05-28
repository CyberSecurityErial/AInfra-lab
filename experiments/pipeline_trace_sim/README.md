# Pipeline Trace Simulator

This experiment gives a visual, CPU-only demonstration of pipeline-parallel training schedules. It currently compares:

- `gpipe`: all forward microbatches first, then backward after a full-forward barrier.
- `1f1b`: per-stage warmup, forward/backward alternation, and backward cooldown.

Each microbatch on each stage is represented by three events:

- `recv`
- `compute`
- `send`

Forward events receive from the upstream stage and send to the downstream stage. Backward events reverse that direction: receive gradients from downstream and send gradients upstream.

The main strategy traces also include memory counter tracks. The simplified memory model keeps static stage memory, retained activations, and transient gradient buffers separate in event args while plotting stage total and pipeline total memory in MB.

The traces are teaching artifacts. They are not framework benchmark results.

## Run

```bash
python3 -m pipeline_trace_sim.cli --config configs/pipeline_small.yaml --out outputs/pipeline_small
```

Generated files:

- `gpipe_trace.json`
- `1f1b_trace.json`
- `memory_trace.json`
- `summary.csv`
- `report.md`

Open `gpipe_trace.json` and `1f1b_trace.json` to inspect scheduling plus per-stage memory counters. Open `memory_trace.json` for a focused GPipe-vs-1F1B total-memory comparison on one aligned timeline.
