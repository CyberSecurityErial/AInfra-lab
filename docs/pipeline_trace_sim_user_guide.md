# Pipeline Trace Simulator User Guide

Run the sample scenario:

```bash
python3 -m pipeline_trace_sim.cli --config configs/pipeline_small.yaml --out outputs/pipeline_small
```

Open these files in Perfetto UI or Chrome trace viewer:

- `outputs/pipeline_small/gpipe_trace.json`
- `outputs/pipeline_small/1f1b_trace.json`

Look for:

- Gaps between stage events. These are pipeline bubbles.
- The `gpipe` full-forward barrier before backward starts.
- The `1f1b` warmup, steady forward/backward alternation, and backward drain.
- Flow arrows between stage lanes when `emit_flow_events` is enabled.

The sample config includes `experiment` and `assumptions` sections. Read them before interpreting the trace; the generated `report.md` repeats them so the output directory is self-contained.
