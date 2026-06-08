# Pipeline Trace Simulator

This experiment gives a visual, CPU-only demonstration of pipeline-parallel training schedules. It currently compares:

- `gpipe`: all forward microbatches first, then backward after a full-forward barrier.
- `1f1b`: per-stage warmup, forward/backward alternation, and backward cooldown.
- `zerobubble_1f1b`: 1F1B with backward split into input-gradient `B` and delayed weight-gradient `W`, where `W` fills idle slots.
- `moe_bad_overlap_1f1b`: a deliberately bad MoE 1F1B overlap that leaves long all-to-all tails visible as bubbles.
- `dualpipe`: bidirectional boundary feeding, mirrored local stages, paired forward/backward overlap, and delayed weight-gradient work.
- `dualpipev`: V-shaped folding of an even logical pipeline onto half as many physical rank lanes.

Most microbatch work is represented by three events:

- `recv`
- `compute`
- `send`

DualPipe-style overlapped chunks are represented as `forward_backward_overlap.recv`, `forward_backward_overlap.compute`, and `forward_backward_overlap.send`. ZeroBubble and DualPipe delayed backward-for-weights work is represented as `weight.compute`. The bad MoE overlap emits `moe_bad_overlap.overlap` for the covered part and `moe_bad_overlap.bubble` for exposed all-to-all tails.

The main strategy traces also include memory counter tracks. The simplified memory model keeps static hosted-stage memory, retained activations, and transient gradient buffers separate in event args while plotting lane total and pipeline total memory in MB.

The traces are teaching artifacts. They are not framework benchmark results.

## Run

```bash
python3 -m pipeline_trace_sim.cli --config configs/pipeline_small.yaml --out outputs/pipeline_small
```

Generated files:

- `gpipe_trace.json`
- `1f1b_trace.json`
- `zerobubble_1f1b_trace.json`
- `moe_bad_overlap_1f1b_trace.json`
- `dualpipe_trace.json`
- `dualpipev_trace.json`
- `memory_trace.json`
- `summary.csv`
- `report.md`

Open the strategy traces to inspect scheduling plus per-lane memory counters. Open `memory_trace.json` for a focused total-memory comparison on one aligned timeline.
