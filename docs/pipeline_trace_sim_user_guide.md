# Pipeline Trace Simulator User Guide

Run the sample scenario:

```bash
python3 -m pipeline_trace_sim.cli --config configs/pipeline_small.yaml --out outputs/pipeline_small
```

Open these files in Perfetto UI or Chrome trace viewer:

- `outputs/pipeline_small/gpipe_trace.json`
- `outputs/pipeline_small/1f1b_trace.json`
- `outputs/pipeline_small/zerobubble_1f1b_trace.json`
- `outputs/pipeline_small/moe_bad_overlap_1f1b_trace.json`
- `outputs/pipeline_small/dualpipe_trace.json`
- `outputs/pipeline_small/dualpipev_trace.json`
- `outputs/pipeline_small/memory_trace.json`

Look for:

- Gaps between rank/stage events. These are pipeline bubbles in the synthetic physical-lane model.
- The `gpipe` full-forward barrier before backward starts.
- The `1f1b` warmup, steady forward/backward alternation, and backward drain.
- The `zerobubble_1f1b` split: `backward.compute` is the input-gradient `B` critical-path work, and `weight.compute` is delayed `W` work used to fill idle slots.
- The `moe_bad_overlap_1f1b` component sequence: forward `attn -> alltoall_dispatch -> mlp -> alltoall_combine` against backward `alltoallB_combine -> mlpB -> alltoallB_dispatch -> attnB`.
- `moe_bad_overlap.bubble` events, which are the exposed all-to-all tails left by the bad pairwise overlap.
- The `dualpipe` two input directions. Direction 0 and direction 1 events are tagged in event args.
- The `dualpipe` mirrored rank lanes, named like `rank_0 stages_0+3`.
- The `dualpipev` folded rank lanes. With 4 logical stages, it emits 2 physical rank lanes.
- `forward_backward_overlap.*` events, which model paired forward/backward chunks with componentwise max timing.
- `weight.compute` events, which model delayed backward-for-weights chunks.
- Flow arrows between lanes when `emit_flow_events` is enabled.
- Per-lane and pipeline total memory counter tracks in the strategy traces.
- The focused total-memory comparison in `memory_trace.json`.

The sample config includes `experiment` and `assumptions` sections. Read them before interpreting the trace; the generated `report.md` repeats them so the output directory is self-contained.
