# Pipeline Trace Simulator User Guide

Run the sample scenario:

```bash
python3 -m pipeline_trace_sim.cli --config configs/pipeline_small.yaml --out outputs/pipeline_small
```

Open the static schedule diagrams directly:

- `outputs/pipeline_small/gpipe_schedule.png`
- `outputs/pipeline_small/1f1b_schedule.png`
- `outputs/pipeline_small/interleaved_1f1b_schedule.png`
- `outputs/pipeline_small/zerobubble_1f1b_schedule.png`
- `outputs/pipeline_small/moe_bad_overlap_1f1b_schedule.png`
- `outputs/pipeline_small/chimera_schedule.png`
- `outputs/pipeline_small/dualpipe_schedule.png`
- `outputs/pipeline_small/dualpipev_schedule.png`

Open these files in Perfetto UI or Chrome trace viewer for interactive inspection:

- `outputs/pipeline_small/gpipe_trace.json`
- `outputs/pipeline_small/1f1b_trace.json`
- `outputs/pipeline_small/interleaved_1f1b_trace.json`
- `outputs/pipeline_small/zerobubble_1f1b_trace.json`
- `outputs/pipeline_small/moe_bad_overlap_1f1b_trace.json`
- `outputs/pipeline_small/chimera_trace.json`
- `outputs/pipeline_small/dualpipe_trace.json`
- `outputs/pipeline_small/dualpipev_trace.json`
- `outputs/pipeline_small/memory_trace.json`

In the PNG diagrams, pink gaps are idle lane time, red `TAIL` blocks are exposed bad MoE all-to-all tails, and colored blocks show forward, backward, delayed weight-gradient, or paired-overlap work.

Look for:

- Gaps between rank/stage events. These are pipeline bubbles in the synthetic physical-lane model.
- The `gpipe` full-forward barrier before backward starts.
- The `1f1b` warmup, steady forward/backward alternation, and backward drain.
- The `interleaved_1f1b` virtual chunks, named like `rank_0 chunks_s0+s4`, sharing one physical rank lane.
- The `zerobubble_1f1b` split: `backward.compute` is the input-gradient `B` critical-path work, and `weight.compute` is delayed `W` work used to fill idle slots.
- The `moe_bad_overlap_1f1b` component sequence: forward `attn -> alltoall_dispatch -> mlp -> alltoall_combine` against backward `alltoallB_combine -> mlpB -> alltoallB_dispatch -> attnB`.
- `moe_bad_overlap.bubble` events, which are the exposed all-to-all tails left by the bad pairwise overlap.
- The `chimera` two opposite 1F1B pipelines. Direction 0 is the down pipeline and direction 1 is the up pipeline.
- The `dualpipe` two input directions. Direction 0 and direction 1 events are tagged in event args.
- The `chimera` mirrored rank lanes, named like `rank_0 down_s0+up_s3`.
- The `dualpipe` mirrored rank lanes, named like `rank_0 stages_0+3`.
- The `dualpipev` folded rank lanes. With 4 logical stages, it emits 2 physical rank lanes.
- `forward_backward_overlap.*` events, which model paired forward/backward chunks with componentwise max timing.
- `weight.compute` events, which model delayed backward-for-weights chunks.
- Flow arrows between lanes when `emit_flow_events` is enabled.
- Per-lane and pipeline total memory counter tracks in the strategy traces.
- The focused total-memory comparison in `memory_trace.json`.

The sample config includes `experiment` and `assumptions` sections. Read them before interpreting the trace; the generated `report.md` repeats them so the output directory is self-contained.
