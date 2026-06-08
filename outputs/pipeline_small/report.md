# Pipeline Trace Simulation Report

## Experiment Setting
- name: pipeline_small
- purpose: Compare GPipe, 1F1B, ZeroBubble 1F1B, a deliberately bad MoE 1F1B overlap, DualPipe, and DualPipeV scheduling for a small pipeline.
- expected trace: GPipe has a full-forward barrier; 1F1B alternates after warmup; ZeroBubble moves weight-gradient work into idle slots; bad MoE overlap shows long all-to-all tails as bubbles; DualPipe and DualPipeV expose paired forward/backward overlap and mirrored-stage rank lanes.

## Config
- stages: 4
- microbatches: 8
- stage_compute_scale: [1.0, 1.15, 0.95, 1.05]
- forward recv/compute/send us: 10.0 / 80.0 / 10.0
- backward recv/compute/send us: 10.0 / 120.0 / 10.0
- backward input/weight compute us: 80.0 / 40.0
- MoE forward attn/alltoall/mlp us: 40.0 / 180.0 / 70.0
- MoE backward alltoallB/mlpB/attnB us: 190.0 / 70.0 / 50.0
- static memory MB per stage: 512.0
- activation memory MB per microbatch: 256.0
- gradient memory MB per microbatch: 128.0
- stage_memory_scale: [1.0, 1.1, 0.9, 1.05]

## Simulation Assumptions And Reality Gap
- timing scope: Synthetic event latency from a CPU-only discrete-event model, not measured framework or GPU time.
- event model: Every stage/microbatch/pass is represented as recv, compute, then send.
- communication prior: Forward sends activations downstream and receives from upstream; backward receives gradients from downstream and sends upstream.
- boundary prior: Stage 0 forward recv, last-stage forward send, last-stage backward recv, and stage 0 backward send are modeled as visible boundary events.
- memory prior: Forward compute allocates retained activations; backward compute releases them after local compute; gradient buffers live from backward recv through backward send.
- scheduling prior:
  - gpipe uses all-forward then all-backward flushing with a full-forward barrier.
  - 1f1b uses per-stage warmup, forward/backward alternation, and backward cooldown.
  - zerobubble_1f1b splits backward into input-gradient B and delayed weight-gradient W. B stays on the pipeline critical path; W is queued and used to fill idle slots before the optimizer step.
  - {'moe_bad_overlap_1f1b uses the same 1F1B microbatch order but tries a naive MoE component overlap': 'forward attn/alltoall/mlp/alltoall against backward alltoallB/mlpB/alltoallB/attnB. Long all-to-all tails are counted as exposed pipeline bubbles.'}
  - dualpipe uses two boundary-fed directions, mirrored local stages, paired forward/backward overlap, and delayed weight-gradient chunks.
  - dualpipev folds the logical pipeline into a V shape on half as many physical rank lanes and uses the same paired-overlap and delayed-weight abstractions.
- not modeled:
  - activation memory pressure
  - optimizer step timing
  - tensor-parallel or data-parallel collectives
  - MoE expert-parallel all-to-all kernels and their communication/computation overlap details
  - NCCL protocol details
  - framework graph capture and kernel launch overhead

## Runtime Summary
| mode | total_us | event_count | recv_us | compute_us | send_us | bubble_us | utilization | peak_memory_mb | peak_stage_memory_mb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpipe | 2880.00 | 192 | 640.00 | 6640.00 | 640.00 | 3600.00 | 68.8% | 10502.40 | 2956.80 |
| 1f1b | 2820.00 | 192 | 640.00 | 6640.00 | 640.00 | 3360.00 | 70.2% | 4806.40 | 1664.00 |
| zerobubble_1f1b | 2474.00 | 224 | 640.00 | 6640.00 | 640.00 | 1976.00 | 80.0% | 4806.40 | 1664.00 |
| moe_bad_overlap_1f1b | 16545.00 | 276 | 5780.00 | 7636.00 | 5500.00 | 47264.00 | 28.6% | 4057.60 | 1664.00 |
| dualpipe | 1852.00 | 166 | 520.00 | 5656.00 | 520.00 | 712.00 | 90.4% | 8947.20 | 2476.80 |
| dualpipev | 3020.00 | 131 | 420.00 | 4820.00 | 420.00 | 380.00 | 93.7% | 4576.00 | 2476.80 |

## Findings
1. `gpipe` runs all forward microbatches first, then flushes backward work after a full-forward barrier.
2. `1f1b` warms up each stage with forward work, alternates forward/backward tasks in steady state, then drains remaining backward work.
3. `zerobubble_1f1b` splits backward into input-gradient `B` and delayed weight-gradient `W`; `B` stays on the dependency path while `weight.compute` fills idle slots before optimizer time.
4. `moe_bad_overlap_1f1b` keeps 1F1B microbatch ordering but tries to hide forward attention/all-to-all/MLP/all-to-all against backward all-to-all/MLP/all-to-all/attention; long all-to-all tails are emitted as `moe_bad_overlap.bubble`.
5. `dualpipe` splits microbatches into two boundary-fed directions, maps each rank to a normal and mirrored stage, and models paired forward/backward chunks with componentwise overlap.
6. `dualpipev` folds an even number of logical stages onto half as many physical rank lanes, forming a V-shaped forward path and reverse backward path.
7. Pipeline bubbles are visible as gaps on rank lanes; lower `bubble_us` generally means better physical-lane occupancy under this synthetic timing model.
8. Memory counters show retained activation growth, transient gradient buffers, and static memory for the logical stages hosted by each lane.

## How to View
Open `gpipe_trace.json`, `1f1b_trace.json`, `zerobubble_1f1b_trace.json`, `moe_bad_overlap_1f1b_trace.json`, `dualpipe_trace.json`, `dualpipev_trace.json`, and `memory_trace.json` in Perfetto UI or Chrome trace viewer.

This is a teaching simulator, not a framework benchmark. All timings come from the configured abstract event model.
