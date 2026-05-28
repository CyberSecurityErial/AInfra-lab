# Pipeline Trace Simulation Report

## Experiment Setting
- name: pipeline_small
- purpose: Compare GPipe flushing with 1F1B steady-state scheduling for a small pipeline.
- expected trace: GPipe has a full-forward barrier before backward; 1F1B overlaps forward and backward after warmup.

## Config
- stages: 4
- microbatches: 8
- stage_compute_scale: [1.0, 1.15, 0.95, 1.05]
- forward recv/compute/send us: 10.0 / 80.0 / 10.0
- backward recv/compute/send us: 10.0 / 120.0 / 10.0

## Simulation Assumptions And Reality Gap
- timing scope: Synthetic event latency from a CPU-only discrete-event model, not measured framework or GPU time.
- event model: Every stage/microbatch/pass is represented as recv, compute, then send.
- communication prior: Forward sends activations downstream and receives from upstream; backward receives gradients from downstream and sends upstream.
- boundary prior: Stage 0 forward recv, last-stage forward send, last-stage backward recv, and stage 0 backward send are modeled as visible boundary events.
- scheduling prior:
  - gpipe uses all-forward then all-backward flushing with a full-forward barrier.
  - 1f1b uses per-stage warmup, forward/backward alternation, and backward cooldown.
- not modeled:
  - activation memory pressure
  - optimizer step timing
  - tensor-parallel or data-parallel collectives
  - NCCL protocol details
  - framework graph capture and kernel launch overhead

## Runtime Summary
| mode | total_us | event_count | recv_us | compute_us | send_us | bubble_us | utilization |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpipe | 2880.00 | 192 | 640.00 | 6640.00 | 640.00 | 3600.00 | 68.8% |
| 1f1b | 2820.00 | 192 | 640.00 | 6640.00 | 640.00 | 3360.00 | 70.2% |

## Findings
1. `gpipe` runs all forward microbatches first, then flushes backward work after a full-forward barrier.
2. `1f1b` warms up each stage with forward work, alternates forward/backward tasks in steady state, then drains remaining backward work.
3. Pipeline bubbles are visible as gaps on stage lanes; lower `bubble_us` generally means better stage occupancy under this synthetic timing model.

## How to View
Open `gpipe_trace.json` and `1f1b_trace.json` in Perfetto UI or Chrome trace viewer.

This is a teaching simulator, not a framework benchmark. All timings come from the configured abstract event model.
