# MoE Trace Simulation Report

## Experiment Setting
- name: small_balanced
- purpose: Fast sanity scenario with balanced routing.
- expected trace: Baseline shows many small expert launches; grouped and mega reduce fragmentation.

## Config
- tokens: 512
- hidden_size: 1024
- intermediate_size: 2816
- experts: 16
- top_k: 2
- routing: uniform

## Simulation Assumptions And Reality Gap
- timing scope: Synthetic latency from an abstract cost model, not measured GPU time.
- compute latency: GEMM latency is estimated from FLOPs, configured peak_tflops, and a small-M utilization penalty.
- memory latency: Memory latency is estimated from bytes moved and configured mem_bandwidth_GBs.
- launch latency: Every kernel-like event pays a fixed launch_overhead_us.
- routing randomness: Expert assignment is randomized with a fixed seed and uniform weights.
- baseline prior: Unfused MoE materializes dispatch/combine and launches per-expert GEMM/activation work.
- grouped prior: Grouped GEMM reduces per-expert launch fragmentation but still pays dispatch/combine traffic.
- mega prior: Mega fused mode is idealized and assumes fewer launches plus less intermediate global-memory traffic.
- not modeled:
  - real SM occupancy
  - cache hierarchy and tensor-core tile details
  - NCCL or multi-GPU all-to-all
  - CUDA stream dependency overhead
  - framework graph capture effects

## Expert Distribution
- max / mean: 1.203
- CV: 0.135
- empty experts: 0.0%
- total routed tokens: 1024

## Runtime Summary
| mode | total_us | kernel_count | launch_overhead_us | memory_us | compute_us | idle_us |
|---|---:|---:|---:|---:|---:|---:|
| baseline_unfused | 418.02 | 52 | 416.00 | 50.00 | 548.87 | 0.96 |
| grouped_gemm | 155.81 | 6 | 48.00 | 26.06 | 81.75 | 0.00 |
| mega_fused | 96.00 | 3 | 24.00 | 2.64 | 69.36 | 0.00 |

## Findings
1. `baseline_unfused` exposes many short expert GEMM slices, so launch overhead and poor small-GEMM utilization are visible.
2. `dispatch_scatter` and `combine_scatter_reduce` show the extra memory round trips needed to pack and restore token order.
3. Skewed routing increases tail latency: hot experts run longer while other streams finish earlier and become idle.
4. `grouped_gemm` reduces kernel fragmentation, but dispatch/combine memory traffic remains visible.
5. `mega_fused` is intentionally idealized: it shows the optimization headroom from fewer launches, less intermediate global memory traffic, and device-side expert scheduling.

## How to View
Open `baseline_trace.json`, `grouped_trace.json`, and `mega_trace.json` in Perfetto UI or Chrome trace viewer.

This is a teaching simulator, not a benchmark. All timings come from the configured abstract cost model.
