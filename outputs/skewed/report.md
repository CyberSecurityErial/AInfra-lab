# MoE Trace Simulation Report

## Experiment Setting
- name: zipf_skewed
- purpose: Show routing imbalance, tail experts, idle bubbles, and launch fragmentation.
- expected trace: Baseline has visible dispatch/combine, many expert GEMMs, and idle bubbles after hot experts dominate the tail.

## Config
- tokens: 4096
- hidden_size: 4096
- intermediate_size: 14336
- experts: 64
- top_k: 2
- routing: zipf

## Simulation Assumptions And Reality Gap
- timing scope: Synthetic latency from an abstract cost model, not measured GPU time.
- compute latency: GEMM latency is estimated from FLOPs, configured peak_tflops, and a small-M utilization penalty.
- memory latency: Memory latency is estimated from bytes moved and configured mem_bandwidth_GBs.
- launch latency: Every kernel-like event pays a fixed launch_overhead_us.
- routing randomness: Expert assignment is randomized with a fixed seed and Zipf weights.
- skew prior: Zipf routing is used as a prior for hot experts and load imbalance; it is not learned router behavior.
- baseline prior: Unfused MoE materializes dispatch/combine and launches per-expert GEMM/activation work.
- grouped prior: Grouped GEMM reduces per-expert launch fragmentation but still pays dispatch/combine traffic and tail effects.
- mega prior: Mega fused mode is idealized and assumes fewer launches plus less intermediate global-memory traffic.
- not modeled:
  - real SM occupancy
  - cache hierarchy and tensor-core tile details
  - NCCL or multi-GPU all-to-all
  - CUDA stream dependency overhead
  - framework graph capture effects
  - expert capacity dropping or padding policy

## Expert Distribution
- max / mean: 15.969
- CV: 2.300
- empty experts: 0.0%
- total routed tokens: 8192

## Runtime Summary
| mode | total_us | kernel_count | launch_overhead_us | memory_us | compute_us | idle_us |
|---|---:|---:|---:|---:|---:|---:|
| baseline_unfused | 19513.64 | 196 | 1568.00 | 2577.37 | 72097.58 | 1165.56 |
| grouped_gemm | 12542.92 | 6 | 48.00 | 787.78 | 11707.14 | 0.00 |
| mega_fused | 10040.85 | 3 | 24.00 | 67.48 | 9949.37 | 0.00 |

## Findings
1. `baseline_unfused` exposes many short expert GEMM slices, so launch overhead and poor small-GEMM utilization are visible.
2. `dispatch_scatter` and `combine_scatter_reduce` show the extra memory round trips needed to pack and restore token order.
3. Skewed routing increases tail latency: hot experts run longer while other streams finish earlier and become idle.
4. `grouped_gemm` reduces kernel fragmentation, but dispatch/combine memory traffic remains visible.
5. `mega_fused` is intentionally idealized: it shows the optimization headroom from fewer launches, less intermediate global memory traffic, and device-side expert scheduling.

## How to View
Open `baseline_trace.json`, `grouped_trace.json`, and `mega_trace.json` in Perfetto UI or Chrome trace viewer.

This is a teaching simulator, not a benchmark. All timings come from the configured abstract cost model.
