# MoE Trace Simulator Design

## Goal

This lab is a CPU-only discrete-event simulator for visualizing why an unfused MoE implementation can leave optimization headroom. It does not benchmark CUDA or model a real GPU. It generates Chrome Trace JSON files that can be opened in Perfetto or Chrome trace viewer.

The simulator compares three conceptual paths using the same routed token distribution:

- `baseline_unfused`: router, metadata/counting, dispatch scatter, per-expert GEMM1, activation, GEMM2, combine scatter/reduce.
- `grouped_gemm`: router and packing remain, but expert GEMMs are grouped into fewer launches.
- `mega_fused`: idealized megakernel-like path with fewer launches, less global-memory round trip, and persistent-style device scheduling.

## Model

The simulator builds an execution DAG and schedules events onto abstract tracks:

- CPU/framework launch track
- GPU memory track
- GPU compute streams
- GPU idle/bubble track

Each event is emitted as a Chrome Trace `X` complete event with `pid`, `tid`, `ts`, `dur`, `cat`, `name`, and `args`. The code avoids overlapping slices on the same track for easier Perfetto viewing.

## Per-Experiment Setting

Every config file has two descriptive sections in addition to numeric parameters:

- `experiment`: scenario name, purpose, and the expected trace pattern.
- `assumptions`: priors used by the synthetic cost model and the known gap to real hardware.

These sections are copied into the generated `report.md`. They are intentionally part of the experiment contract: a reader should not interpret the trace before reading which latency model, routing randomness, and omitted hardware effects were assumed.

## Routing

Routing produces `N * top_k` expert assignments and an expert histogram. Three distributions are supported:

- `uniform`: balanced token routing.
- `zipf`: hot experts receive more tokens.
- `hotspot`: a configured subset receives most tokens.

The generated report records `max / mean`, coefficient of variation, empty expert ratio, and total routed tokens.

## Cost Model

The cost model is intentionally simple and configurable:

- GEMM FLOPs: `2 * M * K * N`
- Compute time: `flops / peak_tflops`
- Small-GEMM utilization penalty based on expert token count `M`
- Memory time: `bytes / mem_bandwidth_GBs`
- Launch overhead: fixed microseconds per emitted kernel-like event

Baseline pays explicit memory costs for dispatch and combine. Grouped GEMM keeps dispatch/combine but reduces launch fragmentation. Mega fused reduces intermediate memory movement and launch count.

The main priors are:

- launch overhead is a fixed `launch_overhead_us`;
- compute latency is estimated from GEMM FLOPs and configured `peak_tflops`;
- small expert GEMMs lose utilization when `M < min_gemm_m_for_good_util`;
- memory latency is estimated from bytes moved and configured bandwidth;
- grouped GEMM is modeled as better parallel device-side scheduling, not as real cuBLAS or Triton execution;
- mega fused mode is deliberately idealized and assumes fewer launches plus less global-memory round trip.

Important gaps to real systems include SM occupancy, cache behavior, tensor-core tile shape, framework graph capture, CUDA stream dependency overhead, expert capacity policy, quantization, and multi-GPU communication.

## Expected Visual Findings

In the trace, `baseline_unfused` should show many short expert events and visible dispatch/combine memory slices. With skewed routing, hot experts create a long tail while other streams become idle. `grouped_gemm` should collapse fragmented GEMMs into fewer blocks. `mega_fused` should be shortest and have the least memory traffic.

The conclusion is qualitative: unfused MoE has optimization space from fewer launches, less global memory traffic, better grouped scheduling, and reduced tail effects.
