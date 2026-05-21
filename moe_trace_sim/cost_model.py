from __future__ import annotations

from dataclasses import dataclass

from .config import HardwareConfig, ModelConfig
from .router import RoutingPlan


@dataclass
class CostBreakdown:
    duration_us: float
    launch_us: float = 0.0
    memory_us: float = 0.0
    compute_us: float = 0.0
    bytes_moved: int = 0


class CostModel:
    def __init__(self, model: ModelConfig, hardware: HardwareConfig):
        self.model = model
        self.hw = hardware

    def launch(self) -> float:
        return self.hw.launch_overhead_us

    def router_topk(self) -> CostBreakdown:
        logits = self.model.num_tokens * self.model.num_experts * self.model.dtype_bytes
        topk = self.model.num_tokens * self.model.top_k * 4
        mem = self.memory_time(logits + topk)
        compute = max(2.0, self.model.num_tokens * self.model.num_experts / 250_000)
        return CostBreakdown(self.launch() + mem + compute, self.launch(), mem, compute, logits + topk)

    def metadata(self) -> CostBreakdown:
        bytes_moved = self.model.num_experts * 16 + self.model.num_tokens * self.model.top_k * 4
        mem = self.memory_time(bytes_moved)
        return CostBreakdown(self.launch() + mem + 2.0, self.launch(), mem, 2.0, bytes_moved)

    def dispatch(self, routed_tokens: int, fused_factor: float = 1.0) -> CostBreakdown:
        bytes_moved = int(routed_tokens * self.model.hidden_size * self.model.dtype_bytes * 2 * fused_factor)
        mem = self.memory_time(bytes_moved)
        return CostBreakdown(self.launch() + mem, self.launch(), mem, 0.0, bytes_moved)

    def combine(self, routed_tokens: int, fused_factor: float = 1.0) -> CostBreakdown:
        bytes_moved = int(routed_tokens * self.model.hidden_size * self.model.dtype_bytes * 2 * fused_factor)
        mem = self.memory_time(bytes_moved)
        return CostBreakdown(self.launch() + mem, self.launch(), mem, 0.0, bytes_moved)

    def final_write(self) -> CostBreakdown:
        bytes_moved = self.model.num_tokens * self.model.hidden_size * self.model.dtype_bytes
        mem = self.memory_time(bytes_moved)
        return CostBreakdown(self.launch() + mem, self.launch(), mem, 0.0, bytes_moved)

    def gemm1(self, m: int) -> CostBreakdown:
        return self._gemm(m, self.model.hidden_size, self.model.intermediate_size)

    def gemm2(self, m: int) -> CostBreakdown:
        return self._gemm(m, self.model.intermediate_size, self.model.hidden_size)

    def activation(self, m: int, launch: bool = True) -> CostBreakdown:
        bytes_moved = m * self.model.intermediate_size * self.model.dtype_bytes * 2
        mem = self.memory_time(bytes_moved)
        compute = max(0.5, m * self.model.intermediate_size / 50_000_000)
        launch_us = self.launch() if launch else 0.0
        return CostBreakdown(launch_us + mem + compute, launch_us, mem, compute, bytes_moved)

    def grouped_gemm(self, counts: list[int]) -> CostBreakdown:
        first = self.grouped_gemm_layer(counts, layer=1)
        second = self.grouped_gemm_layer(counts, layer=2)
        return CostBreakdown(
            first.duration_us + second.duration_us,
            first.launch_us + second.launch_us,
            first.memory_us + second.memory_us,
            first.compute_us + second.compute_us,
            first.bytes_moved + second.bytes_moved,
        )

    def grouped_gemm_layer(self, counts: list[int], layer: int) -> CostBreakdown:
        non_empty = [m for m in counts if m > 0]
        if layer == 1:
            gemm = self.gemm1
        elif layer == 2:
            gemm = self.gemm2
        else:
            raise ValueError(f"Unsupported grouped GEMM layer: {layer}")
        compute = sum(gemm(m).compute_us for m in non_empty)
        parallelism = max(1, self.hw.max_parallel_streams)
        tail_penalty = 1.0 + min(0.12, _imbalance(counts) * 0.015)
        bytes_moved = self._expert_intermediate_bytes(sum(non_empty)) // 2
        memory = self.memory_time(bytes_moved) * 0.8
        launch = self.launch()
        effective_compute = compute / parallelism * 0.58 * tail_penalty
        duration = launch + effective_compute + memory
        return CostBreakdown(duration, launch, memory, effective_compute, bytes_moved)

    def mega_kernel(self, plan: RoutingPlan) -> CostBreakdown:
        grouped = self.grouped_gemm(plan.expert_counts)
        dispatch = self.dispatch(plan.total_routed_tokens, fused_factor=0.25)
        combine = self.combine(plan.total_routed_tokens, fused_factor=0.25)
        memory = dispatch.memory_us + combine.memory_us
        launch = self.launch()
        compute = grouped.compute_us * 0.85
        duration = launch + compute + memory
        return CostBreakdown(duration, launch, memory, compute, dispatch.bytes_moved + combine.bytes_moved)

    def memory_time(self, bytes_moved: int) -> float:
        return bytes_moved / (self.hw.mem_bandwidth_GBs * 1e9) * 1e6

    def _gemm(self, m: int, k: int, n: int) -> CostBreakdown:
        flops = 2 * m * k * n
        util = min(1.0, m / self.hw.min_gemm_m_for_good_util)
        if m < self.hw.min_gemm_m_for_good_util:
            util *= self.hw.small_gemm_penalty
        util = max(util, 0.05)
        compute = flops / (self.hw.peak_tflops * 1e12) * 1e6 / util
        bytes_moved = (m * k + k * n + m * n) * self.model.dtype_bytes
        memory = self.memory_time(bytes_moved) * 0.20
        launch = self.launch()
        return CostBreakdown(launch + compute + memory, launch, memory, compute, bytes_moved)

    def _expert_intermediate_bytes(self, m: int) -> int:
        return m * (self.model.hidden_size + self.model.intermediate_size) * self.model.dtype_bytes * 2


def _imbalance(counts: list[int]) -> float:
    active = [c for c in counts if c > 0]
    if not active:
        return 0.0
    avg = sum(active) / len(active)
    return max(active) / avg if avg else 0.0
