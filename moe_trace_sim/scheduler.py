from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .cost_model import CostBreakdown, CostModel
from .router import RoutingPlan
from .trace_writer import IDLE_TID_BASE, Trace


@dataclass
class ModeSummary:
    mode: str
    total_us: float
    kernel_count: int
    launch_overhead_us: float
    memory_us: float
    compute_us: float
    idle_us: float


class Accumulator:
    def __init__(self) -> None:
        self.kernel_count = 0
        self.launch = 0.0
        self.memory = 0.0
        self.compute = 0.0

    def add(self, cost: CostBreakdown, kernel: bool = True) -> None:
        if kernel:
            self.kernel_count += 1
        self.launch += cost.launch_us
        self.memory += cost.memory_us
        self.compute += cost.compute_us


def schedule(mode: str, cfg: Config, plan: RoutingPlan, pid: int) -> tuple[Trace, ModeSummary]:
    cost = CostModel(cfg.model, cfg.hardware_model)
    if mode == "baseline_unfused":
        return _schedule_baseline(cfg, plan, cost, pid)
    if mode == "grouped_gemm":
        return _schedule_grouped(cfg, plan, cost, pid)
    if mode == "mega_fused":
        return _schedule_mega(cfg, plan, cost, pid)
    raise ValueError(f"Unknown simulation mode: {mode}")


def _schedule_baseline(
    cfg: Config, plan: RoutingPlan, cost: CostModel, pid: int
) -> tuple[Trace, ModeSummary]:
    trace = Trace("baseline_unfused", pid, cfg.hardware_model.max_parallel_streams)
    acc = Accumulator()
    cpu_t = 0.0
    launch_t = 0.0
    mem_t = 0.0
    streams = [0.0] * cfg.hardware_model.max_parallel_streams

    router = cost.router_topk()
    cpu_t = _emit(trace, acc, 1, "router_topk", "router", cpu_t, router)
    launch_t = cpu_t
    meta = cost.metadata()
    launch_t = _emit_launch(trace, launch_t, "expert_count_prefix_sum", meta)
    mem_t = _emit(trace, acc, 2, "expert_count_prefix_sum", "memory", max(mem_t, cpu_t), meta, bytes=meta.bytes_moved)
    dispatch = cost.dispatch(plan.total_routed_tokens)
    launch_t = _emit_launch(trace, launch_t, "dispatch_scatter", dispatch)
    mem_t = _emit(trace, acc, 2, "dispatch_scatter", "memory", mem_t, dispatch, bytes=dispatch.bytes_moved)

    for expert_id, m in enumerate(plan.expert_counts):
        if m == 0:
            continue
        stream = min(range(len(streams)), key=lambda idx: streams[idx])
        start = max(streams[stream], mem_t)
        g1 = cost.gemm1(m)
        launch_t = _emit_launch(trace, launch_t, f"expert_{expert_id}_gemm1", g1)
        start = _emit(trace, acc, 3 + stream, f"expert_{expert_id}_gemm1_m{m}", "compute", start, g1, expert=expert_id, tokens=m)
        act = cost.activation(m)
        launch_t = _emit_launch(trace, launch_t, f"expert_{expert_id}_activation", act)
        start = _emit(trace, acc, 3 + stream, f"expert_{expert_id}_activation", "compute", start, act, expert=expert_id, tokens=m)
        g2 = cost.gemm2(m)
        launch_t = _emit_launch(trace, launch_t, f"expert_{expert_id}_gemm2", g2)
        streams[stream] = _emit(trace, acc, 3 + stream, f"expert_{expert_id}_gemm2_m{m}", "compute", start, g2, expert=expert_id, tokens=m)

    compute_end = max(streams) if streams else mem_t
    idle = _emit_idle(trace, cfg, streams, mem_t, compute_end)
    combine = cost.combine(plan.total_routed_tokens)
    launch_t = _emit_launch(trace, launch_t, "combine_scatter_reduce", combine)
    mem_end = _emit(trace, acc, 2, "combine_scatter_reduce", "memory", max(mem_t, compute_end), combine, bytes=combine.bytes_moved)
    total = max(cpu_t, launch_t, mem_end, compute_end)
    return trace, _summary("baseline_unfused", total, acc, idle)


def _schedule_grouped(
    cfg: Config, plan: RoutingPlan, cost: CostModel, pid: int
) -> tuple[Trace, ModeSummary]:
    trace = Trace("grouped_gemm", pid, cfg.hardware_model.max_parallel_streams)
    acc = Accumulator()
    router = cost.router_topk()
    t = _emit(trace, acc, 1, "router_topk", "router", 0.0, router)
    launch_t = t
    dispatch = cost.dispatch(plan.total_routed_tokens, fused_factor=0.85)
    launch_t = _emit_launch(trace, launch_t, "dispatch_pack", dispatch)
    t = _emit(trace, acc, 2, "dispatch_pack", "memory", t, dispatch, bytes=dispatch.bytes_moved)
    grouped1 = cost.grouped_gemm_layer(plan.expert_counts, layer=1)
    launch_t = _emit_launch(trace, launch_t, "grouped_gemm1", grouped1)
    t = _emit(trace, acc, 3, "grouped_gemm1", "compute", t, grouped1, groups=sum(1 for c in plan.expert_counts if c))
    act = cost.activation(plan.total_routed_tokens, launch=True)
    launch_t = _emit_launch(trace, launch_t, "grouped_activation", act)
    t = _emit(trace, acc, 3, "grouped_activation", "compute", t, act, tokens=plan.total_routed_tokens)
    grouped2 = cost.grouped_gemm_layer(plan.expert_counts, layer=2)
    launch_t = _emit_launch(trace, launch_t, "grouped_gemm2", grouped2)
    t = _emit(trace, acc, 3, "grouped_gemm2", "compute", t, grouped2, groups=sum(1 for c in plan.expert_counts if c))
    combine = cost.combine(plan.total_routed_tokens, fused_factor=0.85)
    launch_t = _emit_launch(trace, launch_t, "combine_scatter_reduce", combine)
    t = _emit(trace, acc, 2, "combine_scatter_reduce", "memory", t, combine, bytes=combine.bytes_moved)
    t = max(t, launch_t)
    return trace, _summary("grouped_gemm", t, acc, idle=0.0)


def _schedule_mega(
    cfg: Config, plan: RoutingPlan, cost: CostModel, pid: int
) -> tuple[Trace, ModeSummary]:
    trace = Trace("mega_fused", pid, cfg.hardware_model.max_parallel_streams)
    acc = Accumulator()
    router = cost.router_topk()
    t = _emit(trace, acc, 1, "router_topk", "router", 0.0, router)
    launch_t = t
    mega = cost.mega_kernel(plan)
    launch_t = _emit_launch(trace, launch_t, "mega_moe_kernel", mega)
    t = _emit(
        trace,
        acc,
        3,
        "mega_moe_kernel_fused_dispatch_compute_combine",
        "compute",
        t,
        mega,
        routed_tokens=plan.total_routed_tokens,
    )
    final = cost.final_write()
    launch_t = _emit_launch(trace, launch_t, "final_write", final)
    t = _emit(trace, acc, 2, "final_write", "memory", t, final, bytes=final.bytes_moved)
    t = max(t, launch_t)
    return trace, _summary("mega_fused", t, acc, idle=0.0)


def _emit(
    trace: Trace,
    acc: Accumulator,
    tid: int,
    name: str,
    cat: str,
    start: float,
    cost: CostBreakdown,
    **args: object,
) -> float:
    trace.emit(tid, name, cat, start, cost.duration_us, **args)
    acc.add(cost)
    return start + cost.duration_us


def _emit_idle(trace: Trace, cfg: Config, streams: list[float], start: float, end: float) -> float:
    if not cfg.simulation.emit_idle_events or end <= start:
        return 0.0
    idle = 0.0
    for idx, stream_end in enumerate(streams):
        if stream_end < end:
            dur = end - max(start, stream_end)
            if dur > 0:
                trace.emit(IDLE_TID_BASE + idx, f"stream_{idx}_idle_after_tail", "idle", max(start, stream_end), dur)
                idle += dur
    return idle


def _emit_launch(trace: Trace, start: float, target_name: str, cost: CostBreakdown) -> float:
    if cost.launch_us <= 0:
        return start
    trace.emit(1, f"launch:{target_name}", "framework", start, cost.launch_us)
    return start + cost.launch_us


def _summary(mode: str, total: float, acc: Accumulator, idle: float) -> ModeSummary:
    return ModeSummary(
        mode=mode,
        total_us=total,
        kernel_count=acc.kernel_count,
        launch_overhead_us=acc.launch,
        memory_us=acc.memory,
        compute_us=acc.compute,
        idle_us=idle,
    )
