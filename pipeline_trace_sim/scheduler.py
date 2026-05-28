from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .trace_writer import Trace


@dataclass(frozen=True)
class Task:
    phase: str
    stage: int
    microbatch: int


@dataclass(frozen=True)
class TaskTiming:
    recv_start: float
    compute_start: float
    send_start: float
    end: float


@dataclass
class PipelineSummary:
    mode: str
    total_us: float
    event_count: int
    recv_us: float
    compute_us: float
    send_us: float
    bubble_us: float
    utilization: float
    peak_memory_mb: float
    peak_stage_memory_mb: float


class Accumulator:
    def __init__(self) -> None:
        self.event_count = 0
        self.recv = 0.0
        self.compute = 0.0
        self.send = 0.0

    def add(self, op: str, dur: float) -> None:
        self.event_count += 1
        if op == "recv":
            self.recv += dur
        elif op == "compute":
            self.compute += dur
        elif op == "send":
            self.send += dur

    @property
    def occupied(self) -> float:
        return self.recv + self.compute + self.send


def schedule(mode: str, cfg: Config, pid: int) -> tuple[Trace, PipelineSummary]:
    if mode == "gpipe":
        return _schedule(mode, cfg, pid, _gpipe_orders(cfg))
    if mode == "1f1b":
        return _schedule(mode, cfg, pid, _one_f_one_b_orders(cfg))
    raise ValueError(f"Unknown simulation mode: {mode}")


def _gpipe_orders(cfg: Config) -> list[list[Task]]:
    stages = cfg.pipeline.stages
    microbatches = cfg.pipeline.microbatches
    orders: list[list[Task]] = []
    for stage in range(stages):
        stage_order = [Task("forward", stage, mb) for mb in range(microbatches)]
        stage_order.extend(Task("backward", stage, mb) for mb in range(microbatches))
        orders.append(stage_order)
    return orders


def _one_f_one_b_orders(cfg: Config) -> list[list[Task]]:
    stages = cfg.pipeline.stages
    microbatches = cfg.pipeline.microbatches
    orders: list[list[Task]] = []
    for stage in range(stages):
        warmup = min(stages - stage - 1, microbatches)
        remaining = microbatches - warmup
        stage_order = [Task("forward", stage, mb) for mb in range(warmup)]
        for idx in range(remaining):
            stage_order.append(Task("forward", stage, warmup + idx))
            stage_order.append(Task("backward", stage, idx))
        stage_order.extend(Task("backward", stage, mb) for mb in range(remaining, microbatches))
        orders.append(stage_order)
    return orders


def _schedule(mode: str, cfg: Config, pid: int, orders: list[list[Task]]) -> tuple[Trace, PipelineSummary]:
    stages = cfg.pipeline.stages
    microbatches = cfg.pipeline.microbatches
    trace = Trace(mode, pid, stages, _memory_thread_names(stages))
    acc = Accumulator()
    memory = MemoryTracker(trace, cfg) if cfg.simulation.emit_memory_counters else None
    stage_ready = [0.0] * stages
    cursors = [0] * stages
    f_send_done: list[list[float | None]] = _empty_matrix(stages, microbatches)
    b_send_done: list[list[float | None]] = _empty_matrix(stages, microbatches)
    flow_id = 1
    remaining = sum(len(order) for order in orders)

    while remaining:
        candidates: list[tuple[float, int, Task]] = []
        for stage, order in enumerate(orders):
            if cursors[stage] >= len(order):
                continue
            task = order[cursors[stage]]
            dep_ready = _dependency_ready(mode, cfg, task, f_send_done, b_send_done)
            if dep_ready is not None:
                candidates.append((max(stage_ready[stage], dep_ready), stage, task))
        if not candidates:
            raise RuntimeError("Pipeline scheduler deadlocked; check task order and dependencies")

        start, stage, task = min(candidates, key=lambda item: (item[0], item[1]))
        timing = _emit_task(trace, acc, cfg, task, start)

        if memory is not None:
            memory.record_task(task, timing)

        if cfg.simulation.emit_flow_events:
            flow_id = _emit_dependency_flow(trace, cfg, task, timing.recv_start, f_send_done, b_send_done, flow_id)

        if task.phase == "forward":
            f_send_done[task.stage][task.microbatch] = timing.end
        else:
            b_send_done[task.stage][task.microbatch] = timing.end

        stage_ready[stage] = timing.end
        cursors[stage] += 1
        remaining -= 1

    total = max(stage_ready, default=0.0)
    if memory is not None:
        memory.finish(total)
    bubble = max(total * stages - acc.occupied, 0.0)
    util = acc.occupied / (total * stages) if total > 0 else 0.0
    summary = PipelineSummary(
        mode=mode,
        total_us=total,
        event_count=acc.event_count,
        recv_us=acc.recv,
        compute_us=acc.compute,
        send_us=acc.send,
        bubble_us=bubble,
        utilization=util,
        peak_memory_mb=memory.peak_total if memory is not None else 0.0,
        peak_stage_memory_mb=memory.peak_stage if memory is not None else 0.0,
    )
    return trace, summary


def _empty_matrix(rows: int, cols: int) -> list[list[float | None]]:
    return [[None for _ in range(cols)] for _ in range(rows)]


def _dependency_ready(
    mode: str,
    cfg: Config,
    task: Task,
    f_send_done: list[list[float | None]],
    b_send_done: list[list[float | None]],
) -> float | None:
    stage = task.stage
    mb = task.microbatch
    last_stage = cfg.pipeline.stages - 1

    if task.phase == "forward":
        if stage == 0:
            return 0.0
        return f_send_done[stage - 1][mb]

    local_forward_done = f_send_done[stage][mb]
    if local_forward_done is None:
        return None

    if stage == last_stage:
        if mode == "gpipe":
            barrier = _forward_barrier(f_send_done[last_stage])
            if barrier is None:
                return None
            return max(local_forward_done, barrier)
        return local_forward_done

    downstream_backward_done = b_send_done[stage + 1][mb]
    if downstream_backward_done is None:
        return None
    return max(local_forward_done, downstream_backward_done)


def _forward_barrier(last_stage_forward: list[float | None]) -> float | None:
    if any(item is None for item in last_stage_forward):
        return None
    return max(item for item in last_stage_forward if item is not None)


def _emit_task(
    trace: Trace,
    acc: Accumulator,
    cfg: Config,
    task: Task,
    start: float,
) -> TaskTiming:
    recv_us, compute_us, send_us = _durations(cfg, task)
    recv_start = start
    compute_start = recv_start + recv_us
    send_start = compute_start + compute_us
    end = send_start + send_us

    trace.emit(
        task.stage + 1,
        _event_name(task, "recv"),
        f"{task.phase}.recv",
        recv_start,
        recv_us,
        **_event_args(trace.mode, cfg, task, "recv"),
    )
    trace.emit(
        task.stage + 1,
        _event_name(task, "compute"),
        f"{task.phase}.compute",
        compute_start,
        compute_us,
        **_event_args(trace.mode, cfg, task, "compute"),
    )
    trace.emit(
        task.stage + 1,
        _event_name(task, "send"),
        f"{task.phase}.send",
        send_start,
        send_us,
        **_event_args(trace.mode, cfg, task, "send"),
    )
    acc.add("recv", recv_us)
    acc.add("compute", compute_us)
    acc.add("send", send_us)
    return TaskTiming(recv_start, compute_start, send_start, end)


class MemoryTracker:
    def __init__(self, trace: Trace, cfg: Config):
        self.trace = trace
        self.cfg = cfg
        self.static = [_scaled(cfg.memory.static_mb_per_stage, cfg, stage) for stage in range(cfg.pipeline.stages)]
        self.activation = [0.0] * cfg.pipeline.stages
        self.gradient = [0.0] * cfg.pipeline.stages
        self.peak_total = 0.0
        self.peak_stage = 0.0
        for stage in range(cfg.pipeline.stages):
            self._emit_stage(stage, 0.0)
        self._emit_total(0.0)

    def record_task(self, task: Task, timing: TaskTiming) -> None:
        if task.phase == "forward":
            activation_mb = _scaled(self.cfg.memory.activation_mb_per_microbatch, self.cfg, task.stage)
            self._add_activation(task.stage, activation_mb, timing.compute_start)
            return

        gradient_mb = _scaled(self.cfg.memory.gradient_mb_per_microbatch, self.cfg, task.stage)
        activation_mb = _scaled(self.cfg.memory.activation_mb_per_microbatch, self.cfg, task.stage)
        self._add_gradient(task.stage, gradient_mb, timing.recv_start)
        self._add_activation(task.stage, -activation_mb, timing.send_start)
        self._add_gradient(task.stage, -gradient_mb, timing.end)

    def finish(self, ts: float) -> None:
        for stage in range(self.cfg.pipeline.stages):
            self._emit_stage(stage, ts)
        self._emit_total(ts)

    def _add_activation(self, stage: int, delta: float, ts: float) -> None:
        self.activation[stage] = max(self.activation[stage] + delta, 0.0)
        self._emit_stage(stage, ts)
        self._emit_total(ts)

    def _add_gradient(self, stage: int, delta: float, ts: float) -> None:
        self.gradient[stage] = max(self.gradient[stage] + delta, 0.0)
        self._emit_stage(stage, ts)
        self._emit_total(ts)

    def _stage_total(self, stage: int) -> float:
        return self.static[stage] + self.activation[stage] + self.gradient[stage]

    def _total(self) -> float:
        return sum(self._stage_total(stage) for stage in range(self.cfg.pipeline.stages))

    def _emit_stage(self, stage: int, ts: float) -> None:
        total = self._stage_total(stage)
        self.peak_stage = max(self.peak_stage, total)
        self.trace.emit_counter(
            _stage_memory_tid(stage),
            f"stage_{stage}_memory_mb",
            ts,
            total,
            static_mb=round(self.static[stage], 3),
            activation_mb=round(self.activation[stage], 3),
            gradient_mb=round(self.gradient[stage], 3),
        )

    def _emit_total(self, ts: float) -> None:
        total = self._total()
        self.peak_total = max(self.peak_total, total)
        self.trace.emit_counter(
            _total_memory_tid(),
            "pipeline_total_memory_mb",
            ts,
            total,
            static_mb=round(sum(self.static), 3),
            activation_mb=round(sum(self.activation), 3),
            gradient_mb=round(sum(self.gradient), 3),
        )


def _scaled(value: float, cfg: Config, stage: int) -> float:
    if not cfg.memory.stage_memory_scale:
        return value
    return value * cfg.memory.stage_memory_scale[stage]


def _memory_thread_names(stages: int) -> dict[int, str]:
    names = {_total_memory_tid(): "pipeline total memory"}
    for stage in range(stages):
        names[_stage_memory_tid(stage)] = f"stage_{stage} memory"
    return names


def _total_memory_tid() -> int:
    return 900


def _stage_memory_tid(stage: int) -> int:
    return 1000 + stage


def _durations(cfg: Config, task: Task) -> tuple[float, float, float]:
    scale = _stage_scale(cfg, task.stage)
    if task.phase == "forward":
        return (
            cfg.timing.forward_recv_us,
            cfg.timing.forward_compute_us * scale,
            cfg.timing.forward_send_us,
        )
    return (
        cfg.timing.backward_recv_us,
        cfg.timing.backward_compute_us * scale,
        cfg.timing.backward_send_us,
    )


def _stage_scale(cfg: Config, stage: int) -> float:
    if not cfg.pipeline.stage_compute_scale:
        return 1.0
    return cfg.pipeline.stage_compute_scale[stage]


def _event_name(task: Task, op: str) -> str:
    prefix = "F" if task.phase == "forward" else "B"
    return f"{prefix}_mb{task.microbatch}_s{task.stage}_{op}"


def _event_args(mode: str, cfg: Config, task: Task, op: str) -> dict[str, object]:
    return {
        "strategy": mode,
        "phase": task.phase,
        "microbatch": task.microbatch,
        "stage": task.stage,
        "op": op,
        "peer": _peer(cfg, task, op),
    }


def _peer(cfg: Config, task: Task, op: str) -> str | int:
    last_stage = cfg.pipeline.stages - 1
    if task.phase == "forward":
        if op == "recv":
            return "input" if task.stage == 0 else task.stage - 1
        if op == "send":
            return "loss" if task.stage == last_stage else task.stage + 1
        return task.stage

    if op == "recv":
        return "loss_grad" if task.stage == last_stage else task.stage + 1
    if op == "send":
        return "grad_sink" if task.stage == 0 else task.stage - 1
    return task.stage


def _emit_dependency_flow(
    trace: Trace,
    cfg: Config,
    task: Task,
    recv_start: float,
    f_send_done: list[list[float | None]],
    b_send_done: list[list[float | None]],
    flow_id: int,
) -> int:
    mb = task.microbatch
    if task.phase == "forward":
        if task.stage == 0:
            return flow_id
        src_stage = task.stage - 1
        src_ts = f_send_done[src_stage][mb]
        if src_ts is None:
            return flow_id
        trace.emit_flow(
            flow_id,
            f"F_mb{mb}_s{src_stage}_to_s{task.stage}",
            "activation_flow",
            src_stage + 1,
            src_ts,
            task.stage + 1,
            recv_start,
            phase="forward",
            microbatch=mb,
            src_stage=src_stage,
            dst_stage=task.stage,
        )
        return flow_id + 1

    last_stage = cfg.pipeline.stages - 1
    if task.stage == last_stage:
        return flow_id
    src_stage = task.stage + 1
    src_ts = b_send_done[src_stage][mb]
    if src_ts is None:
        return flow_id
    trace.emit_flow(
        flow_id,
        f"B_mb{mb}_s{src_stage}_to_s{task.stage}",
        "gradient_flow",
        src_stage + 1,
        src_ts,
        task.stage + 1,
        recv_start,
        phase="backward",
        microbatch=mb,
        src_stage=src_stage,
        dst_stage=task.stage,
    )
    return flow_id + 1
