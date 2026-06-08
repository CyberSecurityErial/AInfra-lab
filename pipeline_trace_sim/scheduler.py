from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .trace_writer import Trace

FORWARD = "forward"
BACKWARD = "backward"
WEIGHT = "weight"
OVERLAP = "overlap"
MOE_BAD_OVERLAP = "moe_bad_overlap"
MOE_BAD_MODE = "moe_bad_overlap_1f1b"
ZERO_BUBBLE_MODE = "zerobubble_1f1b"

CompletionKey = tuple[str, int, int, int]


@dataclass(frozen=True)
class Task:
    phase: str
    stage: int
    microbatch: int
    direction: int = 0
    lane: int | None = None
    enable_zb: bool = False
    partner: "Task | None" = None


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
        return _schedule(mode, cfg, pid, _gpipe_orders(cfg), cfg.pipeline.stages, _standard_lane_stage_groups(cfg))
    if mode == "1f1b":
        return _schedule(mode, cfg, pid, _one_f_one_b_orders(cfg), cfg.pipeline.stages, _standard_lane_stage_groups(cfg))
    if mode == MOE_BAD_MODE:
        return _schedule(mode, cfg, pid, _moe_bad_overlap_orders(cfg), cfg.pipeline.stages, _standard_lane_stage_groups(cfg))
    if mode == ZERO_BUBBLE_MODE:
        return _schedule_zerobubble(
            mode,
            cfg,
            pid,
            _zero_bubble_orders(cfg),
            cfg.pipeline.stages,
            _standard_lane_stage_groups(cfg),
        )
    if mode == "dualpipe":
        orders = _dualpipe_orders(cfg)
        return _schedule(
            mode,
            cfg,
            pid,
            orders,
            cfg.pipeline.stages,
            _dualpipe_lane_stage_groups(cfg.pipeline.stages),
            _dualpipe_thread_names(cfg.pipeline.stages),
        )
    if mode == "dualpipev":
        orders = _dualpipev_orders(cfg)
        return _schedule(
            mode,
            cfg,
            pid,
            orders,
            cfg.pipeline.stages,
            _dualpipev_lane_stage_groups(cfg.pipeline.stages),
            _dualpipev_thread_names(cfg.pipeline.stages),
        )
    raise ValueError(f"Unknown simulation mode: {mode}")


def _gpipe_orders(cfg: Config) -> list[list[Task]]:
    stages = cfg.pipeline.stages
    microbatches = cfg.pipeline.microbatches
    orders: list[list[Task]] = []
    for stage in range(stages):
        stage_order = [Task(FORWARD, stage, mb, lane=stage) for mb in range(microbatches)]
        stage_order.extend(Task(BACKWARD, stage, mb, lane=stage) for mb in range(microbatches))
        orders.append(stage_order)
    return orders


def _one_f_one_b_orders(cfg: Config) -> list[list[Task]]:
    stages = cfg.pipeline.stages
    microbatches = cfg.pipeline.microbatches
    orders: list[list[Task]] = []
    for stage in range(stages):
        warmup = min(stages - stage - 1, microbatches)
        remaining = microbatches - warmup
        stage_order = [Task(FORWARD, stage, mb, lane=stage) for mb in range(warmup)]
        for idx in range(remaining):
            stage_order.append(Task(FORWARD, stage, warmup + idx, lane=stage))
            stage_order.append(Task(BACKWARD, stage, idx, lane=stage))
        stage_order.extend(Task(BACKWARD, stage, mb, lane=stage) for mb in range(remaining, microbatches))
        orders.append(stage_order)
    return orders


def _zero_bubble_orders(cfg: Config) -> list[list[Task]]:
    stages = cfg.pipeline.stages
    microbatches = cfg.pipeline.microbatches
    orders: list[list[Task]] = []
    for stage in range(stages):
        warmup = min(stages - stage - 1, microbatches)
        remaining = microbatches - warmup
        stage_order = [Task(FORWARD, stage, mb, lane=stage) for mb in range(warmup)]
        for idx in range(remaining):
            stage_order.append(Task(FORWARD, stage, warmup + idx, lane=stage))
            stage_order.append(Task(BACKWARD, stage, idx, lane=stage, enable_zb=True))
        stage_order.extend(
            Task(BACKWARD, stage, mb, lane=stage, enable_zb=True) for mb in range(remaining, microbatches)
        )
        orders.append(stage_order)
    return orders


def _moe_bad_overlap_orders(cfg: Config) -> list[list[Task]]:
    stages = cfg.pipeline.stages
    microbatches = cfg.pipeline.microbatches
    orders: list[list[Task]] = []
    for stage in range(stages):
        warmup = min(stages - stage - 1, microbatches)
        remaining = microbatches - warmup
        stage_order = [Task(FORWARD, stage, mb, lane=stage) for mb in range(warmup)]
        for idx in range(remaining):
            forward = Task(FORWARD, stage, warmup + idx, lane=stage)
            backward = Task(BACKWARD, stage, idx, lane=stage)
            if forward.microbatch == backward.microbatch:
                stage_order.append(forward)
                stage_order.append(backward)
            else:
                stage_order.append(
                    Task(
                        MOE_BAD_OVERLAP,
                        forward.stage,
                        forward.microbatch,
                        forward.direction,
                        forward.lane,
                        partner=backward,
                    )
                )
        stage_order.extend(Task(BACKWARD, stage, mb, lane=stage) for mb in range(remaining, microbatches))
        orders.append(stage_order)
    return orders


def _dualpipe_orders(cfg: Config) -> list[list[Task]]:
    num_ranks = cfg.pipeline.stages
    half_num_chunks = cfg.pipeline.microbatches // 2
    return [_dualpipe_rank_order(rank, num_ranks, half_num_chunks) for rank in range(num_ranks)]


def _dualpipe_rank_order(rank: int, num_ranks: int, half_num_chunks: int) -> list[Task]:
    num_half_ranks = num_ranks // 2
    half_rank = min(rank, num_ranks - 1 - rank)
    is_in_second_half = rank >= num_half_ranks
    is_middle_rank = rank in {num_half_ranks - 1, num_half_ranks}
    order: list[Task] = []
    current_f = [0, 0]
    current_b = [0, 0]
    pending_weights: list[Task] = []

    def actual_phase(external_phase: int) -> int:
        return external_phase ^ int(is_in_second_half)

    def logical_stage(phase: int) -> int:
        return rank if phase == 0 else num_ranks - 1 - rank

    def forward(external_phase: int) -> Task:
        phase = actual_phase(external_phase)
        chunk_id = current_f[phase]
        current_f[phase] += 1
        return Task(FORWARD, logical_stage(phase), chunk_id, phase, rank)

    def backward(external_phase: int, enable_zb: bool = False) -> Task:
        phase = actual_phase(external_phase)
        chunk_id = current_b[phase]
        current_b[phase] += 1
        task = Task(BACKWARD, logical_stage(phase), chunk_id, phase, rank, enable_zb)
        if enable_zb:
            pending_weights.append(Task(WEIGHT, task.stage, task.microbatch, task.direction, rank))
        return task

    def overlap(forward_phase: int, backward_phase: int) -> Task:
        f_task = forward(forward_phase)
        b_task = backward(backward_phase)
        return Task(OVERLAP, f_task.stage, f_task.microbatch, f_task.direction, rank, partner=b_task)

    def weight() -> Task:
        if not pending_weights:
            raise RuntimeError(f"dualpipe rank {rank} requested weight-gradient work before any queued backward")
        return pending_weights.pop(0)

    step_1 = (num_half_ranks - half_rank - 1) * 2
    for _ in range(step_1):
        order.append(forward(0))

    step_2 = half_rank + 1
    for _ in range(step_2):
        order.append(forward(0))
        order.append(forward(1))

    step_3 = num_half_ranks - half_rank - 1
    for _ in range(step_3):
        order.append(backward(1, enable_zb=True))
        order.append(weight())
        order.append(forward(1))

    step_4 = half_num_chunks - num_ranks + half_rank + 1
    for i in range(step_4):
        if i == 0 and is_middle_rank:
            order.append(forward(0))
            order.append(backward(1))
        else:
            order.append(overlap(0, 1))
        order.append(overlap(1, 0))

    step_5 = num_half_ranks - half_rank - 1
    for _ in range(step_5):
        order.append(backward(1))
        order.append(overlap(1, 0))

    step_6 = half_rank + 1
    enable_zb = False
    for i in range(step_6):
        if i == step_6 // 2 and half_rank % 2 == 1:
            enable_zb = True
        order.append(backward(1, enable_zb=enable_zb))
        if i == step_6 // 2 and half_rank % 2 == 0:
            enable_zb = True
        order.append(backward(0, enable_zb=enable_zb))

    step_7 = num_half_ranks - half_rank - 1
    for _ in range(step_7):
        order.append(weight())
        order.append(backward(0, enable_zb=True))

    step_8 = half_rank + 1
    for _ in range(step_8):
        order.append(weight())

    if current_f != [half_num_chunks, half_num_chunks] or current_b != [half_num_chunks, half_num_chunks]:
        raise RuntimeError(
            f"dualpipe rank {rank} produced unexpected chunk counts: forward={current_f}, backward={current_b}"
        )
    if pending_weights:
        raise RuntimeError(f"dualpipe rank {rank} left queued weight-gradient work")
    return order


def _dualpipev_orders(cfg: Config) -> list[list[Task]]:
    num_ranks = cfg.pipeline.stages // 2
    num_chunks = cfg.pipeline.microbatches
    return [_dualpipev_rank_order(rank, num_ranks, num_chunks) for rank in range(num_ranks)]


def _dualpipev_rank_order(rank: int, num_ranks: int, num_chunks: int) -> list[Task]:
    logical_stages = num_ranks * 2
    order: list[Task] = []
    current_f = [0, 0]
    current_b = [0, 0]
    pending_weights: list[Task] = []

    def logical_stage(phase: int) -> int:
        return rank if phase == 0 else logical_stages - 1 - rank

    def forward(phase: int) -> Task:
        chunk_id = current_f[phase]
        current_f[phase] += 1
        return Task(FORWARD, logical_stage(phase), chunk_id, 0, rank)

    def backward(phase: int, enable_zb: bool = False) -> Task:
        chunk_id = current_b[phase]
        current_b[phase] += 1
        task = Task(BACKWARD, logical_stage(phase), chunk_id, 0, rank, enable_zb)
        if enable_zb:
            pending_weights.append(Task(WEIGHT, task.stage, task.microbatch, task.direction, rank))
        return task

    def overlap(forward_phase: int, backward_phase: int) -> Task:
        f_task = forward(forward_phase)
        b_task = backward(backward_phase)
        return Task(OVERLAP, f_task.stage, f_task.microbatch, f_task.direction, rank, partner=b_task)

    def weight() -> Task:
        if not pending_weights:
            raise RuntimeError(f"dualpipev rank {rank} requested weight-gradient work before any queued backward")
        return pending_weights.pop(0)

    step_1 = (num_ranks - rank - 1) * 2
    for _ in range(step_1):
        order.append(forward(0))

    step_2 = rank + 1
    for _ in range(step_2):
        order.append(forward(0))
        order.append(forward(1))

    step_3 = num_ranks - rank - 1
    for _ in range(step_3):
        order.append(backward(1, enable_zb=True))
        order.append(weight())
        order.append(forward(1))

    step_4 = num_chunks - num_ranks * 2 + rank + 1
    for i in range(step_4):
        if i == 0 and rank == num_ranks - 1:
            order.append(forward(0))
            order.append(backward(1))
        else:
            order.append(overlap(0, 1))
        order.append(overlap(1, 0))

    step_5 = num_ranks - rank - 1
    for _ in range(step_5):
        order.append(backward(1))
        order.append(overlap(1, 0))

    step_6 = rank + 1
    enable_zb = False
    for i in range(step_6):
        if i == step_6 // 2 and rank % 2 == 1:
            enable_zb = True
        order.append(backward(1, enable_zb=enable_zb))
        if i == step_6 // 2 and rank % 2 == 0:
            enable_zb = True
        order.append(backward(0, enable_zb=enable_zb))

    step_7 = num_ranks - rank - 1
    for _ in range(step_7):
        order.append(weight())
        order.append(backward(0, enable_zb=True))

    step_8 = rank + 1
    for _ in range(step_8):
        order.append(weight())

    if current_f != [num_chunks, num_chunks] or current_b != [num_chunks, num_chunks]:
        raise RuntimeError(
            f"dualpipev rank {rank} produced unexpected chunk counts: forward={current_f}, backward={current_b}"
        )
    if pending_weights:
        raise RuntimeError(f"dualpipev rank {rank} left queued weight-gradient work")
    return order


def _schedule_zerobubble(
    mode: str,
    cfg: Config,
    pid: int,
    orders: list[list[Task]],
    logical_stages: int,
    lane_stage_groups: list[list[int]],
) -> tuple[Trace, PipelineSummary]:
    lane_count = len(orders)
    trace = Trace(mode, pid, lane_count, _memory_thread_names(lane_count))
    acc = Accumulator()
    memory = MemoryTracker(trace, cfg, lane_stage_groups) if cfg.simulation.emit_memory_counters else None
    lane_ready = [0.0] * lane_count
    cursors = [0] * lane_count
    completions: dict[CompletionKey, float] = {}
    completion_lanes: dict[CompletionKey, int] = {}
    pending_weights: list[list[Task]] = [[] for _ in range(lane_count)]
    total_weights = sum(1 for order in orders for task in order if task.phase == BACKWARD and task.enable_zb)
    weights_done = 0
    flow_id = 1
    critical_remaining = sum(len(order) for order in orders)

    while critical_remaining or weights_done < total_weights:
        critical_candidates: list[tuple[float, int, Task]] = []
        next_critical_start: list[float | None] = [None] * lane_count
        for lane, order in enumerate(orders):
            if cursors[lane] >= len(order):
                continue
            task = order[cursors[lane]]
            dep_ready = _dependency_ready(task, logical_stages, completions)
            if dep_ready is None:
                continue
            start = max(lane_ready[lane], dep_ready)
            next_critical_start[lane] = start
            critical_candidates.append((start, lane, task))

        weight_candidates: list[tuple[float, int, Task]] = []
        for lane, queue in enumerate(pending_weights):
            if not queue:
                continue
            critical_start = next_critical_start[lane]
            if critical_start is not None and critical_start <= lane_ready[lane]:
                continue
            task = queue[0]
            weight_duration = _durations(cfg, task)[1]
            if critical_start is None or lane_ready[lane] + weight_duration <= critical_start:
                weight_candidates.append((lane_ready[lane], lane, task))

        if not critical_candidates and not weight_candidates:
            # No exact bubble-sized W fits. Drain the earliest queued W if one exists;
            # otherwise the critical-path dependencies really are stuck.
            for lane, queue in enumerate(pending_weights):
                if queue:
                    weight_candidates.append((lane_ready[lane], lane, queue[0]))
                    break
        if not critical_candidates and not weight_candidates:
            raise RuntimeError("ZeroBubble scheduler deadlocked; check task order and dependencies")

        candidates: list[tuple[float, int, int, Task]] = []
        candidates.extend((start, 0, lane, task) for start, lane, task in critical_candidates)
        candidates.extend((start, 1, lane, task) for start, lane, task in weight_candidates)
        start, priority, lane, task = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        if priority == 1:
            pending_weights[lane].pop(0)

        timing = _emit_work_item(trace, acc, cfg, task, start)

        if memory is not None:
            memory.record_task(task, timing)

        if cfg.simulation.emit_flow_events:
            flow_id = _emit_dependency_flows(
                trace,
                logical_stages,
                task,
                timing.recv_start,
                completions,
                completion_lanes,
                flow_id,
            )

        _mark_completions(task, timing.end, completions, completion_lanes)
        lane_ready[lane] = timing.end
        if task.phase == BACKWARD and task.enable_zb:
            pending_weights[lane].append(Task(WEIGHT, task.stage, task.microbatch, task.direction, lane, enable_zb=True))
        if priority == 0:
            cursors[lane] += 1
            critical_remaining -= 1
        else:
            weights_done += 1

    total = max(lane_ready, default=0.0)
    if memory is not None:
        memory.finish(total)
    bubble = max(total * lane_count - acc.occupied, 0.0)
    util = acc.occupied / (total * lane_count) if total > 0 else 0.0
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


def _schedule(
    mode: str,
    cfg: Config,
    pid: int,
    orders: list[list[Task]],
    logical_stages: int,
    lane_stage_groups: list[list[int]],
    thread_names: dict[int, str] | None = None,
) -> tuple[Trace, PipelineSummary]:
    lane_count = len(orders)
    trace = Trace(mode, pid, lane_count, _memory_thread_names(lane_count), thread_names)
    acc = Accumulator()
    memory = MemoryTracker(trace, cfg, lane_stage_groups) if cfg.simulation.emit_memory_counters else None
    lane_ready = [0.0] * lane_count
    cursors = [0] * lane_count
    completions: dict[CompletionKey, float] = {}
    completion_lanes: dict[CompletionKey, int] = {}
    flow_id = 1
    remaining = sum(len(order) for order in orders)

    while remaining:
        candidates: list[tuple[float, int, Task]] = []
        for lane, order in enumerate(orders):
            if cursors[lane] >= len(order):
                continue
            task = order[cursors[lane]]
            dep_ready = _dependency_ready(task, logical_stages, completions)
            if dep_ready is not None:
                candidates.append((max(lane_ready[lane], dep_ready), lane, task))
        if not candidates:
            raise RuntimeError("Pipeline scheduler deadlocked; check task order and dependencies")

        start, lane, task = min(candidates, key=lambda item: (item[0], item[1]))
        timing = _emit_work_item(trace, acc, cfg, task, start)

        if memory is not None:
            memory.record_task(task, timing)

        if cfg.simulation.emit_flow_events:
            flow_id = _emit_dependency_flows(
                trace,
                logical_stages,
                task,
                timing.recv_start,
                completions,
                completion_lanes,
                flow_id,
            )

        _mark_completions(task, timing.end, completions, completion_lanes)
        lane_ready[lane] = timing.end
        cursors[lane] += 1
        remaining -= 1

    total = max(lane_ready, default=0.0)
    if memory is not None:
        memory.finish(total)
    bubble = max(total * lane_count - acc.occupied, 0.0)
    util = acc.occupied / (total * lane_count) if total > 0 else 0.0
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


def _dependency_ready(
    task: Task,
    logical_stages: int,
    completions: dict[CompletionKey, float],
) -> float | None:
    ready_times: list[float] = []
    for item in _constituent_tasks(task):
        ready = _single_task_dependency_ready(item, logical_stages, completions)
        if ready is None:
            return None
        ready_times.append(ready)
    return max(ready_times, default=0.0)


def _single_task_dependency_ready(
    task: Task,
    logical_stages: int,
    completions: dict[CompletionKey, float],
) -> float | None:
    if task.phase == WEIGHT:
        return 0.0

    stage = task.stage
    mb = task.microbatch
    direction = task.direction
    last_stage = logical_stages - 1

    if task.phase == FORWARD:
        if stage == 0:
            return 0.0
        return completions.get(_completion_key(FORWARD, direction, stage - 1, mb))

    if task.phase == BACKWARD:
        local_forward_done = completions.get(_completion_key(FORWARD, direction, stage, mb))
        if local_forward_done is None:
            return None
        if stage == last_stage:
            return local_forward_done
        downstream_backward_done = completions.get(_completion_key(BACKWARD, direction, stage + 1, mb))
        if downstream_backward_done is None:
            return None
        return max(local_forward_done, downstream_backward_done)

    raise ValueError(f"Unexpected task phase in dependency check: {task.phase}")


def _completion_key(phase: str, direction: int, stage: int, microbatch: int) -> CompletionKey:
    return (phase, direction, stage, microbatch)


def _mark_completions(
    task: Task,
    end: float,
    completions: dict[CompletionKey, float],
    completion_lanes: dict[CompletionKey, int],
) -> None:
    for item in _constituent_tasks(task):
        if item.phase not in {FORWARD, BACKWARD}:
            continue
        key = _completion_key(item.phase, item.direction, item.stage, item.microbatch)
        completions[key] = end
        completion_lanes[key] = _lane(item)


def _emit_work_item(
    trace: Trace,
    acc: Accumulator,
    cfg: Config,
    task: Task,
    start: float,
) -> TaskTiming:
    if task.phase == MOE_BAD_OVERLAP:
        return _emit_moe_bad_overlap_task(trace, acc, cfg, task, start)
    if trace.mode == MOE_BAD_MODE and task.phase in {FORWARD, BACKWARD}:
        return _emit_moe_component_task(trace, acc, cfg, task, start)
    if task.phase == OVERLAP:
        return _emit_overlap_task(trace, acc, cfg, task, start)
    if task.phase == WEIGHT:
        return _emit_weight_task(trace, acc, cfg, task, start)
    return _emit_regular_task(trace, acc, cfg, task, start)


def _emit_regular_task(
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
        _lane(task) + 1,
        _event_name(task, "recv"),
        f"{task.phase}.recv",
        recv_start,
        recv_us,
        **_event_args(trace.mode, cfg, task, "recv"),
    )
    trace.emit(
        _lane(task) + 1,
        _event_name(task, "compute"),
        f"{task.phase}.compute",
        compute_start,
        compute_us,
        **_event_args(trace.mode, cfg, task, "compute"),
    )
    trace.emit(
        _lane(task) + 1,
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


def _emit_weight_task(
    trace: Trace,
    acc: Accumulator,
    cfg: Config,
    task: Task,
    start: float,
) -> TaskTiming:
    _, compute_us, _ = _durations(cfg, task)
    end = start + compute_us
    trace.emit(
        _lane(task) + 1,
        _event_name(task, "compute"),
        "weight.compute",
        start,
        compute_us,
        **_event_args(trace.mode, cfg, task, "compute"),
    )
    acc.add("compute", compute_us)
    return TaskTiming(start, start, end, end)


def _emit_overlap_task(
    trace: Trace,
    acc: Accumulator,
    cfg: Config,
    task: Task,
    start: float,
) -> TaskTiming:
    forward, backward = _overlap_pair(task)
    f_recv, f_compute, f_send = _durations(cfg, forward)
    b_recv, b_compute, b_send = _durations(cfg, backward)
    recv_us = max(f_recv, b_recv)
    compute_us = max(f_compute, b_compute)
    send_us = max(f_send, b_send)
    recv_start = start
    compute_start = recv_start + recv_us
    send_start = compute_start + compute_us
    end = send_start + send_us

    for op, op_start, op_dur in (
        ("recv", recv_start, recv_us),
        ("compute", compute_start, compute_us),
        ("send", send_start, send_us),
    ):
        trace.emit(
            _lane(task) + 1,
            _overlap_event_name(forward, backward, op),
            f"forward_backward_overlap.{op}",
            op_start,
            op_dur,
            **_overlap_args(trace.mode, cfg, forward, backward, op),
        )
        acc.add(op, op_dur)
    return TaskTiming(recv_start, compute_start, send_start, end)


def _emit_moe_component_task(
    trace: Trace,
    acc: Accumulator,
    cfg: Config,
    task: Task,
    start: float,
) -> TaskTiming:
    recv_us, _, send_us = _durations(cfg, task)
    recv_start = start
    component_start = recv_start + recv_us
    t = component_start

    trace.emit(
        _lane(task) + 1,
        _event_name(task, "recv"),
        f"{task.phase}.recv",
        recv_start,
        recv_us,
        **_event_args(trace.mode, cfg, task, "recv"),
    )
    acc.add("recv", recv_us)

    for idx, (name, op, dur) in enumerate(_moe_components(cfg, task)):
        trace.emit(
            _lane(task) + 1,
            _moe_component_event_name(task, name),
            f"moe.{task.phase}.{name}",
            t,
            dur,
            **_moe_component_args(trace.mode, task, idx, name, op, "sequential"),
        )
        acc.add(op, dur)
        t += dur

    send_start = t
    end = send_start + send_us
    trace.emit(
        _lane(task) + 1,
        _event_name(task, "send"),
        f"{task.phase}.send",
        send_start,
        send_us,
        **_event_args(trace.mode, cfg, task, "send"),
    )
    acc.add("send", send_us)
    return TaskTiming(recv_start, component_start, send_start, end)


def _emit_moe_bad_overlap_task(
    trace: Trace,
    acc: Accumulator,
    cfg: Config,
    task: Task,
    start: float,
) -> TaskTiming:
    forward, backward = _overlap_pair(task)
    f_recv, _, f_send = _durations(cfg, forward)
    b_recv, _, b_send = _durations(cfg, backward)
    recv_us = max(f_recv, b_recv)
    send_us = max(f_send, b_send)
    recv_start = start
    component_start = recv_start + recv_us
    t = component_start

    trace.emit(
        _lane(task) + 1,
        _overlap_event_name(forward, backward, "recv"),
        "moe_bad_overlap.pp_recv",
        recv_start,
        recv_us,
        **_moe_bad_pair_args(trace.mode, forward, backward, "pp_recv", None, None, recv_us, 0.0),
    )
    acc.add("recv", recv_us)

    for idx, (forward_component, backward_component) in enumerate(
        zip(_moe_components(cfg, forward), _moe_components(cfg, backward), strict=True)
    ):
        f_name, f_op, f_dur = forward_component
        b_name, b_op, b_dur = backward_component
        overlap_us = min(f_dur, b_dur)
        tail_us = max(f_dur, b_dur) - overlap_us
        if overlap_us > 0:
            trace.emit(
                _lane(task) + 1,
                _moe_bad_pair_event_name(forward, backward, f_name, b_name, "overlap"),
                "moe_bad_overlap.overlap",
                t,
                overlap_us,
                **_moe_bad_pair_args(
                    trace.mode,
                    forward,
                    backward,
                    f"pair_{idx}",
                    forward_component,
                    backward_component,
                    overlap_us,
                    tail_us,
                ),
            )
            acc.add(_moe_pair_accounting_op(f_op, b_op), overlap_us)
        if tail_us > 0:
            long_side = "forward" if f_dur > b_dur else "backward"
            long_name, long_op, _ = forward_component if long_side == "forward" else backward_component
            trace.emit(
                _lane(task) + 1,
                _moe_bad_pair_event_name(forward, backward, f_name, b_name, "bubble"),
                "moe_bad_overlap.bubble",
                t + overlap_us,
                tail_us,
                **_moe_bad_pair_args(
                    trace.mode,
                    forward,
                    backward,
                    f"pair_{idx}",
                    forward_component,
                    backward_component,
                    overlap_us,
                    tail_us,
                    long_side=long_side,
                    long_component=long_name,
                    long_component_op=long_op,
                ),
            )
        t += overlap_us + tail_us

    send_start = t
    end = send_start + send_us
    trace.emit(
        _lane(task) + 1,
        _overlap_event_name(forward, backward, "send"),
        "moe_bad_overlap.pp_send",
        send_start,
        send_us,
        **_moe_bad_pair_args(trace.mode, forward, backward, "pp_send", None, None, send_us, 0.0),
    )
    acc.add("send", send_us)
    return TaskTiming(recv_start, component_start, send_start, end)


class MemoryTracker:
    def __init__(self, trace: Trace, cfg: Config, lane_stage_groups: list[list[int]]):
        self.trace = trace
        self.cfg = cfg
        self.lane_stage_groups = lane_stage_groups
        self.static = [
            sum(_scaled(cfg.memory.static_mb_per_stage, cfg, stage) for stage in stages)
            for stages in lane_stage_groups
        ]
        self.activation = [0.0] * len(lane_stage_groups)
        self.gradient = [0.0] * len(lane_stage_groups)
        self.peak_total = 0.0
        self.peak_stage = 0.0
        for lane in range(len(lane_stage_groups)):
            self._emit_stage(lane, 0.0)
        self._emit_total(0.0)

    def record_task(self, task: Task, timing: TaskTiming) -> None:
        for item in _constituent_tasks(task):
            if item.phase == FORWARD:
                activation_mb = _scaled(self.cfg.memory.activation_mb_per_microbatch, self.cfg, item.stage)
                self._add_activation(_lane(item), activation_mb, timing.compute_start)
            elif item.phase == BACKWARD:
                gradient_mb = _scaled(self.cfg.memory.gradient_mb_per_microbatch, self.cfg, item.stage)
                activation_mb = _scaled(self.cfg.memory.activation_mb_per_microbatch, self.cfg, item.stage)
                self._add_gradient(_lane(item), gradient_mb, timing.recv_start)
                self._add_activation(_lane(item), -activation_mb, timing.send_start)
                self._add_gradient(_lane(item), -gradient_mb, timing.end)

    def finish(self, ts: float) -> None:
        for lane in range(len(self.lane_stage_groups)):
            self._emit_stage(lane, ts)
        self._emit_total(ts)

    def _add_activation(self, lane: int, delta: float, ts: float) -> None:
        self.activation[lane] = max(self.activation[lane] + delta, 0.0)
        self._emit_stage(lane, ts)
        self._emit_total(ts)

    def _add_gradient(self, lane: int, delta: float, ts: float) -> None:
        self.gradient[lane] = max(self.gradient[lane] + delta, 0.0)
        self._emit_stage(lane, ts)
        self._emit_total(ts)

    def _stage_total(self, lane: int) -> float:
        return self.static[lane] + self.activation[lane] + self.gradient[lane]

    def _total(self) -> float:
        return sum(self._stage_total(lane) for lane in range(len(self.lane_stage_groups)))

    def _emit_stage(self, lane: int, ts: float) -> None:
        total = self._stage_total(lane)
        self.peak_stage = max(self.peak_stage, total)
        self.trace.emit_counter(
            _stage_memory_tid(lane),
            f"stage_{lane}_memory_mb",
            ts,
            total,
            static_mb=round(self.static[lane], 3),
            activation_mb=round(self.activation[lane], 3),
            gradient_mb=round(self.gradient[lane], 3),
            logical_stages="+".join(str(stage) for stage in self.lane_stage_groups[lane]),
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


def _constituent_tasks(task: Task) -> list[Task]:
    if task.phase not in {OVERLAP, MOE_BAD_OVERLAP}:
        return [task]
    forward, backward = _overlap_pair(task)
    return [forward, backward]


def _overlap_pair(task: Task) -> tuple[Task, Task]:
    if task.phase not in {OVERLAP, MOE_BAD_OVERLAP} or task.partner is None:
        raise ValueError("paired forward/backward task requires a backward partner")
    forward = Task(FORWARD, task.stage, task.microbatch, task.direction, task.lane)
    return forward, task.partner


def _lane(task: Task) -> int:
    return task.stage if task.lane is None else task.lane


def _scaled(value: float, cfg: Config, stage: int) -> float:
    if not cfg.memory.stage_memory_scale:
        return value
    return value * cfg.memory.stage_memory_scale[stage]


def _memory_thread_names(lane_count: int) -> dict[int, str]:
    names = {_total_memory_tid(): "pipeline total memory"}
    for lane in range(lane_count):
        names[_stage_memory_tid(lane)] = f"stage_{lane} memory"
    return names


def _total_memory_tid() -> int:
    return 900


def _stage_memory_tid(stage: int) -> int:
    return 1000 + stage


def _durations(cfg: Config, task: Task) -> tuple[float, float, float]:
    scale = _stage_scale(cfg, task.stage)
    if task.phase == FORWARD:
        return (
            cfg.timing.forward_recv_us,
            cfg.timing.forward_compute_us * scale,
            cfg.timing.forward_send_us,
        )
    if task.phase == BACKWARD:
        backward_compute = cfg.timing.backward_input_compute_us if task.enable_zb else cfg.timing.backward_compute_us
        return (
            cfg.timing.backward_recv_us,
            backward_compute * scale,
            cfg.timing.backward_send_us,
        )
    if task.phase == WEIGHT:
        return (0.0, cfg.timing.backward_weight_compute_us * scale, 0.0)
    raise ValueError(f"Unexpected phase for duration lookup: {task.phase}")


def _stage_scale(cfg: Config, stage: int) -> float:
    if not cfg.pipeline.stage_compute_scale:
        return 1.0
    return cfg.pipeline.stage_compute_scale[stage]


def _moe_components(cfg: Config, task: Task) -> list[tuple[str, str, float]]:
    scale = _stage_scale(cfg, task.stage)
    if task.phase == FORWARD:
        return [
            ("attn", "compute", cfg.timing.moe_attn_us * scale),
            ("alltoall_dispatch", "send", cfg.timing.moe_alltoall_us),
            ("mlp", "compute", cfg.timing.moe_mlp_us * scale),
            ("alltoall_combine", "send", cfg.timing.moe_alltoall_us),
        ]
    if task.phase == BACKWARD:
        return [
            ("alltoallB_combine", "recv", cfg.timing.moe_backward_alltoall_us),
            ("mlpB", "compute", cfg.timing.moe_backward_mlp_us * scale),
            ("alltoallB_dispatch", "recv", cfg.timing.moe_backward_alltoall_us),
            ("attnB", "compute", cfg.timing.moe_backward_attn_us * scale),
        ]
    raise ValueError(f"MoE component timing is only defined for forward/backward tasks, got {task.phase}")


def _moe_pair_accounting_op(forward_op: str, backward_op: str) -> str:
    if "compute" in {forward_op, backward_op}:
        return "compute"
    if "send" in {forward_op, backward_op}:
        return "send"
    return "recv"


def _moe_component_event_name(task: Task, component: str) -> str:
    prefix = "F" if task.phase == FORWARD else "B"
    return f"MoE_{prefix}_d{task.direction}_mb{task.microbatch}_s{task.stage}_{component}"


def _moe_bad_pair_event_name(forward: Task, backward: Task, forward_component: str, backward_component: str, suffix: str) -> str:
    return (
        f"bad_moe_F{forward.microbatch}_{forward_component}_"
        f"B{backward.microbatch}_{backward_component}_{suffix}_s{forward.stage}"
    )


def _moe_component_args(
    mode: str,
    task: Task,
    order_idx: int,
    component: str,
    op: str,
    policy: str,
) -> dict[str, object]:
    return {
        "strategy": mode,
        "phase": task.phase,
        "microbatch": task.microbatch,
        "stage": task.stage,
        "logical_stage": task.stage,
        "lane": _lane(task),
        "direction": task.direction,
        "component": component,
        "component_index": order_idx,
        "component_op": op,
        "moe_sequence": "forward:attn-alltoall-mlp-alltoall; backward:alltoallB-mlpB-alltoallB-attnB",
        "overlap_policy": policy,
    }


def _moe_bad_pair_args(
    mode: str,
    forward: Task,
    backward: Task,
    pair: str,
    forward_component: tuple[str, str, float] | None,
    backward_component: tuple[str, str, float] | None,
    overlap_us: float,
    bubble_us: float,
    long_side: str | None = None,
    long_component: str | None = None,
    long_component_op: str | None = None,
) -> dict[str, object]:
    f_name, f_op, f_dur = forward_component or ("pp", "recv_send", overlap_us)
    b_name, b_op, b_dur = backward_component or ("pp", "recv_send", overlap_us)
    args: dict[str, object] = {
        "strategy": mode,
        "phase": MOE_BAD_OVERLAP,
        "pair": pair,
        "lane": _lane(forward),
        "forward_microbatch": forward.microbatch,
        "forward_stage": forward.stage,
        "forward_component": f_name,
        "forward_component_op": f_op,
        "forward_component_us": round(f_dur, 3),
        "backward_microbatch": backward.microbatch,
        "backward_stage": backward.stage,
        "backward_component": b_name,
        "backward_component_op": b_op,
        "backward_component_us": round(b_dur, 3),
        "overlap_us": round(overlap_us, 3),
        "uncovered_tail_us": round(bubble_us, 3),
        "overlap_policy": "bad_1f1b_pairwise_min_then_wait_for_long_tail",
        "reason": "long_alltoall_leaves_uncovered_tail" if bubble_us > 0 else "paired_pp_boundary",
    }
    if long_side is not None:
        args.update(
            {
                "long_side": long_side,
                "long_component": long_component,
                "long_component_op": long_component_op,
            }
        )
    return args


def _event_name(task: Task, op: str) -> str:
    prefix = {FORWARD: "F", BACKWARD: "B", WEIGHT: "W"}[task.phase]
    return f"{prefix}_d{task.direction}_mb{task.microbatch}_s{task.stage}_{op}"


def _overlap_event_name(forward: Task, backward: Task, op: str) -> str:
    return (
        f"FB_Fd{forward.direction}_mb{forward.microbatch}_s{forward.stage}_"
        f"Bd{backward.direction}_mb{backward.microbatch}_s{backward.stage}_{op}"
    )


def _event_args(mode: str, cfg: Config, task: Task, op: str) -> dict[str, object]:
    return {
        "strategy": mode,
        "phase": task.phase,
        "microbatch": task.microbatch,
        "stage": task.stage,
        "logical_stage": task.stage,
        "lane": _lane(task),
        "direction": task.direction,
        "direction_label": _direction_label(mode, task.direction),
        "op": op,
        "zero_bubble": task.enable_zb,
        "peer": _peer(cfg, task, op),
    }


def _overlap_args(mode: str, cfg: Config, forward: Task, backward: Task, op: str) -> dict[str, object]:
    return {
        "strategy": mode,
        "phase": "forward_backward_overlap",
        "op": op,
        "overlap_policy": "componentwise_max",
        "lane": _lane(forward),
        "forward_microbatch": forward.microbatch,
        "forward_stage": forward.stage,
        "forward_direction": forward.direction,
        "forward_direction_label": _direction_label(mode, forward.direction),
        "forward_peer": _peer(cfg, forward, op),
        "backward_microbatch": backward.microbatch,
        "backward_stage": backward.stage,
        "backward_direction": backward.direction,
        "backward_direction_label": _direction_label(mode, backward.direction),
        "backward_peer": _peer(cfg, backward, op),
    }


def _direction_label(mode: str, direction: int) -> str:
    if mode == "dualpipe":
        return "first_boundary_input" if direction == 0 else "last_boundary_input"
    if mode == "dualpipev":
        return "v_path"
    return "single_path"


def _peer(cfg: Config, task: Task, op: str) -> str | int:
    last_stage = cfg.pipeline.stages - 1
    if task.phase == FORWARD:
        if op == "recv":
            return f"input_d{task.direction}" if task.stage == 0 else task.stage - 1
        if op == "send":
            return f"loss_d{task.direction}" if task.stage == last_stage else task.stage + 1
        return task.stage

    if task.phase == BACKWARD:
        if op == "recv":
            return f"loss_grad_d{task.direction}" if task.stage == last_stage else task.stage + 1
        if op == "send":
            return f"grad_sink_d{task.direction}" if task.stage == 0 else task.stage - 1
        return task.stage

    if task.phase == WEIGHT:
        return task.stage
    return "overlap"


def _emit_dependency_flows(
    trace: Trace,
    logical_stages: int,
    task: Task,
    recv_start: float,
    completions: dict[CompletionKey, float],
    completion_lanes: dict[CompletionKey, int],
    flow_id: int,
) -> int:
    for item in _constituent_tasks(task):
        if item.phase == WEIGHT:
            continue
        source = _flow_source(item, logical_stages)
        if source is None:
            continue
        src_key, src_stage = source
        src_ts = completions.get(src_key)
        src_lane = completion_lanes.get(src_key)
        if src_ts is None or src_lane is None:
            continue
        dst_lane = _lane(item)
        trace.emit_flow(
            flow_id,
            _flow_name(item, src_stage),
            "activation_flow" if item.phase == FORWARD else "gradient_flow",
            src_lane + 1,
            src_ts,
            dst_lane + 1,
            recv_start,
            phase=item.phase,
            microbatch=item.microbatch,
            direction=item.direction,
            src_stage=src_stage,
            dst_stage=item.stage,
            src_lane=src_lane,
            dst_lane=dst_lane,
        )
        flow_id += 1
    return flow_id


def _flow_source(task: Task, logical_stages: int) -> tuple[CompletionKey, int] | None:
    if task.phase == FORWARD:
        if task.stage == 0:
            return None
        src_stage = task.stage - 1
        return _completion_key(FORWARD, task.direction, src_stage, task.microbatch), src_stage
    if task.phase == BACKWARD:
        if task.stage == logical_stages - 1:
            return None
        src_stage = task.stage + 1
        return _completion_key(BACKWARD, task.direction, src_stage, task.microbatch), src_stage
    return None


def _flow_name(task: Task, src_stage: int) -> str:
    prefix = "F" if task.phase == FORWARD else "B"
    return f"{prefix}_d{task.direction}_mb{task.microbatch}_s{src_stage}_to_s{task.stage}"


def _standard_lane_stage_groups(cfg: Config) -> list[list[int]]:
    return [[stage] for stage in range(cfg.pipeline.stages)]


def _dualpipe_lane_stage_groups(stages: int) -> list[list[int]]:
    return [[rank, stages - 1 - rank] for rank in range(stages)]


def _dualpipe_thread_names(stages: int) -> dict[int, str]:
    return {
        rank + 1: f"rank_{rank} stages_{rank}+{stages - 1 - rank}"
        for rank in range(stages)
    }


def _dualpipev_lane_stage_groups(stages: int) -> list[list[int]]:
    ranks = stages // 2
    return [[rank, stages - 1 - rank] for rank in range(ranks)]


def _dualpipev_thread_names(stages: int) -> dict[int, str]:
    ranks = stages // 2
    return {
        rank + 1: f"rank_{rank} stages_{rank}+{stages - 1 - rank}"
        for rank in range(ranks)
    }
