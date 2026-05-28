# Pipeline Trace Simulator Design

This lab is a CPU-only discrete-event simulator for visualizing pipeline training schedules. It emits Chrome Trace JSON that can be opened in Perfetto or Chrome trace viewer.

It does not run a model, CUDA kernels, NCCL, or a training framework. The goal is to make scheduling policy differences visible.

## Model

The simulator has:

- `stages`: pipeline stages.
- `microbatches`: microbatches in one training step.
- `timing`: synthetic forward/backward recv, compute, and send durations in microseconds.
- `stage_compute_scale`: optional per-stage multiplier for compute events.
- `memory`: synthetic static stage memory, retained activation memory, transient gradient memory, and optional per-stage memory multipliers.

Each stage has one timeline lane. Each task emits `recv -> compute -> send` on that lane.

Memory counters are emitted as Chrome Trace counter events (`ph: "C"`). Each strategy trace includes one pipeline total memory counter and one per-stage total memory counter. Counter event args carry the static, activation, and gradient components for inspection.

Forward dependencies:

- Stage 0 receives from an input boundary.
- Stage `s > 0` can receive microbatch `m` after stage `s - 1` has sent it.
- Each stage sends activations to the next stage, with the last stage sending to a loss boundary.

Backward dependencies:

- The last stage receives gradient from a loss boundary.
- Stage `s < last` can receive gradient for microbatch `m` after stage `s + 1` has sent it.
- Each stage sends gradients upstream, with stage 0 sending to a gradient sink boundary.

## Strategies

`gpipe` schedules all forward tasks on every stage before backward tasks. The last stage starts backward only after all last-stage forward sends are complete, creating the visible flush barrier.

`1f1b` builds a local order for each stage:

1. Warm up with `stages - stage - 1` forward microbatches, capped by total microbatches.
2. Alternate one forward task and one backward task.
3. Drain remaining backward tasks.

The event scheduler then resolves inter-stage dependencies and timestamps.

## Memory Model

The model is intentionally simple and tied to scheduling moments:

- Static memory is present on every stage at timestamp 0.
- Forward compute allocates retained activation memory for that stage and microbatch.
- Backward recv allocates a transient gradient buffer.
- Backward compute releases the retained activation after local compute completes.
- Backward send completion releases the transient gradient buffer.

This makes the difference between GPipe and 1F1B visible: GPipe accumulates activations through the forward flush before backward releases them, while 1F1B starts releasing earlier during the steady-state alternation.

## Output

The CLI writes one trace file per strategy:

- `gpipe_trace.json`
- `1f1b_trace.json`
- `memory_trace.json`

It also writes `summary.csv` and `report.md`. Summary `bubble_us` is calculated as total stage capacity time minus occupied recv/compute/send time. `peak_memory_mb` reports the maximum pipeline total memory observed in the counter stream.
