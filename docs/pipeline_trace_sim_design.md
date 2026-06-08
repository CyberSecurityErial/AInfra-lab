# Pipeline Trace Simulator Design

This lab is a CPU-only discrete-event simulator for visualizing pipeline training schedules. It emits Chrome Trace JSON that can be opened in Perfetto or Chrome trace viewer, plus static PNG schedule diagrams for quick comparison.

It does not run a model, CUDA kernels, NCCL, or a training framework. The goal is to make scheduling policy differences visible.

## Model

The simulator has:

- `stages`: physical pipeline rank lanes for regular schedules.
- `microbatches`: microbatches in one training step.
- `interleaved_virtual_chunks`: virtual model chunks per physical rank for interleaved 1F1B.
- `timing`: synthetic forward/backward recv, compute, and send durations in microseconds.
- `backward_input_compute_us` and `backward_weight_compute_us`: a synthetic split used when ZeroBubble-style delayed weight-gradient work is modeled.
- MoE component timing: `moe_attn_us`, `moe_alltoall_us`, `moe_mlp_us`, `moe_backward_alltoall_us`, `moe_backward_mlp_us`, and `moe_backward_attn_us`.
- `stage_compute_scale`: optional per-logical-stage multiplier for compute events.
- `memory`: synthetic static stage memory, retained activation memory, transient gradient memory, and optional per-stage memory multipliers.

For `gpipe` and `1f1b`, each logical stage has one timeline lane. For `interleaved_1f1b`, each physical rank lane hosts `interleaved_virtual_chunks` virtual stages spaced by the physical stage count. For `chimera` and `dualpipe`, each physical rank lane hosts a normal stage and its mirrored stage. For `dualpipev`, the logical pipeline is folded onto `stages / 2` physical rank lanes.

Most tasks emit `recv -> compute -> send`. Delayed weight-gradient tasks emit only `weight.compute`. ZeroBubble backward-input tasks use `backward_input_compute_us`; their deferred weight-gradient tasks use `backward_weight_compute_us`. Paired DualPipe/DualPipeV chunks emit `forward_backward_overlap.recv -> compute -> send`; each component duration is the max of the paired forward and backward component durations. The deliberate bad MoE overlap mode emits component-level events and marks uncovered all-to-all tails as `moe_bad_overlap.bubble`.

Memory counters are emitted as Chrome Trace counter events (`ph: "C"`). Each strategy trace includes one pipeline total memory counter and one per-lane memory counter. Counter event args carry static, activation, gradient, and hosted logical-stage components for inspection.

Forward dependencies:

- Logical stage 0 receives from an input boundary.
- Logical stage `s > 0` can receive microbatch `m` after stage `s - 1` has sent it on the same modeled path/direction.
- The last logical stage sends to a loss boundary.

Backward dependencies:

- The last logical stage receives gradient from a loss boundary after its local forward work is complete.
- Logical stage `s < last` can receive gradient for microbatch `m` after stage `s + 1` has sent it.
- Each stage sends gradients upstream, with logical stage 0 sending to a gradient sink boundary.

## Strategies

`gpipe` schedules all forward tasks on every stage before backward tasks. The last stage starts backward only after all last-stage forward sends are complete, creating the visible flush barrier.

`1f1b` builds a local order for each stage:

1. Warm up with `stages - stage - 1` forward microbatches, capped by total microbatches.
2. Alternate one forward task and one backward task.
3. Drain remaining backward tasks.

`interleaved_1f1b` models virtual pipeline chunks per physical rank:

- The logical pipeline depth is `stages * interleaved_virtual_chunks`.
- Physical rank `r` hosts virtual stages `r`, `r + stages`, and so on.
- Each hosted virtual stage gets a chunk-local 1F1B queue, and the lane scheduler selects runnable work across those queues.
- Compute and activation/gradient memory are scaled by `1 / interleaved_virtual_chunks` per virtual chunk so total local model work remains comparable to regular 1F1B.
- Pipeline recv/send events remain visible at every virtual-stage boundary, so the trace shows the extra communication boundaries introduced by chunking.

`zerobubble_1f1b` keeps the normal 1F1B critical-path order but splits backward work:

- `B`: backward input-gradient work. It receives downstream gradient, computes `dX`, sends upstream gradient, and unlocks the previous pipeline stage.
- `W`: backward weight-gradient work. It is queued after local `B` completes and is emitted as `weight.compute`. It does not unlock another stage, so the scheduler uses it only when that lane has no immediately runnable critical-path task.

This preserves the simplified synchronous-step model: all `W` tasks are drained before the simulated step finishes, and no optimizer update is modeled before they complete.

`moe_bad_overlap_1f1b` is a deliberately poor comparison schedule for MoE-heavy 1F1B. It keeps the normal 1F1B microbatch order, but steady-state work pairs one forward chunk with one backward chunk using this naive component map:

| window | forward component | backward component | modeled problem |
|---|---|---|---|
| 1 | `attn` | `alltoallB_combine` | backward all-to-all is much longer than attention |
| 2 | `alltoall_dispatch` | `mlpB` | forward all-to-all is much longer than MLP backward |
| 3 | `mlp` | `alltoallB_dispatch` | backward all-to-all is much longer than MLP |
| 4 | `alltoall_combine` | `attnB` | forward all-to-all is much longer than attention backward |

For each window, the shorter duration is emitted as `moe_bad_overlap.overlap`; the remaining long all-to-all tail is emitted as `moe_bad_overlap.bubble` and does not count as useful lane occupancy in `bubble_us`. This is intended as a teaching counterexample: trying to hide long MoE all-to-all with small compute blocks leaves exposed waits.

`chimera` models the bidirectional-pipeline baseline that predates DualPipe-style paired chunk overlap:

- The configured microbatches are split evenly across two opposite 1F1B pipelines.
- Direction 0 runs through logical stages `0 -> stages - 1`; direction 1 is placed on mirrored rank lanes using logical stages `stages - 1 -> 0`.
- Each physical rank lane hosts logical stages `rank` and `stages - 1 - rank`.
- The lane scheduler selects runnable work from either direction, but emits ordinary `recv -> compute -> send` tasks rather than DualPipe componentwise-overlapped forward/backward chunks.
- The model keeps a synchronous flush before step completion, so it avoids stale weights in this teaching abstraction.

`dualpipe` translates the public DeepSeek DualPipe eight-step loop into trace tasks:

- Microbatches are split into two halves.
- Direction 0 enters from the first boundary and direction 1 enters from the last boundary.
- Each physical rank lane hosts logical stages `rank` and `stages - 1 - rank`, matching the mirrored-module placement.
- Main-loop paired chunks are modeled as componentwise-overlapped forward/backward work.
- ZeroBubble-style backward chunks use `backward_input_compute_us` and enqueue later `weight.compute` tasks.

`dualpipev` translates the public DualPipeV eight-step loop into a V-shaped layout:

- `stages` must be even.
- Physical rank count is `stages / 2`.
- Rank `r` hosts logical stages `r` and `stages - 1 - r`.
- The forward path runs up the first side of the V and then back down the mirrored side; backward dependencies reverse that logical path.
- Paired overlap and delayed weight-gradient chunks use the same synthetic timing rules as `dualpipe`.

The event scheduler resolves inter-stage dependencies and timestamps after each strategy builds per-lane task order.

## Memory Model

The model is intentionally simple and tied to scheduling moments:

- Static memory is present on every physical lane at timestamp 0.
- A lane's static memory is the sum of the logical stages hosted by that lane.
- Forward compute allocates retained activation memory for that logical stage and microbatch.
- Backward recv allocates a transient gradient buffer.
- Backward compute releases the retained activation after local compute completes.
- Backward send completion releases the transient gradient buffer.
- Delayed weight-gradient compute does not change activation or gradient counters.

This makes the difference between strategies visible: GPipe accumulates activations through the forward flush, 1F1B starts releasing earlier during steady-state alternation, Interleaved 1F1B trades virtual-stage boundaries for shorter chunk-local work and fewer visible idle gaps, ZeroBubble moves non-critical `W` work into idle slots, the bad MoE overlap exposes communication tails as bubble, Chimera and DualPipe duplicate mirrored modules per rank, and DualPipeV folds logical stages onto fewer lanes.

## Output

The CLI writes one trace file per requested strategy, for example:

- `gpipe_trace.json`
- `gpipe_schedule.png`
- `1f1b_trace.json`
- `1f1b_schedule.png`
- `interleaved_1f1b_trace.json`
- `interleaved_1f1b_schedule.png`
- `zerobubble_1f1b_trace.json`
- `zerobubble_1f1b_schedule.png`
- `moe_bad_overlap_1f1b_trace.json`
- `moe_bad_overlap_1f1b_schedule.png`
- `chimera_trace.json`
- `chimera_schedule.png`
- `dualpipe_trace.json`
- `dualpipe_schedule.png`
- `dualpipev_trace.json`
- `dualpipev_schedule.png`
- `memory_trace.json`

It also writes `summary.csv` and `report.md`; the report embeds the schedule PNGs. Summary `bubble_us` is calculated as total physical-lane capacity time minus occupied recv/compute/send time. `peak_memory_mb` reports the maximum pipeline total memory observed in the counter stream.
