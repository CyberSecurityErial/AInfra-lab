from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CompleteEvent:
    name: str
    cat: str
    pid: int
    tid: int
    ts: float
    dur: float
    args: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cat": self.cat,
            "ph": "X",
            "pid": self.pid,
            "tid": self.tid,
            "ts": round(self.ts, 3),
            "dur": round(max(self.dur, 0.001), 3),
            "args": self.args,
        }


@dataclass
class FlowEvent:
    name: str
    cat: str
    pid: int
    tid: int
    ts: float
    phase: str
    flow_id: int
    args: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cat": self.cat,
            "ph": self.phase,
            "pid": self.pid,
            "tid": self.tid,
            "ts": round(self.ts, 3),
            "id": self.flow_id,
            "args": self.args,
        }


@dataclass
class CounterEvent:
    name: str
    cat: str
    pid: int
    tid: int
    ts: float
    args: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cat": self.cat,
            "ph": "C",
            "pid": self.pid,
            "tid": self.tid,
            "ts": round(self.ts, 3),
            "args": self.args,
        }


class Trace:
    def __init__(self, mode: str, pid: int, stage_count: int, extra_threads: dict[int, str] | None = None):
        self.mode = mode
        self.pid = pid
        self.stage_count = stage_count
        self.extra_threads = extra_threads or {}
        self.events: list[CompleteEvent | FlowEvent | CounterEvent] = []

    def emit(self, tid: int, name: str, cat: str, start: float, dur: float, **args: Any) -> None:
        self.events.append(CompleteEvent(name, cat, self.pid, tid, start, dur, {"mode": self.mode, **args}))

    def emit_flow(
        self,
        flow_id: int,
        name: str,
        cat: str,
        src_tid: int,
        src_ts: float,
        dst_tid: int,
        dst_ts: float,
        **args: Any,
    ) -> None:
        flow_args = {"mode": self.mode, **args}
        self.events.append(FlowEvent(name, cat, self.pid, src_tid, src_ts, "s", flow_id, flow_args))
        self.events.append(FlowEvent(name, cat, self.pid, dst_tid, dst_ts, "t", flow_id, flow_args))

    def emit_counter(self, tid: int, name: str, start: float, value: float, **args: Any) -> None:
        counter_args = {"memory_mb": round(value, 3), **args}
        self.events.append(CounterEvent(name, "memory", self.pid, tid, start, counter_args))

    def write(self, path: str | Path) -> None:
        metadata = [
            {"ph": "M", "name": "process_name", "pid": self.pid, "args": {"name": self.mode}},
        ]
        for stage in range(self.stage_count):
            metadata.append(
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": self.pid,
                    "tid": stage + 1,
                    "args": {"name": f"stage_{stage}"},
                }
            )
        for tid, name in sorted(self.extra_threads.items()):
            metadata.append(
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": self.pid,
                    "tid": tid,
                    "args": {"name": name},
                }
            )
        events = sorted(self.events, key=lambda event: (event.ts, event.tid))
        payload = {"traceEvents": metadata + [event.to_json() for event in events]}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
