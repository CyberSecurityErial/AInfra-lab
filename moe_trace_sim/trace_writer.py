from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IDLE_TID_BASE = 100


@dataclass
class TraceEvent:
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


class Trace:
    def __init__(self, mode: str, pid: int, stream_count: int = 4):
        self.mode = mode
        self.pid = pid
        self.stream_count = stream_count
        self.events: list[TraceEvent] = []

    def emit(self, tid: int, name: str, cat: str, start: float, dur: float, **args: Any) -> None:
        self.events.append(TraceEvent(name, cat, self.pid, tid, start, dur, {"mode": self.mode, **args}))

    def write(self, path: str | Path) -> None:
        metadata = [
            {"ph": "M", "name": "process_name", "pid": self.pid, "args": {"name": self.mode}},
            {"ph": "M", "name": "thread_name", "pid": self.pid, "tid": 1, "args": {"name": "CPU/framework"}},
            {"ph": "M", "name": "thread_name", "pid": self.pid, "tid": 2, "args": {"name": "GPU memory"}},
        ]
        for stream in range(self.stream_count):
            metadata.append(
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": self.pid,
                    "tid": 3 + stream,
                    "args": {"name": f"GPU compute stream {stream}"},
                }
            )
            metadata.append(
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": self.pid,
                    "tid": IDLE_TID_BASE + stream,
                    "args": {"name": f"GPU idle stream {stream}"},
                }
            )
        payload = {"traceEvents": metadata + [e.to_json() for e in self.events]}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
