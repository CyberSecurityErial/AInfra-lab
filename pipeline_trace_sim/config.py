from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    stages: int = 4
    microbatches: int = 8
    stage_compute_scale: list[float] = field(default_factory=list)


@dataclass
class TimingConfig:
    forward_recv_us: float = 10.0
    forward_compute_us: float = 80.0
    forward_send_us: float = 10.0
    backward_recv_us: float = 10.0
    backward_compute_us: float = 120.0
    backward_send_us: float = 10.0


@dataclass
class MemoryConfig:
    static_mb_per_stage: float = 512.0
    activation_mb_per_microbatch: float = 256.0
    gradient_mb_per_microbatch: float = 128.0
    stage_memory_scale: list[float] = field(default_factory=list)


@dataclass
class SimulationConfig:
    modes: list[str] = field(default_factory=lambda: ["gpipe", "1f1b"])
    trace_unit: str = "us"
    emit_flow_events: bool = True
    emit_memory_counters: bool = True
    write_memory_trace: bool = True


@dataclass
class Config:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    experiment: dict[str, Any] = field(default_factory=dict)
    assumptions: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> Config:
    raw = _load_mapping(Path(path))
    cfg = Config(
        pipeline=PipelineConfig(**raw.get("pipeline", {})),
        timing=TimingConfig(**raw.get("timing", {})),
        memory=MemoryConfig(**raw.get("memory", {})),
        simulation=SimulationConfig(**raw.get("simulation", {})),
        experiment=raw.get("experiment", {}),
        assumptions=raw.get("assumptions", {}),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.pipeline.stages <= 0:
        raise ValueError("pipeline.stages must be positive")
    if cfg.pipeline.microbatches <= 0:
        raise ValueError("pipeline.microbatches must be positive")
    if cfg.pipeline.stage_compute_scale and len(cfg.pipeline.stage_compute_scale) != cfg.pipeline.stages:
        raise ValueError("pipeline.stage_compute_scale must either be empty or match pipeline.stages")
    if cfg.memory.stage_memory_scale and len(cfg.memory.stage_memory_scale) != cfg.pipeline.stages:
        raise ValueError("memory.stage_memory_scale must either be empty or match pipeline.stages")
    valid_modes = {"gpipe", "1f1b"}
    unknown = [mode for mode in cfg.simulation.modes if mode not in valid_modes]
    if unknown:
        raise ValueError(f"Unknown simulation mode(s): {', '.join(unknown)}")
    for name, value in vars(cfg.timing).items():
        if value < 0:
            raise ValueError(f"timing.{name} must be non-negative")
    for name, value in vars(cfg.memory).items():
        if isinstance(value, list):
            if any(item < 0 for item in value):
                raise ValueError(f"memory.{name} must be non-negative")
        elif value < 0:
            raise ValueError(f"memory.{name} must be non-negative")


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded or {}
    except ModuleNotFoundError:
        return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by this lab's config files."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    lines = text.splitlines()

    for idx, raw_line in enumerate(lines):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if stripped.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"List item without list parent: {raw_line}")
            parent.append(_coerce_value(stripped[2:].strip()))
            continue

        key, sep, value = stripped.partition(":")
        if not sep:
            raise ValueError(f"Invalid config line: {raw_line}")
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _coerce_value(value)
        else:
            next_container: Any = [] if _next_nonempty_is_list(lines, idx, indent) else {}
            parent[key] = next_container
            stack.append((indent, next_container))

    return root


def _next_nonempty_is_list(lines: list[str], idx: int, current_indent: int) -> bool:
    for line in lines[idx + 1 :]:
        clean = line.split("#", 1)[0].rstrip()
        if not clean.strip():
            continue
        indent = len(clean) - len(clean.lstrip(" "))
        return indent > current_indent and clean.strip().startswith("- ")
    return False


def _coerce_value(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")
