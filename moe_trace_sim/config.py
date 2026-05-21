from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    num_tokens: int = 4096
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_experts: int = 64
    top_k: int = 2
    dtype_bytes: int = 2


@dataclass
class RoutingConfig:
    distribution: str = "uniform"
    zipf_alpha: float = 1.2
    seed: int = 42
    capacity_factor: float = 1.25
    hotspot_experts: int = 4
    hotspot_fraction: float = 0.75


@dataclass
class HardwareConfig:
    peak_tflops: float = 100.0
    mem_bandwidth_GBs: float = 1500.0
    launch_overhead_us: float = 8.0
    small_gemm_penalty: float = 0.55
    min_gemm_m_for_good_util: int = 256
    max_parallel_streams: int = 4


@dataclass
class SimulationConfig:
    modes: list[str] = field(
        default_factory=lambda: ["baseline_unfused", "grouped_gemm", "mega_fused"]
    )
    trace_unit: str = "us"
    emit_memory_events: bool = True
    emit_idle_events: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    hardware_model: HardwareConfig = field(default_factory=HardwareConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    experiment: dict[str, Any] = field(default_factory=dict)
    assumptions: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> Config:
    raw = _load_mapping(Path(path))
    return Config(
        model=ModelConfig(**raw.get("model", {})),
        routing=RoutingConfig(**raw.get("routing", {})),
        hardware_model=HardwareConfig(**raw.get("hardware_model", {})),
        simulation=SimulationConfig(**raw.get("simulation", {})),
        experiment=raw.get("experiment", {}),
        assumptions=raw.get("assumptions", {}),
    )


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

    for raw_line in text.splitlines():
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
            next_container: Any = [] if _next_nonempty_is_list(text, raw_line) else {}
            parent[key] = next_container
            stack.append((indent, next_container))

    return root


def _next_nonempty_is_list(text: str, current_line: str) -> bool:
    lines = text.splitlines()
    try:
        idx = lines.index(current_line)
    except ValueError:
        return False
    current_indent = len(current_line) - len(current_line.lstrip(" "))
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
