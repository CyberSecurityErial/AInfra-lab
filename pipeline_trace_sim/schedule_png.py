from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from moe_trace_sim.simple_png import Canvas, Color

from .scheduler import PipelineSummary
from .trace_writer import CompleteEvent, Trace

_BG = (250, 251, 252)
_PANEL = (255, 255, 255)
_INK = (28, 34, 43)
_MUTED = (91, 103, 117)
_GRID = (225, 230, 236)
_LANE_A = (245, 247, 250)
_LANE_B = (239, 243, 247)
_IDLE = (252, 229, 231)
_FWD = (42, 119, 190)
_BWD = (219, 132, 49)
_WEIGHT = (68, 151, 91)
_OVERLAP = (126, 91, 184)
_MOE_PAIR = (56, 145, 151)
_MOE_FWD = (51, 132, 191)
_MOE_BWD = (205, 111, 57)
_BUBBLE = (205, 63, 70)
_BORDER = (166, 176, 189)


@dataclass(frozen=True)
class _LaneEvent:
    tid: int
    name: str
    cat: str
    start: float
    dur: float
    args: dict[str, object]

    @property
    def end(self) -> float:
        return self.start + self.dur


def write_schedule_png(path: str | Path, trace: Trace, summary: PipelineSummary) -> None:
    events = _complete_events(trace)
    width = 1248
    if not events:
        canvas = Canvas(width, 180, _BG)
        _draw_text(canvas, 20, 20, f"{trace.mode} SCHEDULE", _INK, 2)
        _draw_text(canvas, 20, 54, "NO COMPLETE EVENTS", _MUTED, 1)
        canvas.write(path)
        return

    tids = sorted({event.tid for event in events})
    lane_names = _lane_names(trace, tids)
    lane_count = len(tids)
    total = max(max(event.end for event in events), summary.total_us, 1.0)

    left = 172
    right = 36
    top = 96
    lane_h = 30
    lane_gap = 12
    plot_w = width - left - right
    plot_h = lane_count * lane_h + (lane_count - 1) * lane_gap
    bottom = 96
    height = top + plot_h + bottom

    canvas = Canvas(width, height, _BG)
    _draw_text(canvas, 20, 18, f"{_mode_title(trace.mode)} SCHEDULE", _INK, 2)
    _draw_text(
        canvas,
        20,
        46,
        (
            f"TOTAL {_format_us(summary.total_us)}  "
            f"UTIL {summary.utilization * 100:.1f}%  "
            f"BUBBLE {_format_us(summary.bubble_us)}  "
            f"EVENTS {summary.event_count}"
        ),
        _MUTED,
        1,
    )

    canvas.rect(left, top - 24, plot_w, plot_h + 24, _PANEL)
    _outline(canvas, left, top - 24, plot_w, plot_h + 24, _GRID)
    _draw_axis(canvas, left, top, plot_w, plot_h, total)

    by_tid: dict[int, list[_LaneEvent]] = {tid: [] for tid in tids}
    for event in events:
        by_tid[event.tid].append(event)

    for lane_idx, tid in enumerate(tids):
        y = top + lane_idx * (lane_h + lane_gap)
        lane_bg = _LANE_A if lane_idx % 2 == 0 else _LANE_B
        canvas.rect(left, y, plot_w, lane_h, lane_bg)
        _draw_text(canvas, 20, y + 10, lane_names[tid], _INK, 1)

        lane_events = sorted(by_tid[tid], key=lambda item: (item.start, item.end, item.cat))
        _draw_idle_gaps(canvas, lane_events, left, y, plot_w, lane_h, total)
        for event in lane_events:
            x = _x(event.start, left, plot_w, total)
            w = max(1, _x(event.end, left, plot_w, total) - x)
            color = _color(event)
            canvas.rect(x, y + 4, w, lane_h - 8, color)
            _outline(canvas, x, y + 4, w, lane_h - 8, _darken(color))
            label = _event_label(event)
            if label and _text_width(label, 1) <= w - 6:
                _draw_text(canvas, x + 3, y + 11, label, _text_color(color), 1)

    _draw_legend(canvas, left, top + plot_h + 30)
    canvas.write(path)


def _complete_events(trace: Trace) -> list[_LaneEvent]:
    return [
        _LaneEvent(event.tid, event.name, event.cat, event.ts, event.dur, event.args)
        for event in trace.events
        if isinstance(event, CompleteEvent)
    ]


def _lane_names(trace: Trace, tids: list[int]) -> dict[int, str]:
    if trace.thread_names is None:
        names = {stage + 1: f"stage_{stage}" for stage in range(trace.stage_count)}
    else:
        names = dict(trace.thread_names)
    return {tid: _short_lane_name(names.get(tid, f"tid_{tid}")) for tid in tids}


def _short_lane_name(name: str) -> str:
    if name.startswith("rank_") and " stages_" in name:
        rank, stages = name.split(" stages_", 1)
        rank_id = rank.removeprefix("rank_")
        return f"RANK {rank_id} S{stages.replace('+', '+')}"
    if name.startswith("stage_"):
        return f"STAGE {name.removeprefix('stage_')}"
    return name.replace("_", " ").upper()


def _draw_axis(canvas: Canvas, left: int, top: int, plot_w: int, plot_h: int, total: float) -> None:
    for idx in range(5):
        ts = total * idx / 4
        x = _x(ts, left, plot_w, total)
        canvas.rect(x, top - 10, 1, plot_h + 10, _GRID)
        label = _format_us(ts)
        label_w = _text_width(label, 1)
        label_x = max(left, min(left + plot_w - label_w, x - label_w // 2))
        _draw_text(canvas, label_x, top - 21, label, _MUTED, 1)


def _draw_idle_gaps(
    canvas: Canvas,
    lane_events: list[_LaneEvent],
    left: int,
    y: int,
    plot_w: int,
    lane_h: int,
    total: float,
) -> None:
    cursor = 0.0
    for event in lane_events:
        if event.start > cursor + 0.001:
            _draw_gap(canvas, cursor, event.start, left, y, plot_w, lane_h, total)
        cursor = max(cursor, event.end)
    if total > cursor + 0.001:
        _draw_gap(canvas, cursor, total, left, y, plot_w, lane_h, total)


def _draw_gap(
    canvas: Canvas,
    start: float,
    end: float,
    left: int,
    y: int,
    plot_w: int,
    lane_h: int,
    total: float,
) -> None:
    x = _x(start, left, plot_w, total)
    w = max(1, _x(end, left, plot_w, total) - x)
    canvas.rect(x, y + 4, w, lane_h - 8, _IDLE)


def _draw_legend(canvas: Canvas, x: int, y: int) -> None:
    items = [
        ("FWD", _FWD),
        ("BWD", _BWD),
        ("W", _WEIGHT),
        ("F/B OVERLAP", _OVERLAP),
        ("MOE PAIR", _MOE_PAIR),
        ("IDLE/BUBBLE", _IDLE),
        ("A2A TAIL", _BUBBLE),
    ]
    cursor = x
    for label, color in items:
        canvas.rect(cursor, y + 1, 16, 10, color)
        _outline(canvas, cursor, y + 1, 16, 10, _darken(color))
        _draw_text(canvas, cursor + 22, y + 3, label, _INK, 1)
        cursor += 22 + _text_width(label, 1) + 28


def _x(ts: float, left: int, plot_w: int, total: float) -> int:
    return left + int(round((ts / total) * plot_w))


def _color(event: _LaneEvent) -> Color:
    cat = event.cat
    if cat == "moe_bad_overlap.bubble":
        return _BUBBLE
    if cat.startswith("moe_bad_overlap"):
        return _MOE_PAIR
    if cat.startswith("forward_backward_overlap"):
        return _OVERLAP
    if cat.startswith("moe.forward"):
        return _MOE_FWD
    if cat.startswith("moe.backward"):
        return _MOE_BWD
    if cat.startswith("forward"):
        return _FWD
    if cat.startswith("backward"):
        return _BWD
    if cat.startswith("weight"):
        return _WEIGHT
    return (118, 128, 142)


def _event_label(event: _LaneEvent) -> str:
    args = event.args
    cat = event.cat
    if cat == "moe_bad_overlap.bubble":
        return "TAIL"
    if cat.startswith("forward_backward_overlap") or cat.startswith("moe_bad_overlap"):
        f_mb = args.get("forward_microbatch")
        b_mb = args.get("backward_microbatch")
        if f_mb is not None and b_mb is not None:
            return f"F{f_mb}/B{b_mb}"
        return "F/B"

    phase = args.get("phase")
    mb = args.get("microbatch")
    if phase == "forward" and mb is not None:
        return f"F{mb}"
    if phase == "backward" and mb is not None:
        return f"B{mb}"
    if phase == "weight" and mb is not None:
        return f"W{mb}"
    return ""


def _mode_title(mode: str) -> str:
    return mode.replace("_", " ").upper()


def _format_us(value: float) -> str:
    if value >= 100 or abs(value - round(value)) < 0.01:
        return f"{value:.0f}US"
    return f"{value:.1f}US"


def _outline(canvas: Canvas, x: int, y: int, w: int, h: int, color: Color) -> None:
    if w <= 0 or h <= 0:
        return
    canvas.rect(x, y, w, 1, color)
    canvas.rect(x, y + h - 1, w, 1, color)
    canvas.rect(x, y, 1, h, color)
    canvas.rect(x + w - 1, y, 1, h, color)


def _darken(color: Color) -> Color:
    return tuple(max(0, int(channel * 0.72)) for channel in color)  # type: ignore[return-value]


def _text_color(color: Color) -> Color:
    r, g, b = color
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return _INK if luminance > 168 else (255, 255, 255)


def _text_width(text: str, scale: int) -> int:
    if not text:
        return 0
    return len(text) * 6 * scale - scale


def _draw_text(canvas: Canvas, x: int, y: int, text: str, color: Color, scale: int) -> None:
    cursor = x
    for char in text.upper():
        glyph = _FONT.get(char, _FONT["?"])
        for row_idx, row in enumerate(glyph):
            for col_idx, bit in enumerate(row):
                if bit == "1":
                    canvas.rect(cursor + col_idx * scale, y + row_idx * scale, scale, scale, color)
        cursor += 6 * scale


_FONT: dict[str, list[str]] = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "?": ["01110", "10001", "00001", "00010", "00100", "00000", "00100"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00100", "00100"],
    ":": ["00000", "00100", "00100", "00000", "00100", "00100", "00000"],
    "%": ["11001", "11010", "00010", "00100", "01000", "01011", "10011"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
}
