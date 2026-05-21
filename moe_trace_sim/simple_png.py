from __future__ import annotations

import struct
import zlib
from pathlib import Path

Color = tuple[int, int, int]


class Canvas:
    def __init__(self, width: int, height: int, bg: Color = (255, 255, 255)):
        self.width = width
        self.height = height
        self.pixels = [[bg for _ in range(width)] for _ in range(height)]

    def rect(self, x: int, y: int, w: int, h: int, color: Color) -> None:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + w), min(self.height, y + h)
        for yy in range(y0, y1):
            row = self.pixels[yy]
            for xx in range(x0, x1):
                row[xx] = color

    def line_h(self, x: int, y: int, w: int, color: Color) -> None:
        self.rect(x, y, w, 1, color)

    def write(self, path: str | Path) -> None:
        raw = bytearray()
        for row in self.pixels:
            raw.append(0)
            for r, g, b in row:
                raw.extend((r, g, b))
        png = bytearray(b"\x89PNG\r\n\x1a\n")
        png.extend(_chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)))
        png.extend(_chunk(b"IDAT", zlib.compress(bytes(raw), 9)))
        png.extend(_chunk(b"IEND", b""))
        Path(path).write_bytes(bytes(png))


def _chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
