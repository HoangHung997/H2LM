from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ScanConfig:
    dpi: int = 300
    tile_size: int = 768
    overlap: int = 96
    preview_size: int = 1000
    max_pdf_bytes: int = 134217728
    max_document_pages: int = 5000
    max_selected_pages: int = 20
    max_page_pixels: int = 16000000
    max_total_pixels: int = 180000000
    max_tiles_per_page: int = 64
    max_output_bytes: int = 536870912
    timeout_seconds: int = 180

    def __post_init__(self) -> None:
        bounds = {
            "dpi": (72, 600), "tile_size": (128, 1536), "overlap": (0, 512),
            "preview_size": (128, 2048), "max_pdf_bytes": (1024, 268435456),
            "max_document_pages": (1, 10000), "max_selected_pages": (1, 50),
            "max_page_pixels": (10000, 32000000), "max_total_pixels": (10000, 500000000),
            "max_tiles_per_page": (1, 256), "max_output_bytes": (1024, 1073741824),
            "timeout_seconds": (1, 600),
        }
        for key, (lo, hi) in bounds.items():
            value = getattr(self, key)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"{key} must be an integer in [{lo}, {hi}]")
        if self.overlap >= self.tile_size:
            raise ValueError("overlap must be smaller than tile_size")

    @classmethod
    def load(cls, path: str | Path) -> ScanConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("Scan config must be a mapping")
        if set(raw) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown scan config key")
        return cls(**raw)


def tiles(width: int, height: int, size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Pixel-edge boxes [left, top, right, bottom), no resizing or dropped margins."""
    if any(type(v) is not int for v in (width, height, size, overlap)):
        raise ValueError("Tile dimensions must be integers")
    if min(width, height, size) < 1 or not 0 <= overlap < size:
        raise ValueError("Invalid tile dimensions/overlap")

    def starts(length: int) -> list[int]:
        if length <= size:
            return [0]
        points = list(range(0, length - size + 1, size - overlap))
        if points[-1] != length - size:
            points.append(length - size)
        return points

    return [(x, y, min(x + size, width), min(y + size, height))
            for y in starts(height) for x in starts(width)]
