"""Tiny generated TWO-PAGE Vietnamese example: engineering input, not a document dataset."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


def encode(text: str) -> list[int]:
    return [10 + value for value in text.encode("utf-8")]


def decode(ids: list[int]) -> str:
    if any(not 10 <= value < 266 for value in ids):
        raise ValueError("Invalid byte-codec token")
    return bytes(value - 10 for value in ids).decode("utf-8", errors="strict")


def example(index: int, max_pages: int, side: int = 128) -> dict:
    if type(index) is not int or not 0 <= index < 64 or not 32 <= side <= 256:
        raise ValueError("Invalid smoke fixture parameters")
    candidates = [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                  Path("C:/Windows/Fonts/arial.ttf")]
    font = next((p for p in candidates if p.is_file()), None)
    if font is None:
        raise ValueError("A local Vietnamese-capable font is required; no fonts are bundled")
    numbers = [21 + index, 65 - index]
    tiles, hashes = [], []
    for page, number in enumerate(numbers, 1):
        with Image.new("RGB", (side, side), "white") as image:
            painter = ImageDraw.Draw(image)
            font_object = ImageFont.truetype(str(font), max(8, side // 9))
            for y, line in enumerate([f"Trang {page}", f"Số: {number}", "MẪU THỬ"]):
                painter.text((4, 6 + y * (side // 4)), line, font=font_object, fill="black")
            raw = bytearray(image.tobytes())
            hashes.append(hashlib.sha256(raw).hexdigest())
            pixels = torch.frombuffer(raw, dtype=torch.uint8).reshape(side, side, 3)
            tiles.append(pixels.permute(2, 0, 1).float() / 127.5 - 1.0)
    prompt, target = "Số trang 1?", str(numbers[0])
    prefix = [1, *encode(prompt), 7]
    answer = [*encode(target), 2]
    return {"input_ids": torch.tensor([prefix + answer]),
            "labels": torch.tensor([[-100] * len(prefix) + answer]),
            "images": torch.stack(tiles)[None],
            "geometry": torch.tensor([[[0, 0, 1, 1, 0], [0, 0, 1, 1, 1 / max_pages]]]),
            "provenance": {"origin": "synthetic_programmatic", "index": index,
                           "image_hashes": hashes, "font_sha256": hashlib.sha256(font.read_bytes()).hexdigest(),
                           "prompt": prompt, "target": target, "independent_benchmark": False,
                           "scope": "two-page-scale-smoke-not-legal-material"}}
