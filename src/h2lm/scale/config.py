from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ScaleConfig:
    name: str = "h2lm-1b-byte-prototype"
    vocab_size: int = 266
    d_model: int = 1792
    layers: int = 26
    heads: int = 14
    kv_heads: int = 2
    intermediate: int = 5120
    vision_dim: int = 768
    vision_layers: int = 12
    vision_heads: int = 12
    vision_intermediate: int = 2048
    patch_size: int = 16
    resampler_tokens: int = 32
    max_tiles: int = 16
    max_text_tokens: int = 2048
    max_total_tokens: int = 4096
    max_image_side: int = 512
    max_pages: int = 64

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or len(self.name) > 100:
            raise ValueError("Invalid model name")
        limits = {
            "vocab_size": (266, 65536), "d_model": (32, 2048), "layers": (1, 32),
            "heads": (1, 32), "kv_heads": (1, 32), "intermediate": (32, 6144),
            "vision_dim": (16, 1024), "vision_layers": (1, 16), "vision_heads": (1, 32),
            "vision_intermediate": (16, 4096), "patch_size": (4, 32),
            "resampler_tokens": (1, 64), "max_tiles": (1, 32),
            "max_text_tokens": (8, 8192), "max_total_tokens": (16, 16384),
            "max_image_side": (32, 1024), "max_pages": (1, 256),
        }
        for key, (lo, hi) in limits.items():
            value = getattr(self, key)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"{key} must be an integer in [{lo}, {hi}]")
        if self.d_model % self.heads or self.heads % self.kv_heads:
            raise ValueError("Invalid grouped-query attention dimensions")
        if (self.d_model // self.heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if self.vision_dim % self.vision_heads or self.vision_dim % 4:
            raise ValueError("Invalid vision dimensions")
        if self.max_image_side % self.patch_size:
            raise ValueError("Maximum image side must divide by patch size")
        if self.max_total_tokens < self.max_text_tokens:
            raise ValueError("Total context cannot be smaller than text context")

    @classmethod
    def read(cls, path: str | Path) -> ScaleConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("Model config must be a mapping")
        if set(raw) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown model config key")
        return cls(**raw)


def tiny_config() -> ScaleConfig:
    return ScaleConfig(name="h2lm-scale-unit-test", d_model=64, layers=2, heads=4,
                       kv_heads=2, intermediate=128, vision_dim=32, vision_layers=2,
                       vision_heads=4, vision_intermediate=64, patch_size=8,
                       resampler_tokens=4, max_tiles=4, max_text_tokens=128,
                       max_total_tokens=256, max_image_side=128, max_pages=4)
