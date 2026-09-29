from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class VisionConfig:
    image_size: int
    patch_size: int
    in_channels: int
    d_vision: int
    num_layers: int
    num_heads: int
    dropout: float = 0.0


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    max_text_tokens: int
    d_model: int
    num_layers: int
    num_heads: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0


@dataclass(frozen=True)
class H2LMConfig:
    name: str
    model: ModelConfig
    vision: VisionConfig


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required config key: {key}")
    return mapping[key]


def load_config(path: str | Path) -> H2LMConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    model_raw = _require(raw, "model")
    vision_raw = _require(raw, "vision")

    model = ModelConfig(
        vocab_size=int(_require(model_raw, "vocab_size")),
        max_text_tokens=int(_require(model_raw, "max_text_tokens")),
        d_model=int(_require(model_raw, "d_model")),
        num_layers=int(_require(model_raw, "num_layers")),
        num_heads=int(_require(model_raw, "num_heads")),
        mlp_ratio=float(model_raw.get("mlp_ratio", 4.0)),
        dropout=float(model_raw.get("dropout", 0.0)),
    )

    vision = VisionConfig(
        image_size=int(_require(vision_raw, "image_size")),
        patch_size=int(_require(vision_raw, "patch_size")),
        in_channels=int(_require(vision_raw, "in_channels")),
        d_vision=int(_require(vision_raw, "d_vision")),
        num_layers=int(_require(vision_raw, "num_layers")),
        num_heads=int(_require(vision_raw, "num_heads")),
        dropout=float(vision_raw.get("dropout", 0.0)),
    )

    if model.d_model % model.num_heads != 0:
        raise ValueError("model.d_model must be divisible by model.num_heads")
    if vision.d_vision % vision.num_heads != 0:
        raise ValueError("vision.d_vision must be divisible by vision.num_heads")
    if vision.image_size % vision.patch_size != 0:
        raise ValueError("vision.image_size must be divisible by vision.patch_size")

    return H2LMConfig(name=str(_require(raw, "name")), model=model, vision=vision)
