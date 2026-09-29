from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from h2lm.config import H2LMConfig


class H2VisionEncoder(nn.Module):
    """Small, editable reference document-vision encoder.

    This is intentionally simple. It proves the training/inference contract
    before the production document encoder is locked by benchmark.
    """

    def __init__(self, config: H2LMConfig) -> None:
        super().__init__()
        cfg = config.vision
        self.image_size = cfg.image_size
        self.patch_size = cfg.patch_size
        self.grid_size = cfg.image_size // cfg.patch_size

        self.patch_embed = nn.Conv2d(
            cfg.in_channels,
            cfg.d_vision,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            bias=True,
        )
        self.position = nn.Parameter(
            torch.zeros(1, self.grid_size * self.grid_size, cfg.d_vision)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_vision,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.d_vision * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.d_vision)

        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must have shape [batch, channels, height, width]")

        _, _, height, width = pixel_values.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Reference encoder expects {self.image_size}x{self.image_size} images; "
                f"received {height}x{width}. High-resolution tiling is a later milestone."
            )

        patches = self.patch_embed(pixel_values)
        patches = patches.flatten(2).transpose(1, 2)
        patches = patches + self.position
        return self.norm(self.blocks(patches))


class H2LM(nn.Module):
    """Reference H2LM multimodal prefix model initialized from random weights."""

    def __init__(self, config: H2LMConfig) -> None:
        super().__init__()
        self.config = config
        model_cfg = config.model

        self.vision = H2VisionEncoder(config)
        self.vision_projector = nn.Linear(config.vision.d_vision, model_cfg.d_model)

        self.token_embedding = nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)
        self.text_position = nn.Embedding(model_cfg.max_text_tokens, model_cfg.d_model)
        self.vision_modality = nn.Parameter(torch.zeros(1, 1, model_cfg.d_model))
        self.text_modality = nn.Parameter(torch.zeros(1, 1, model_cfg.d_model))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=model_cfg.d_model,
            nhead=model_cfg.num_heads,
            dim_feedforward=int(model_cfg.d_model * model_cfg.mlp_ratio),
            dropout=model_cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=model_cfg.num_layers)
        self.final_norm = nn.LayerNorm(model_cfg.d_model)
        self.lm_head = nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False)

        # Weight tying keeps the reference model smaller and is easy to remove in config later.
        self.lm_head.weight = self.token_embedding.weight

        nn.init.normal_(self.vision_modality, std=0.02)
        nn.init.normal_(self.text_modality, std=0.02)

    @staticmethod
    def _prefix_causal_mask(
        vision_tokens: int,
        text_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        total = vision_tokens + text_tokens
        mask = torch.ones(total, total, dtype=torch.bool, device=device)

        if vision_tokens:
            mask[:vision_tokens, :vision_tokens] = False

        if text_tokens:
            if vision_tokens:
                mask[vision_tokens:, :vision_tokens] = False
            text_future_mask = torch.triu(
                torch.ones(text_tokens, text_tokens, dtype=torch.bool, device=device),
                diagonal=1,
            )
            mask[vision_tokens:, vision_tokens:] = text_future_mask

        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")

        batch_size, text_len = input_ids.shape
        if text_len > self.config.model.max_text_tokens:
            raise ValueError(
                f"text length {text_len} exceeds configured maximum "
                f"{self.config.model.max_text_tokens}"
            )

        positions = torch.arange(text_len, device=input_ids.device)
        text_hidden = self.token_embedding(input_ids)
        text_hidden = text_hidden + self.text_position(positions)[None, :, :]
        text_hidden = text_hidden + self.text_modality

        if pixel_values is None:
            vision_hidden = text_hidden.new_zeros((batch_size, 0, text_hidden.shape[-1]))
        else:
            if pixel_values.shape[0] != batch_size:
                raise ValueError("pixel_values batch size must match input_ids")
            vision_hidden = self.vision_projector(self.vision(pixel_values))
            vision_hidden = vision_hidden + self.vision_modality

        vision_len = vision_hidden.shape[1]
        hidden = torch.cat([vision_hidden, text_hidden], dim=1)
        mask = self._prefix_causal_mask(vision_len, text_len, input_ids.device)
        hidden = self.decoder(hidden, mask=mask)
        hidden = self.final_norm(hidden)

        text_output = hidden[:, vision_len:, :]
        logits = self.lm_head(text_output)

        result: dict[str, Any] = {
            "logits": logits,
            "vision_tokens": vision_len,
        }

        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            if text_len < 2:
                raise ValueError("at least two text tokens are required to compute loss")
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
