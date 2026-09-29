"""Dense document vision + learned resampling + language decoder, built from random weights."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .config import ScaleConfig


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return normed.to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width: int, intermediate: int) -> None:
        super().__init__()
        self.gate = nn.Linear(width, intermediate, bias=False)
        self.up = nn.Linear(width, intermediate, bias=False)
        self.down = nn.Linear(intermediate, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def rope(x: torch.Tensor) -> torch.Tensor:
    length, width = x.shape[-2:]
    frequencies = 10000.0 ** (-torch.arange(0, width, 2, device=x.device).float() / width)
    angles = torch.arange(length, device=x.device).float()[:, None] * frequencies[None, :]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, width: int, heads: int, kv_heads: int, use_rope: bool = False) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, width // heads
        self.use_rope = use_rope
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(width, kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None,
                allowed: torch.Tensor | None = None) -> torch.Tensor:
        source = x if memory is None else memory
        batch, length, width = x.shape
        q = self.q(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(source).view(batch, -1, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(source).view(batch, -1, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            q, k = rope(q), rope(k)
        repeats = self.heads // self.kv_heads
        k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
        # For SDPA bool mask, True means the key is allowed (unlike Transformer mask).
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
        return self.out(out.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, width: int, heads: int, kv_heads: int, intermediate: int,
                 use_rope: bool = False) -> None:
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.attention = Attention(width, heads, kv_heads, use_rope)
        self.mlp = SwiGLU(width, intermediate)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), allowed=allowed)
        return x + self.mlp(self.norm2(x))


def spatial_position(height: int, width: int, dim: int, device: torch.device,
                     dtype: torch.dtype) -> torch.Tensor:
    frequency = 10000.0 ** (-torch.arange(dim // 4, device=device).float() / (dim // 4))
    y, x = torch.meshgrid(torch.arange(height, device=device),
                          torch.arange(width, device=device), indexing="ij")
    ax, ay = x.flatten()[:, None] * frequency, y.flatten()[:, None] * frequency
    return torch.cat((ax.sin(), ax.cos(), ay.sin(), ay.cos()), -1).to(dtype)[None]


class Vision(nn.Module):
    def __init__(self, cfg: ScaleConfig) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, cfg.vision_dim, cfg.patch_size, cfg.patch_size, bias=False)
        self.blocks = nn.ModuleList([Block(cfg.vision_dim, cfg.vision_heads, cfg.vision_heads,
                                          cfg.vision_intermediate) for _ in range(cfg.vision_layers)])
        self.norm = RMSNorm(cfg.vision_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch(images)
        h, w = patches.shape[-2:]
        x = patches.flatten(2).transpose(1, 2)
        x = x + spatial_position(h, w, x.shape[-1], x.device, x.dtype)
        for layer in self.blocks:
            x = layer(x)
        return self.norm(x)


class Fusion(nn.Module):
    def __init__(self, cfg: ScaleConfig) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(cfg.resampler_tokens, cfg.vision_dim))
        self.norm_q, self.norm_kv = RMSNorm(cfg.vision_dim), RMSNorm(cfg.vision_dim)
        self.norm_ff, self.norm_out = RMSNorm(cfg.vision_dim), RMSNorm(cfg.vision_dim)
        self.cross = Attention(cfg.vision_dim, cfg.vision_heads, cfg.vision_heads)
        self.mlp = SwiGLU(cfg.vision_dim, cfg.vision_intermediate)
        # bbox x0,y0,x1,y1 plus normalized physical page number, not hidden document text.
        self.geometry = nn.Linear(5, cfg.vision_dim, bias=False)
        self.project = nn.Linear(cfg.vision_dim, cfg.d_model, bias=False)

    def forward(self, visual: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        q = self.queries[None].expand(visual.shape[0], -1, -1)
        q = q + self.cross(self.norm_q(q), self.norm_kv(visual))
        q = q + self.mlp(self.norm_ff(q))
        q = q + self.geometry(geometry)[:, None]
        return self.project(self.norm_out(q))


class Language(nn.Module):
    def __init__(self, cfg: ScaleConfig) -> None:
        super().__init__()
        # Independent useful input/output matrices, no artificially duplicated aliases.
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.heads, cfg.kv_heads,
                                          cfg.intermediate, use_rope=True) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)


def prefix_mask(vision_valid: torch.Tensor, text_valid: torch.Tensor) -> torch.Tensor:
    b, v = vision_valid.shape
    t = text_valid.shape[1]
    mask = torch.zeros(b, v + t, v + t, dtype=torch.bool, device=text_valid.device)
    mask[:, :v, :v] = vision_valid[:, None, :]
    mask[:, v:, :v] = vision_valid[:, None, :]
    mask[:, v:, v:] = torch.ones(t, t, device=text_valid.device, dtype=torch.bool).tril()[None]
    mask[:, v:, v:] &= text_valid[:, None, :]
    # A padded query attends only itself, never infects valid queries or creates all-masked NaNs.
    valid = torch.cat((vision_valid, text_valid), 1)
    mask &= valid[:, :, None]
    diagonal = torch.eye(v + t, dtype=torch.bool, device=mask.device)[None]
    mask |= diagonal & ~valid[:, :, None]
    return mask[:, None]


class H2LM1B(nn.Module):
    def __init__(self, cfg: ScaleConfig) -> None:
        super().__init__()
        self.config = cfg
        self.vision, self.fusion, self.language = Vision(cfg), Fusion(cfg), Language(cfg)

    def reset_parameters(self) -> None:
        # Explicit after to_empty: no uninitialized tensor is counted as a runnable model.
        for name, p in self.named_parameters():
            if p.ndim == 1 and name.endswith("weight"):
                nn.init.ones_(p)
            else:
                nn.init.normal_(p, std=0.02)

    def forward(self, input_ids: torch.Tensor, images: torch.Tensor | None = None,
                geometry: torch.Tensor | None = None, tile_mask: torch.Tensor | None = None,
                text_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None) -> dict:
        cfg = self.config
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a long [batch, text] tensor")
        batch, text = input_ids.shape
        if not 1 <= batch <= 4 or not 1 <= text <= cfg.max_text_tokens:
            raise ValueError("Batch/text outside configured limits")
        if input_ids.min() < 0 or input_ids.max() >= cfg.vocab_size:
            raise ValueError("Token ID outside vocabulary")
        if text_mask is None:
            text_mask = input_ids.ne(3)
        if text_mask.shape != input_ids.shape or text_mask.dtype != torch.bool:
            raise ValueError("Invalid text mask")
        if not text_mask[:, 0].all() or (text_mask[:, 1:] & ~text_mask[:, :-1]).any():
            raise ValueError("Only nonempty right-padded text is supported")
        hidden = self.language.embedding(input_ids)
        if images is None:
            if geometry is not None or tile_mask is not None:
                raise ValueError("Geometry/mask without images")
            visual = hidden.new_empty(batch, 0, cfg.d_model)
            visual_valid = text_mask.new_empty(batch, 0)
        else:
            if images.ndim != 5 or images.shape[0] != batch or images.shape[2] != 3:
                raise ValueError("images must be [batch, tiles, 3, height, width]")
            tiles, h, w = images.shape[1], images.shape[-2], images.shape[-1]
            if (not 1 <= tiles <= cfg.max_tiles or min(h, w) < cfg.patch_size
                    or max(h, w) > cfg.max_image_side or h % cfg.patch_size or w % cfg.patch_size):
                raise ValueError("Invalid tile geometry/count")
            if not torch.isfinite(images).all():
                raise ValueError("Non-finite image input")
            if geometry is None or geometry.shape != (batch, tiles, 5):
                raise ValueError("Required bbox/page geometry is missing or malformed")
            if (not torch.isfinite(geometry).all() or geometry.min() < 0 or geometry.max() > 1
                    or not (geometry[..., 2:4] > geometry[..., 0:2]).all()):
                raise ValueError("Invalid normalized source geometry")
            if tile_mask is None:
                tile_mask = torch.ones(batch, tiles, device=images.device, dtype=torch.bool)
            if tile_mask.shape != (batch, tiles) or tile_mask.dtype != torch.bool:
                raise ValueError("Invalid tile mask")
            visual = self.vision(images.reshape(-1, 3, h, w))
            visual = self.fusion(visual, geometry.reshape(-1, 5).to(visual.dtype))
            visual = visual.reshape(batch, tiles * cfg.resampler_tokens, cfg.d_model)
            visual_valid = tile_mask.repeat_interleave(cfg.resampler_tokens, 1)
        v = visual.shape[1]
        if v + text > cfg.max_total_tokens:
            raise ValueError("Total multimodal context limit exceeded; no silent truncation")
        hidden = torch.cat((visual, hidden), 1)
        allowed = prefix_mask(visual_valid, text_mask)
        for block in self.language.blocks:
            hidden = block(hidden, allowed)
        logits = self.language.head(self.language.norm(hidden[:, v:]))
        result = {"logits": logits, "vision_tokens": v}
        if labels is not None:
            if labels.shape != input_ids.shape or labels.dtype != torch.long or text < 2:
                raise ValueError("Invalid shifted-label shape or dtype")
            target = labels[:, 1:].masked_fill(~text_mask[:, 1:], -100)
            valid_targets = target[target != -100]
            if (valid_targets.numel() == 0 or valid_targets.min() < 0
                    or valid_targets.max() >= cfg.vocab_size):
                raise ValueError("No supervised target or out-of-range target")
            result["loss"] = F.cross_entropy(logits[:, :-1].float().reshape(-1, cfg.vocab_size),
                                              target.reshape(-1), ignore_index=-100)
        return result


def parameter_report(model: H2LM1B) -> dict:
    unique = dict(model.named_parameters())
    all_names = list(model.named_parameters(remove_duplicate=False))
    groups = defaultdict(int)
    for name, p in unique.items():
        groups[name.split(".")[0]] += p.numel()
    total = sum(p.numel() for p in unique.values())
    return {"unique_parameters": total, "trainable_parameters": sum(p.numel() for p in unique.values()
                                                                   if p.requires_grad),
            "parameter_tensors": len(unique), "aliases": len(all_names) - len(unique),
            "components": dict(groups), "fp32_weight_bytes": 4 * total,
            "bf16_weight_bytes": 2 * total, "at_least_one_billion": total >= 1_000_000_000,
            "initialized_storage": all(p.device.type != "meta" for p in unique.values()),
            "quality_certified": False}


def build(cfg: ScaleConfig, dtype: torch.dtype = torch.float32,
          device: str = "cpu", initialize: bool = True,
          backing_file: Path | None = None) -> H2LM1B:
    with torch.device("meta"):
        model = H2LM1B(cfg)
    model.to(dtype=dtype)
    if device != "meta":
        if backing_file is None:
            model.to_empty(device=device)
        else:
            if device != "cpu":
                raise ValueError("File-backed allocation is CPU-only")
            count = sum(p.numel() for p in model.parameters())
            with backing_file.open("xb") as stream:
                stream.truncate(count * torch.empty((), dtype=dtype).element_size())
            arena = torch.from_file(str(backing_file), shared=True, size=count, dtype=dtype)
            offset = 0
            for name, p in list(model.named_parameters()):
                owner, leaf = name.rsplit(".", 1)
                # Independent tensor version counters over DISJOINT storage ranges.
                # Slicing one arena would share counters and invalidate streaming backward.
                value = torch.empty(0, dtype=dtype).set_(arena.untyped_storage(), offset, p.shape)
                model.get_submodule(owner)._parameters[leaf] = nn.Parameter(value)
                offset += p.numel()
            assert offset == count  # Non-overlapping real backing storage, not repeated weights.
        if initialize:
            model.reset_parameters()
    return model
