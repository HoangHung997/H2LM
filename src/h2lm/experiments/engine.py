"""Real gradient training on the existing H2LM, bounded CPU runner and resumable checkpoints."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from h2lm.config import H2LMConfig, ModelConfig, VisionConfig
from h2lm.modeling.h2lm import H2LM, H2VisionEncoder

from .data import VOCAB, decode, dump, encode, load, sha


def validate(cfg: dict) -> None:
    limits = {"steps": (1, 3000), "batch_size": (1, 16), "seed": (0, 100000),
              "threads": (1, 4), "seconds": (1, 600), "save_every": (1, 1000),
              "image_size": (32, 192), "patch_size": (8, 32), "d_model": (32, 128),
              "vision_layers": (1, 2), "text_layers": (1, 3),
              "max_text_tokens": (32, 128), "max_new_tokens": (1, 64)}
    if set(cfg) != set(limits) | {"learning_rate", "allow_silver"}:
        raise ValueError("Missing or unknown experiment config keys")
    for key, (low, high) in limits.items():
        if type(cfg[key]) is not int or not low <= cfg[key] <= high:
            raise ValueError(f"Invalid bounded config: {key}")
    lr = cfg["learning_rate"]
    if type(lr) not in (int, float) or not math.isfinite(lr) or not 0 < lr <= 0.01:
        raise ValueError("Invalid learning rate")
    if type(cfg["allow_silver"]) is not bool or cfg["d_model"] % 4:
        raise ValueError("Invalid silver flag or model dimension")
    if cfg["image_size"] % cfg["patch_size"] or (cfg["image_size"] // cfg["patch_size"]) % 2:
        raise ValueError("Image size must produce an even patch grid")


class StripVision(H2VisionEncoder):
    """Experimental learned resampler: two horizontal strips, NOT general page parsing."""
    def __init__(self, config: H2LMConfig) -> None:
        super().__init__(config)
        self.compactor = nn.Linear(self.grid_size ** 2 // 2 * config.vision.d_vision,
                                   config.vision.d_vision)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        hidden = super().forward(pixels)
        return self.compactor(hidden.reshape(hidden.shape[0], 2, -1))


def make_model(cfg: dict) -> H2LM:
    config = H2LMConfig("h2lm-micro-experiment", ModelConfig(
        VOCAB, cfg["max_text_tokens"], cfg["d_model"], cfg["text_layers"], 4), VisionConfig(
        cfg["image_size"], cfg["patch_size"], 1, cfg["d_model"], cfg["vision_layers"], 4))
    model = H2LM(config)
    model.vision = StripVision(config)
    # Auxiliary supervised field reading on GENERATED images only, not legal decision labels.
    # This head is unused in generation; it prevents the visual encoder learning only a language prior.
    model.add_module("visual_field_head", nn.Linear(
        2 * cfg["d_model"], 24))
    # TransformerEncoder clones layers: initialize independently, with a sane tied embedding scale.
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.MultiheadAttention):
            nn.init.normal_(module.in_proj_weight, std=0.02)
            nn.init.zeros_(module.in_proj_bias)
    return model


def source_hash() -> str:
    paths = list(Path(__file__).parent.glob("*.py"))
    root = Path(__file__).parents[1]
    paths += [root / "config.py", root / "modeling/h2lm.py"]
    return hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in sorted(paths))).hexdigest()


def weights_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(bytes(tensor.detach().cpu().contiguous().clone().untyped_storage()))
    return h.hexdigest()


def batch(records: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = max(len(r["ids"]) for r in records)
    ids = torch.full((len(records), length), 3, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    for i, row in enumerate(records):
        ids[i, :len(row["ids"])] = torch.tensor(row["ids"])
        labels[i, :len(row["labels"])] = torch.tensor(row["labels"])
    return ids, torch.stack([r["tensor"] for r in records]), labels


@torch.no_grad()
def loss_on(model: H2LM, records: list[dict]) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(records), 8):
        ids, pixels, labels = batch(records[i:i + 8])
        logits = model(ids, pixels)["logits"][:, :-1]
        target = labels[:, 1:]
        total += float(F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1),
                                       ignore_index=-100, reduction="sum"))
        count += int((target != -100).sum())
    return total / count


@torch.no_grad()
def predict(model: H2LM, pixels: torch.Tensor, prompt: str, max_new_tokens: int) -> dict:
    model.eval()
    ids = [1, *encode(prompt), 7]
    generated = []
    ended = False
    for _ in range(max_new_tokens):
        if len(ids) >= model.config.model.max_text_tokens:
            break
        logits = model(torch.tensor([ids]), pixels[None])["logits"][0, -1].clone()
        logits[:2] = -float("inf")
        logits[3:10] = -float("inf")  # Only EOS or ordinary bytes may be generated.
        next_id = int(logits.argmax())
        if next_id == 2:
            ended = True
            break
        generated.append(next_id)
        ids.append(next_id)
    try:
        text, valid = decode(generated), True
    except UnicodeDecodeError:
        text, valid = bytes(i - 10 for i in generated).decode("utf-8", errors="replace"), False
    return {"text": text, "valid_utf8": valid, "eos": ended}


def distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        row = [i]
        for j, cb in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = row
    return previous[-1]


def evaluate(model: H2LM, records: list[dict], cfg: dict) -> dict:
    groups = {}
    for row in records:
        key = row["split"] + ":" + row["label_kind"]
        groups.setdefault(key, []).append(row)
    reports = {}
    for key, rows in groups.items():
        examples = []
        errors, characters = 0, 0
        for row in rows:
            normal = predict(model, row["tensor"], row["prompt"], cfg["max_new_tokens"])
            blank = predict(model, torch.ones_like(row["tensor"]), row["prompt"], cfg["max_new_tokens"])
            match = normal["valid_utf8"] and normal["eos"] and normal["text"] == row["target"]
            errors += distance(normal["text"], row["target"])
            characters += len(row["target"])
            examples.append({"id": row["id"], "prompt": row["prompt"], "target": row["target"],
                             "prediction": normal, "exact": match, "blank_prediction": blank,
                             "blank_exact": blank["eos"] and blank["text"] == row["target"]})
        reports[key] = {"count": len(rows), "exact": sum(r["exact"] for r in examples),
                        "blank_exact": sum(r["blank_exact"] for r in examples),
                        "char_error_rate": errors / characters, "examples": examples}
    return reports


def save_checkpoint(out: Path, state: dict) -> Path:
    path = out / f"step-{state['step']:06d}.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)
    dump(out / "latest.tmp", {"file": path.name, "sha256": sha(path)})
    os.replace(out / "latest.tmp", out / "latest.json")
    return path


def read_checkpoint(out: Path) -> dict:
    ref = json.loads((out / "latest.json").read_text())
    path = (out / ref["file"]).resolve()
    if not path.is_relative_to(out.resolve()) or path.stat().st_size > 100_000_000:
        raise ValueError("Invalid checkpoint path/size")
    if sha(path) != ref["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    return torch.load(path, map_location="cpu", weights_only=True)


def train(manifest: Path, out: Path, cfg: dict, resume: bool = False,
          segment_steps: int | None = None) -> dict:
    validate(cfg)
    if segment_steps is not None and (type(segment_steps) is not int or not 1 <= segment_steps <= cfg["steps"]):
        raise ValueError("Invalid segment length")
    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(cfg["seed"])
    records = load(manifest, cfg["image_size"], cfg["max_text_tokens"], cfg["allow_silver"])
    rows = [r for r in records if r["split"] == "train"]
    contract = {"config": cfg, "dataset_sha256": sha(manifest), "source_sha256": source_hash(),
                "torch": str(torch.__version__)}
    out.mkdir(parents=True, exist_ok=resume)
    lock = out / "RUNNING.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    started = time.monotonic()
    try:
        model = make_model(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
        generator = torch.Generator().manual_seed(cfg["seed"])
        step = 0
        initial_hash = weights_hash(model)
        initial_vision = model.vision.patch_embed.weight.detach().clone()
        initial_loss = loss_on(model, rows)
        if resume:
            saved = read_checkpoint(out)
            if saved["contract"] != contract:
                raise ValueError("Resume config/data/source/library mismatch")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            generator.set_state(saved["sampler_rng"])
            torch.set_rng_state(saved["torch_rng"])
            step, initial_hash, initial_loss = saved["step"], saved["initial_hash"], saved["initial_loss"]
            initial_vision = saved["initial_vision"]
        else:
            dump(out / "contract.json", contract)
        synthetic = sum(r["label_kind"] == "synthetic_programmatic" for r in rows)
        silver = len(rows) - synthetic
        weights = torch.tensor([1 / (silver or 1) if r["label_kind"] == "silver_ai_draft"
                                else 1 / (synthetic or 1) for r in rows], dtype=torch.double)
        interrupted = False
        segment_end = min(cfg["steps"], step + (segment_steps or cfg["steps"]))
        try:
            while step < segment_end and time.monotonic() - started < cfg["seconds"]:
                chosen = torch.multinomial(weights, cfg["batch_size"], True, generator=generator)
                selected = [rows[int(i)] for i in chosen]
                ids, pixels, labels = batch(selected)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(ids, pixels)["logits"][:, :-1]
                target = labels[:, 1:]
                losses = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1),
                                         ignore_index=-100, reduction="none").view_as(target)
                # Each answer has equal influence; long, mostly predictable labels must not dominate.
                loss = (losses.sum(dim=1) / (target != -100).sum(dim=1)).mean()
                synthetic_indices = [i for i, r in enumerate(selected)
                                     if r["label_kind"] == "synthetic_programmatic"]
                if synthetic_indices:
                    visual = model.vision(pixels[synthetic_indices])
                    field_logits = model.visual_field_head(visual.flatten(1)).view(-1, 12)
                    field_targets = torch.tensor([selected[i]["synthetic_fields"]
                                                  for i in synthetic_indices]).view(-1)
                    loss = loss + 0.5 * F.cross_entropy(field_logits, field_targets)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                if step % cfg["save_every"] == 0:
                    print(json.dumps({"step": step, "loss": float(loss.detach()),
                                      "gradient_norm": float(norm)}), flush=True)
                    save_checkpoint(out, {"step": step, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "contract": contract,
                        "sampler_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                        "initial_hash": initial_hash, "initial_loss": initial_loss,
                        "initial_vision": initial_vision})
        except KeyboardInterrupt:
            interrupted = True
        checkpoint = save_checkpoint(out, {"step": step, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "contract": contract,
            "sampler_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
            "initial_hash": initial_hash, "initial_loss": initial_loss, "initial_vision": initial_vision})
        summary = {"experiment": "H2LM neural micro-fit", "architecture": "two-strip-resampler-v1", "step": step,
                   "requested_steps": cfg["steps"], "training_seconds_this_invocation": time.monotonic() - started,
                   "interrupted": interrupted, "completed_steps": step == cfg["steps"],
                   "parameters": model.parameter_count(), "model_config": asdict(model.config),
                   "initial_train_loss": initial_loss, "final_train_loss": loss_on(model, rows),
                   "weights_changed": weights_hash(model) != initial_hash,
                   "vision_patch_max_change": float((model.vision.patch_embed.weight - initial_vision).abs().max().detach()),
                   "checkpoint": checkpoint.name, "checkpoint_sha256": sha(checkpoint),
                   "training_counts": {"synthetic": synthetic, "silver": silver},
                   "production_ready": False, "independent_benchmark": False,
                   "contract": contract}
        # Actual free-running generation; targets are only used AFTER generation for scoring.
        summary["evaluation"] = evaluate(model, records, cfg) if step == cfg["steps"] else {}
        restored = make_model(cfg)
        restored.load_state_dict(read_checkpoint(out)["model"])
        sample = records[0]
        summary["reload_matches"] = predict(model, sample["tensor"], sample["prompt"], 8) == predict(
            restored, sample["tensor"], sample["prompt"], 8)
        dump(out / f"report-{step:06d}.json", summary)
        return summary
    finally:
        lock.unlink(missing_ok=True)
