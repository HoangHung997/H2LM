"""Bounded sharded state_dict storage. No pretrained weights, pickle objects or silent overwrite."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from .model import H2LM1B, parameter_report


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, content: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save(model: H2LM1B, folder: Path, training: dict, shard_bytes: int = 64 * 1024**2) -> dict:
    if type(shard_bytes) is not int or not 1024 <= shard_bytes <= 256 * 1024**2:
        raise ValueError("Invalid shard size")
    if not parameter_report(model)["initialized_storage"]:
        raise ValueError("Cannot export meta model as real weights")
    folder.mkdir(parents=True, exist_ok=False)
    shards, pending, total = [], {}, 0

    def flush() -> None:
        nonlocal pending, total
        if not pending:
            return
        filename = f"weights-{len(shards) + 1:05d}.pt"
        path = folder / filename
        torch.save(pending, path)
        shards.append({"file": filename, "sha256": digest_file(path),
                       "file_bytes": path.stat().st_size,
                       "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                                        "numel": v.numel()} for k, v in pending.items()}})
        pending, total = {}, 0

    for name, tensor in model.state_dict().items():
        if tensor.device.type != "cpu" or not torch.isfinite(tensor).all():
            raise ValueError("Export requires finite CPU tensors")
        size = tensor.numel() * tensor.element_size()
        if pending and total + size > shard_bytes:
            flush()
        if size > 256 * 1024**2:
            raise ValueError("A tensor exceeds the bounded shard reader")
        # Clone this bounded shard: a mapped tensor may reference the entire 1B storage.
        pending[name] = tensor.detach().clone()
        total += size
    flush()
    index = {"schema_version": 1, "architecture": "h2lm-scale-prefix-v1", "config": asdict(model.config),
             "source_sha256": source_hash(), "parameters": parameter_report(model),
             "training": training, "shards": shards, "quality_certified": False,
             "optimizer": {"name": "SGD", "momentum": 0, "weight_decay": 0,
                           "lr": training["learning_rate"], "state_tensors": 0},
             "torch_rng": torch.get_rng_state().tolist(), "torch": str(torch.__version__)}
    # This commit marker is absent if serialization failed at any point.
    write_json(folder / "index.json", index)
    return index


def index_at(folder: Path) -> dict:
    path = folder / "index.json"
    if path.stat().st_size > 4 * 1024**2 or (folder / "FAILED.json").exists():
        raise ValueError("Invalid/incomplete checkpoint index")
    index = json.loads(path.read_text(encoding="utf-8"))
    if index.get("architecture") != "h2lm-scale-prefix-v1" or index.get("schema_version") != 1:
        raise ValueError("Wrong checkpoint architecture")
    return index


def load_into(model: H2LM1B, folder: Path) -> dict:
    index = index_at(folder)
    if index["config"] != asdict(model.config) or index["source_sha256"] != source_hash():
        raise ValueError("Checkpoint config/source mismatch")
    parameters = dict(model.state_dict())
    seen = set()
    shards = index["shards"]
    if not isinstance(shards, list) or not 1 <= len(shards) <= 128:
        raise ValueError("Invalid checkpoint shard list")
    # Verify every shard before modifying a parameter. Checksum is integrity, not authentication.
    for shard in shards:
        path = (folder / shard["file"]).resolve()
        if (not path.is_relative_to(folder.resolve()) or path.stat().st_size > 270 * 1024**2
                or path.stat().st_size != shard["file_bytes"] or digest_file(path) != shard["sha256"]):
            raise ValueError("Checkpoint shard checksum/path/size mismatch")
        names = set(shard["tensors"])
        if seen & names or not names <= set(parameters):
            raise ValueError("Duplicate/unknown tensors in checkpoint")
        for name, description in shard["tensors"].items():
            p = parameters[name]
            if (description["shape"] != list(p.shape) or description["numel"] != p.numel()
                    or description["dtype"] != str(p.dtype)):
                raise ValueError("Checkpoint tensor shape/dtype mismatch")
        seen |= names
    if seen != set(parameters) or index["parameters"]["unique_parameters"] != sum(p.numel() for p in parameters.values()):
        raise ValueError("Incomplete or incorrect parameter inventory")
    with torch.no_grad():
        for shard in shards:
            tensors = torch.load(folder / shard["file"], map_location="cpu", weights_only=True)
            if set(tensors) != set(shard["tensors"]):
                raise ValueError("Unexpected shard content")
            for name, value in tensors.items():
                p = parameters[name]
                if value.shape != p.shape or value.dtype != p.dtype or not torch.isfinite(value).all():
                    raise ValueError("Invalid stored tensor")
                p.copy_(value)
            del tensors
    return index


def set_latest(run: Path, folder: Path) -> None:
    path = run / "latest.tmp"
    write_json(path, {"folder": folder.name, "index_sha256": digest_file(folder / "index.json")})
    os.replace(path, run / "latest.json")


def get_latest(run: Path) -> Path:
    data = json.loads((run / "latest.json").read_text(encoding="utf-8"))
    folder = (run / data["folder"]).resolve()
    if not folder.is_relative_to(run.resolve()) or digest_file(folder / "index.json") != data["index_sha256"]:
        raise ValueError("Latest checkpoint pointer mismatch")
    return folder
