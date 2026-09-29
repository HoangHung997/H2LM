"""Reproducible synthetic VQA and explicitly opted-in real-scan silver-label fit."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

OFFSET, VOCAB = 10, 266  # Separate experiment byte codec; NOT the production tokenizer.


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def encode(text: str) -> list[int]:
    return [b + OFFSET for b in text.encode("utf-8")]


def decode(ids: list[int]) -> str:
    return bytes(i - OFFSET for i in ids).decode("utf-8")


def image_tensor(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as image:
        if image.width * image.height > 16_000_000:
            raise ValueError("Oversized source image")
        with (
            image.convert("L") as gray,
            ImageOps.contain(gray, (size, size)) as fitted,
            Image.new("L", (size, size), 255) as canvas,
        ):
            canvas.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
            return torch.tensor(list(canvas.getdata()), dtype=torch.float32).view(
                1, size, size) / 127.5 - 1.0


def prepare(output: Path, real_review: Path | None = None, allow_silver: bool = False) -> dict:
    """No downloads, pretrained weights, OCR, teacher services or user configuration needed."""
    if real_review is not None and not allow_silver:
        raise ValueError("Real AI drafts require explicit --allow-silver; never human_verified")
    fonts = [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
             Path("C:/Windows/Fonts/arial.ttf")]
    font_path = next((p for p in fonts if p.is_file()), None)
    if font_path is None:
        raise ValueError("Install a Vietnamese-capable system font; no font file is distributed")
    output.mkdir(parents=True, exist_ok=False)
    images = output / "images"
    images.mkdir()
    rng = random.Random(4091)
    pairs = [(a, b) for a in range(1, 13) for b in range(1, 13) if a != b]
    rng.shuffle(pairs)
    records = []
    for index, (a, b) in enumerate(pairs[:80]):
        split = "train" if index < 64 else "validation"
        group = f"synthetic-{a}-{b}"
        path = images / f"{group}.png"
        with Image.new("L", (128, 128), 255) as image:
            painter = ImageDraw.Draw(image)
            font = ImageFont.truetype(str(font_path), 18)
            dx, dy = rng.randrange(3), rng.randrange(3)
            painter.text((4 + dx, 22 + dy), f"Điều: {a}", font=font, fill=20)
            painter.text((4 + dx, 68 + dy), f"Khoản: {b}", font=font, fill=20)
            # Mild digital blur only, not a physical scanner acceptance claim.
            with image.filter(ImageFilter.GaussianBlur(0.15)) as blurred:
                blurred.save(path)
        for task, answer in (("Điều?", str(a)), ("Khoản?", str(b))):
            records.append({"id": f"{group}-{len(records)}", "group": group,
                            "split": split, "prompt": task, "target": answer,
                            "image": path.relative_to(output).as_posix(), "sha256": sha(path),
                            "label_kind": "synthetic_programmatic", "production_approved": False,
                            "synthetic_fields": [a - 1, b - 1]})
    if real_review is not None:
        source = json.loads((real_review / "draft_review.json").read_text(encoding="utf-8"))
        permission = source.get("permission", {})
        if permission.get("public_sharing_allowed_by_user") is not True:
            raise ValueError("Real seed permission is missing")
        if set(permission.get("applies_to_document_ids", [])) != {"da700", "dt391", "thach-bich"}:
            raise ValueError("Unsupported real seed scope")
        selected = {f"{prefix}-{field}" for prefix in ("da", "dt", "tb")
                    for field in ("number", "date")}
        found = set()
        for region in source["regions"]:
            if region["id"] not in selected:
                continue
            found.add(region["id"])
            if region["status"] != "readable" or region["review_status"] != "assistant_visual_draft":
                raise ValueError("Only selected readable AI drafts may enter this isolated experiment")
            if region.get("training_eligible") is not False:
                raise ValueError("Do not relabel experimental drafts as production ground truth")
            path = (real_review / region["crop_path"]).resolve()
            if not path.is_relative_to(real_review.resolve()) or sha(path) != region["crop_sha256"]:
                raise ValueError("Real crop hash/path mismatch")
            destination = images / f"{region['id']}.png"
            destination.write_bytes(path.read_bytes())
            document = region["document_id"]
            records.append({"id": region["id"], "group": ("project-da700" if document == "da700"
                            else "bqp-project-assignment-template"),
                            "split": "real_probe" if document == "da700" else "train",
                            "prompt": "Số?" if region["id"].endswith("number") else "Ngày?",
                            "target": region["target"], "image": destination.relative_to(output).as_posix(),
                            "sha256": sha(destination), "label_kind": "silver_ai_draft",
                            "region_id": region["id"], "source_sha256": region["source_sha256"],
                            "page_image_sha256": region["page_image_sha256"],
                            "production_approved": False})
        if found != selected:
            raise ValueError("Missing selected real regions")
    manifest = {"schema_version": 1, "experiment_only": True, "records": records,
                "font_sha256": sha(font_path), "font_name": font_path.name,
                "label_policy": "programmatic synthetic + optional unverified AI silver",
                "independent_benchmark": False,
                "real_probe_note": "document-disjoint but previously inspected development samples"}
    dump(output / "manifest.json", manifest)
    return manifest


def load(path: Path, size: int, max_text_tokens: int, allow_silver: bool = False) -> list[dict]:
    if path.stat().st_size > 2_000_000:
        raise ValueError("Dataset manifest too large")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("experiment_only") is not True:
        raise ValueError("Only experimental schema 1 is supported")
    records = data["records"]
    if not 2 <= len(records) <= 512:
        raise ValueError("Expected 2-512 bounded records")
    seen, groups, hashes = set(), {}, {}
    for record in records:
        if record["id"] in seen or record["split"] not in ("train", "validation", "real_probe"):
            raise ValueError("Duplicate ID or invalid split")
        seen.add(record["id"])
        if record["label_kind"] not in ("synthetic_programmatic", "silver_ai_draft"):
            raise ValueError("Unsupported label provenance")
        if record["label_kind"] == "synthetic_programmatic":
            fields = record.get("synthetic_fields")
            if not isinstance(fields, list) or len(fields) != 2 or any(
                type(v) is not int or not 0 <= v < 12 for v in fields
            ):
                raise ValueError("Missing generated field supervision")
        if record["label_kind"] == "silver_ai_draft" and not allow_silver:
            raise ValueError("Silver draft use must be explicitly enabled")
        if record.get("production_approved") is not False:
            raise ValueError("Production labels cannot be smuggled into experiment")
        for key, mapping in ((record["group"], groups), (record["sha256"], hashes)):
            if key in mapping and mapping[key] != record["split"]:
                raise ValueError("Group/image leakage across splits")
            mapping[key] = record["split"]
        image = (path.parent / record["image"]).resolve()
        if not image.is_relative_to(path.parent.resolve()) or image.stat().st_size > 8_000_000:
            raise ValueError("Out-of-scope or oversized image")
        if sha(image) != record["sha256"]:
            raise ValueError("Image checksum mismatch")
        prompt = [1, *encode(record["prompt"]), 7]
        answer = [*encode(record["target"]), 2]
        if not record["target"] or len(prompt) + len(answer) > max_text_tokens:
            raise ValueError("Empty target or overlong sample; no silent truncation")
        record["tensor"] = image_tensor(image, size)
        record["ids"] = prompt + answer
        record["labels"] = [-100] * len(prompt) + answer
        record["prefix"] = prompt
    if not any(r["split"] == "train" for r in records):
        raise ValueError("No training samples")
    return records
