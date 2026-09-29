"""Prepare auditable visual-label drafts. Nothing here approves labels for training."""
from __future__ import annotations

import argparse
import html
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

from .config import ScanConfig
from .ingest import file_hash, prepare_pdf, write_json

logger = logging.getLogger(__name__)
_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,79}")
_SHA = re.compile(r"[a-f0-9]{64}")


def _id(value: Any) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("Invalid identifier")
    return value


def validate_seed(seed: dict[str, Any]) -> None:
    """Structural checks are NOT a visual/human review or an authenticity check."""
    if not isinstance(seed, dict):
        raise TypeError("Review seed must be an object")
    if seed.get("schema_version") != 1 or seed.get("coordinate_space") != (
        "upright_render_normalized_0_1000_xyxy_pixel_edges"
    ):
        raise ValueError("Unsupported review schema or coordinates")
    if seed.get("review", {}).get("training_eligible") is not False:
        raise ValueError("This tool accepts development drafts only, not training approval")
    documents = seed.get("documents")
    if not isinstance(documents, list) or not 1 <= len(documents) <= 20:
        raise ValueError("Expected 1-20 development documents")
    by_id = {}
    for document in documents:
        identifier = _id(document["id"])
        if identifier in by_id:
            raise ValueError("Duplicate document ID")
        filename = document["filename"]
        if not isinstance(filename, str) or any(c in filename for c in ("/", "\\", "\x00")):
            raise ValueError("Expected a local PDF basename")
        if not filename.lower().endswith(".pdf") or not _SHA.fullmatch(document["sha256"]):
            raise ValueError("Invalid filename or source checksum")
        if type(document["pages"]) is not int or not 1 <= document["pages"] <= 20:
            raise ValueError("Unsupported page count for this small review batch")
        if document.get("split") != "development_seed":
            raise ValueError("Public development samples cannot become an independent holdout")
        _id(document["family_id"])
        _id(document["leakage_group"])
        by_id[identifier] = document
    permission = seed.get("permission", {})
    if permission.get("public_sharing_allowed_by_user") is not True:
        raise ValueError("Explicit public-sharing permission is required for this public seed")
    if set(permission.get("applies_to_document_ids", [])) != set(by_id):
        raise ValueError("Permission scope must match the supplied documents")
    regions = seed.get("regions")
    if not isinstance(regions, list) or not 1 <= len(regions) <= 500:
        raise ValueError("Expected a bounded list of draft regions")
    region_ids = {}
    for region in regions:
        identifier = _id(region["id"])
        if identifier in region_ids or region["document_id"] not in by_id:
            raise ValueError("Duplicate region or unknown document")
        if type(region["page"]) is not int or not 1 <= region["page"] <= by_id[
            region["document_id"]
        ]["pages"]:
            raise ValueError("Region page outside source")
        box = region["bbox_1000"]
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError("Expected four integer coordinates")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            raise ValueError("Region outside page")
        if region.get("status") not in ("readable", "uncertain", "unreadable", "missing"):
            raise ValueError("Unknown readability status")
        if region["status"] != "readable" and region.get("target") is not None:
            raise ValueError("Uncertain/unreadable/missing text cannot be filled by guessing")
        if region["status"] == "readable" and region.get("target") is None:
            raise ValueError("Readable draft needs a proposed target")
        if region.get("review_status") != "assistant_visual_draft" or region.get(
            "training_eligible"
        ) is not False:
            raise ValueError("Draft labels cannot auto-promote to verified ground truth")
        region_ids[identifier] = region
    questions = seed.get("questions", [])
    if not isinstance(questions, list) or len(questions) > 200:
        raise ValueError("Too many questions")
    question_ids = set()
    for question in questions:
        identifier = _id(question["id"])
        if identifier in question_ids:
            raise ValueError("Duplicate question ID")
        question_ids.add(identifier)
        scope = question.get("document_scope")
        if not scope or not set(scope) <= set(by_id):
            raise ValueError("Question needs explicit known document scope")
        if any(r not in region_ids or region_ids[r]["document_id"] not in scope
               for r in question.get("evidence_ids", [])):
            raise ValueError("Question references unknown or out-of-scope evidence")
        if question.get("target") is not None:
            if not question.get("evidence_ids") or any(
                region_ids[r]["status"] != "readable" for r in question["evidence_ids"]
            ):
                raise ValueError("A positive answer requires readable image evidence")
        if question.get("review_status") != "assistant_visual_draft" or question.get(
            "training_eligible"
        ) is not False:
            raise ValueError("QA labels also remain drafts")


def build_review(seed_path: str | Path, input_directory: str | Path, output: str | Path,
                 config: ScanConfig | None = None) -> dict[str, Any]:
    from PIL import Image

    seed_path, inputs, out = Path(seed_path), Path(input_directory).resolve(), Path(output)
    if seed_path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Seed description exceeds size bound")
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    validate_seed(seed)
    cfg = config or ScanConfig()
    # Verify every input before creating output; filename itself is never a label source.
    for document in seed["documents"]:
        path = (inputs / document["filename"]).resolve()
        if not path.is_relative_to(inputs) or not path.is_file():
            raise ValueError("Missing or out-of-scope source PDF")
        if path.stat().st_size > cfg.max_pdf_bytes or file_hash(path) != document["sha256"]:
            raise ValueError("Source PDF checksum/size mismatch")
    out.mkdir(parents=True, exist_ok=False)
    reports = {}
    try:
        for document in seed["documents"]:
            report = prepare_pdf(inputs / document["filename"], out / document["id"], cfg,
                                 page_count=cfg.max_selected_pages,
                                 document_id=document["id"], family_id=document["family_id"])
            if len(report["pages"]) != document["pages"] or not report["complete_document"]:
                raise ValueError("Source page count differs from reviewed draft scope")
            reports[document["id"]] = report
        evidence = []
        crops = out / "regions"
        crops.mkdir()
        cards = []
        for region in seed["regions"]:
            page = reports[region["document_id"]]["pages"][region["page"] - 1]
            image_path = out / region["document_id"] / page["page"]["path"]
            if file_hash(image_path) != page["page"]["sha256"]:
                raise ValueError("Prepared image changed before annotation binding")
            with Image.open(image_path) as image:
                x0, y0, x1, y1 = region["bbox_1000"]
                box = [math.floor(x0 * image.width / 1000), math.floor(y0 * image.height / 1000),
                       math.ceil(x1 * image.width / 1000), math.ceil(y1 * image.height / 1000)]
                target_path = crops / (region["id"] + ".png")
                with image.crop(box) as crop:
                    crop.save(target_path)
            relative = target_path.relative_to(out).as_posix()
            evidence.append({**region, "source_sha256": reports[region["document_id"]]["source_sha256"],
                             "page_image_sha256": page["page"]["sha256"], "bbox_pixels": box,
                             "crop_path": relative, "crop_sha256": file_hash(target_path)})
            proposed = html.escape(json.dumps(region["target"], ensure_ascii=False))
            label = html.escape(region["id"])
            cards.append(f'<section><h2>{label} — trang {region["page"]}</h2>'
                         f'<img src="{relative}" alt="Ảnh bằng chứng">'
                         f'<pre>{proposed}</pre><p>Nhãn nháp của AI; chưa duyệt độc lập.</p></section>')
        report = {"status": "draft_review_bundle", "seed_sha256": file_hash(seed_path),
                  "documents": {k: {"pages": len(v["pages"]),
                                     "tiles": sum(len(p["tiles"]) for p in v["pages"]),
                                     "source_sha256": v["source_sha256"]} for k, v in reports.items()},
                  "regions": evidence, "questions": seed["questions"],
                  "permission": seed["permission"], "training_eligible": False,
                  "independent_human_review": "not_performed", "neural_training_run": False}
        write_json(out / "draft_review.json", report)
        text = ('<!doctype html><html lang="vi"><meta charset="utf-8">'
                '<title>H2LM — Nhãn scan thật</title>'
                '<style>body{font:17px sans-serif;max-width:1000px;margin:auto;padding:24px}'
                'img{max-width:100%;border:1px solid #aaa}pre{white-space:pre-wrap}'
                'section{margin-bottom:35px}</style><h1>H2LM — Bộ nhãn scan thật đầu tiên</h1>'
                '<p>Ảnh thật do người dùng cung cấp. Nhãn AI nháp, chưa duyệt độc lập; '
                'không phải kết quả model H2LM hay điểm OCR. Không xác minh chữ ký số.</p>'
                + ''.join(cards) + '</html>')
        (out / "review.html").write_text(text, encoding="utf-8")
        return report
    except (Exception, KeyboardInterrupt) as error:
        write_json(out / "FAILED.json", {"status": "failed", "error_type": type(error).__name__,
                                         "training_eligible": False})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local review of unapproved scan labels")
    parser.add_argument("--seed", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        report = build_review(args.seed, args.input, args.output)
        print(f"Prepared {len(report['documents'])} documents / {len(report['regions'])} draft regions")
        print("NOT approved for training. Open the local review.html and compare every target.")
        return 0
    except Exception:
        logger.exception("Review preparation failed")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
