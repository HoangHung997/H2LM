"""Local bounded PDF -> page images/tiles; original remains authoritative and unchanged."""
from __future__ import annotations

import hashlib
import html
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import ScanConfig, tiles


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def source_fingerprint() -> str:
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _copy_bounded(source: Path, target: Path, limit: int) -> str:
    h = hashlib.sha256()
    total = 0
    with source.open("rb") as inp, target.open("xb") as out:
        header = inp.read(5)
        if header != b"%PDF-":
            raise ValueError("Not a PDF with a valid header")
        out.write(header)
        h.update(header)
        total += len(header)
        for block in iter(lambda: inp.read(1024 * 1024), b""):
            total += len(block)
            if total > limit:
                raise ValueError("Input exceeds max_pdf_bytes")
            out.write(block)
            h.update(block)
    return h.hexdigest()


def prepare_pdf(
    source: str | Path, output: str | Path, config: ScanConfig | None = None, *,
    start_page: int = 1, page_count: int = 5, rotation: int = 0,
    document_id: str | None = None, family_id: str | None = None, split: str = "unassigned",
) -> dict[str, Any]:
    """Read only specified pages. Success is prepared images, NOT recognized text.

    Page numbering is 1-based physical PDF order. rotation is additional clockwise degrees.
    Native PDF parsing runs in a timed subprocess. This is not an OS security sandbox.
    """
    cfg = config or ScanConfig()
    if type(start_page) is not int or start_page < 1:
        raise ValueError("start_page must be positive")
    if type(page_count) is not int or not 1 <= page_count <= cfg.max_selected_pages:
        raise ValueError("Invalid page_count; process documents in explicit bounded batches")
    if type(rotation) is not int or rotation not in (0, 90, 180, 270):
        raise ValueError("rotation must be 0, 90, 180 or 270 clockwise")
    if split not in ("unassigned", "train", "validation", "test"):
        raise ValueError("Invalid split")
    for name, value in (("document_id", document_id), ("family_id", family_id)):
        if value is not None and (not isinstance(value, str) or not value.strip()
                                  or len(value) > 200 or any(ord(c) < 32 for c in value)):
            raise ValueError(f"Invalid {name}")
    src, out = Path(source).resolve(), Path(output).absolute()
    if not src.is_file() or src.stat().st_size > cfg.max_pdf_bytes:
        raise ValueError("Missing PDF or input exceeds max_pdf_bytes")
    # Refuse overwrite, including old failed runs and symlink targets.
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        before = src.stat()
        checksum = _copy_bounded(src, out / "source.pdf", cfg.max_pdf_bytes)
        after = src.stat()
        if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
            raise ValueError("Input changed while snapshot was copied")
        request = {
            "config": asdict(cfg), "source_sha256": checksum,
            "start_page": start_page, "page_count": page_count, "rotation": rotation,
            "document_id": document_id or f"sha256:{checksum}",
            "family_id": family_id or f"unreviewed:{checksum}",
            "family_review_required": family_id is None, "split": split,
            "scan_source_sha256": source_fingerprint(),
        }
        write_json(out / "request.json", request)
        # No shell, URLs, remote requests, API keys or OCR executable.
        with (out / "worker.log").open("xb") as log:
            result = subprocess.run(
                [sys.executable, "-m", "h2lm.scans.worker", str(out / "request.json")],
                stdout=log, stderr=subprocess.STDOUT, timeout=cfg.timeout_seconds, check=False,
            )
        if result.returncode != 0:
            raise ValueError("PDF worker failed; inspect local worker.log (no completed manifest)")
        report = json.loads((out / "prepared.json").read_text(encoding="utf-8"))
        if report.get("source_sha256") != checksum or file_hash(src) != checksum:
            raise ValueError("Input changed during preparation")
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        # Completion marker only after worker succeeds and snapshot agrees with original.
        write_json(out / "manifest.json", report)
        return report
    except (Exception, KeyboardInterrupt) as error:
        write_json(out / "FAILED.json", {"status": "failed", "error_type": type(error).__name__,
                                         "message": str(error), "usable_for_training": False})
        raise


class OutputBudget:
    """Logical output cap, not an operating-system disk/memory quota."""
    def __init__(self, root: Path, limit: int) -> None:
        self.root, self.limit = root, limit
        self.used = sum(p.stat().st_size for p in root.iterdir() if p.is_file())

    def save_image(self, image: Any, relative: str) -> dict[str, Any]:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # Worst-case RGB PNG plus overhead. Fail before writing, never silently downsample.
        upper_bound = image.width * image.height * 4 + 65536
        if self.used + upper_bound > self.limit:
            raise ValueError("Output budget exhausted; reduce page batch, not image quality silently")
        with path.open("xb") as stream:
            image.save(stream, format="PNG")
        self.used += path.stat().st_size
        return {"path": relative, "sha256": file_hash(path), "pixels": list(image.size)}


def _pdf_point(page: Any, width: int, height: int, rotation: int, x: int, y: int) -> list[float]:
    import ctypes

    import pypdfium2.raw as raw

    px, py = ctypes.c_double(), ctypes.c_double()
    ok = raw.FPDF_DeviceToPage(page.raw, 0, 0, width, height, rotation // 90,
                              x, y, ctypes.byref(px), ctypes.byref(py))
    if not ok or not all(math.isfinite(v) for v in (px.value, py.value)):
        raise ValueError("Cannot map rendered coordinates back to PDF")
    return [px.value, py.value]


def _quality(image: Any) -> dict[str, Any]:
    from PIL import ImageChops, ImageStat

    with image.convert("L") as gray:
        # Measurements are advisory. A blank-looking page may contain a faint stamp/text.
        histogram = gray.histogram()
        n = gray.width * gray.height
        dark = sum(histogram[:200]) / n
        stats = ImageStat.Stat(gray)
        with gray.resize((256, 256)) as small:
            with ImageChops.offset(small, 1, 0) as offset:
                with ImageChops.difference(small, offset) as diff:
                    edge = ImageStat.Stat(diff).mean[0]
        warnings = []
        if dark < 0.0005:
            warnings.append("blank_or_faint_candidate_manual_review")
        if stats.stddev[0] < 12:
            warnings.append("low_contrast_candidate_manual_review")
        return {"mean_luma": stats.mean[0], "std_luma": stats.stddev[0],
                "dark_fraction": dark, "horizontal_edge_indicator": edge,
                "warnings": warnings, "readability": "not_assessed",
                "calibrated_quality_score": None, "auto_discard": False}


def render_request(request_path: str | Path) -> None:
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw
    from PIL import Image

    request_path = Path(request_path).resolve()
    out = request_path.parent
    request = json.loads(request_path.read_text(encoding="utf-8"))
    cfg = ScanConfig(**request["config"])
    source = out / "source.pdf"
    if file_hash(source) != request["source_sha256"]:
        raise ValueError("Snapshot checksum mismatch")
    pages = []
    budget = OutputBudget(out, cfg.max_output_bytes)
    scale = cfg.dpi / 72.0
    with pdfium.PdfDocument(source) as pdf:
        total_pages = len(pdf)
        if total_pages < 1 or total_pages > cfg.max_document_pages:
            raise ValueError("Document page count exceeds configured limit")
        if raw.FPDF_GetFormType(pdf.raw) != 0:
            raise ValueError("Interactive forms need separately reviewed flattening; not silently omitted")
        first = request["start_page"] - 1
        if first >= total_pages:
            raise ValueError("Start page outside document")
        last = min(first + request["page_count"], total_pages)
        rotation = request["rotation"]
        planned = []
        # Preflight selected pages before any large bitmap allocation.
        for index in range(first, last):
            with closing(pdf[index]) as page:
                pw, ph = page.get_size()
                if not all(math.isfinite(v) and 0 < v <= 20000 for v in (pw, ph)):
                    raise ValueError("Unsupported page dimensions")
                w, h = math.ceil(pw * scale), math.ceil(ph * scale)
                if rotation in (90, 270):
                    w, h = h, w
                if w * h > cfg.max_page_pixels or max(w, h) > 20000:
                    raise ValueError("Page exceeds max_page_pixels/dimensions; no silent resize")
                boxes = tiles(w, h, cfg.tile_size, cfg.overlap)
                if len(boxes) > cfg.max_tiles_per_page:
                    raise ValueError("Page exceeds max_tiles_per_page")
                planned.append((index, w, h, boxes))
        if sum(w * h for _, w, h, _ in planned) > cfg.max_total_pixels:
            raise ValueError("Selected pages exceed max_total_pixels")
        for index, w, h, boxes in planned:
            with closing(pdf[index]) as page:
                folder = f"pages/{index + 1:05d}"
                with closing(page.render(scale=scale, rotation=rotation, may_draw_forms=False,
                                         rev_byteorder=True)) as bitmap:
                    with bitmap.to_pil().convert("RGB") as image:
                        if image.size != (w, h):
                            raise ValueError("Render dimensions disagree with preflight")
                        full = budget.save_image(image, f"{folder}/page.png")
                        with image.copy() as preview:
                            preview.thumbnail((cfg.preview_size, cfg.preview_size),
                                              Image.Resampling.LANCZOS)
                            overview = budget.save_image(preview, f"{folder}/overview.png")
                        quality = _quality(image)
                        regions = []
                        for tile_index, box in enumerate(boxes):
                            with image.crop(box) as tile:
                                item = budget.save_image(tile, f"{folder}/tile-{tile_index:04d}.png")
                            x0, y0, x1, y1 = box
                            item["bbox_pixels_xyxy"] = list(box)
                            item["quad_pdf_canvas_units"] = [
                                _pdf_point(page, w, h, rotation, x, y)
                                for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
                            ]
                            regions.append(item)
                        # Only count existing PDF text objects. Never extract them as input or labels.
                        with closing(page.get_textpage()) as textpage:
                            native_chars = textpage.count_chars()
                        pages.append({
                            "page_number": index + 1, "pdf_rotation": page.get_rotation(),
                            "additional_clockwise_rotation": rotation, "pdf_bbox": list(page.get_bbox()),
                            "render_canvas_pixels": [w, h], "page": full, "overview": overview,
                            "tiles": regions, "quality": quality,
                            "native_text_chars": native_chars,
                            "native_text_state": ("present_untrusted" if native_chars else "none"),
                            "vision_required": True, "label_status": "unlabeled",
                            "expected_text": None, "eligible_for_supervised_training": False,
                        })
    report = {
        "schema_version": 1, "status": "prepared_images_only", "scope": "M1-scan-data-preparation",
        **{k: request[k] for k in ("source_sha256", "document_id", "family_id", "split",
                                   "family_review_required", "scan_source_sha256")},
        "total_document_pages": total_pages, "selected_pages": [first + 1, last],
        "complete_document": first == 0 and last == total_pages, "pages": pages,
        "source_snapshot": "source.pdf", "config": asdict(cfg),
        "native_text_policy": "never_used_as_input_or_ground_truth",
        "transforms": "render_and_explicit_rotation_only; no_deskew_no_denoise_no_binarize",
        "coordinate_system": "pixel edges, top-left; PDF quads from FPDF_DeviceToPage",
        "dpi_note": "requested dpi assumes 1 PDF canvas unit=1/72 inch; not scanner resolution",
        "engine_versions": {x: importlib.metadata.version(x) for x in ("pypdfium2", "Pillow")},
        "runtime": {"python": sys.version.split()[0], "platform": sys.platform},
        "ocr_performed": False, "neural_model_run": False, "production_ready": False,
        "requires_visual_review": True,
    }
    write_json(out / "prepared.json", report)
    title = html.escape(str(request["document_id"]))
    cards = []
    for p in pages:
        cards.append(f'<section><h2>Trang {p["page_number"]}</h2>'
                     f'<a href="{p["page"]["path"]}"><img loading="lazy" '
                     f'src="{p["overview"]["path"]}" alt="Trang PDF"></a>'
                     f'<p>{len(p["tiles"])} vùng ảnh; chữ ẩn: {p["native_text_state"]}. '
                     'Chưa nhận dạng chữ, chưa đánh giá đọc đúng.</p></section>')
    text = ('<!doctype html><html lang="vi"><meta charset="utf-8"><title>H2LM Scan Review</title>'
            '<style>body{max-width:1000px;margin:auto;font:18px sans-serif;padding:24px}'
            'img{max-width:100%;border:1px solid #ccc}section{margin-bottom:32px}</style>'
            '<h1>H2LM — Kiểm tra ảnh scan</h1>'
            '<p>Chỉ chuẩn bị ảnh. Không phải kết quả OCR hay phân tích pháp luật. '
            'Chỉ dùng khi manifest.json tồn tại và không có FAILED.json.</p>'
            f'<p>{title}</p>' + ''.join(cards) + '</html>')
    (out / "review.html").write_text(text, encoding="utf-8")
