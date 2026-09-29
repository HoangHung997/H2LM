from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pypdf
import pypdfium2 as pdfium
import pytest
from PIL import Image, ImageChops, ImageDraw
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, RectangleObject

from h2lm.scans import ingest
from h2lm.scans.config import ScanConfig, tiles

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pdf_file(tmp_path):
    path = tmp_path / "Bản scan thử & khoảng trắng.pdf"
    with Image.new("RGB", (360, 480), "white") as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((5, 5, 150, 12), fill="black")
        draw.line((1, 1, 359, 479), fill="black", width=2)
        draw.ellipse((260, 330, 355, 465), outline="red", width=3)
        image.save(path, "PDF", resolution=72)
    return path


@pytest.fixture
def config():
    return ScanConfig(dpi=72, tile_size=128, overlap=20, preview_size=128)


@pytest.mark.parametrize("key,value", [
    ("dpi", 0), ("dpi", 601), ("dpi", True), ("dpi", "300"),
    ("tile_size", 127), ("overlap", -1), ("overlap", 513),
    ("max_pdf_bytes", 0), ("max_document_pages", 0), ("max_selected_pages", 51),
    ("max_page_pixels", 9999), ("max_total_pixels", 0), ("max_tiles_per_page", 0),
    ("max_output_bytes", 0), ("timeout_seconds", 0), ("timeout_seconds", 601),
    ("preview_size", 3000),
])
def test_config_rejects_limits(key, value):
    with pytest.raises(ValueError):
        ScanConfig(**{key: value})


def test_config_unknown_keys_and_overlap(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("silent_downsample: true\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ScanConfig.load(path)
    with pytest.raises(ValueError):
        ScanConfig(tile_size=128, overlap=128)


@pytest.mark.parametrize("w,h,size,overlap", [(10, 20, 32, 4), (32, 32, 32, 0),
    (77, 101, 32, 4), (120, 60, 32, 31), (65, 65, 32, 0), (1, 1, 1, 0)])
def test_tiles_cover_every_pixel_exactly_once_or_more(w, h, size, overlap):
    boxes = tiles(w, h, size, overlap)
    covered = bytearray(w * h)
    for x0, y0, x1, y1 in boxes:
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        assert x1 - x0 <= size and y1 - y0 <= size
        for y in range(y0, y1):
            covered[y * w + x0:y * w + x1] = bytes([1]) * (x1 - x0)
    assert all(covered)
    assert len(boxes) == len(set(boxes))


@pytest.mark.parametrize("dims", [(0, 1, 1, 0), (1, -1, 1, 0), (2, 2, 2, 2), (True, 1, 1, 0)])
def test_bad_tile_args(dims):
    with pytest.raises(ValueError):
        tiles(*dims)


def test_image_only_pdf_preserves_source_and_tiles(pdf_file, config, tmp_path):
    before = pdf_file.read_bytes()
    out = tmp_path / "prepared"
    report = ingest.prepare_pdf(pdf_file, out, config)
    assert pdf_file.read_bytes() == before == (out / "source.pdf").read_bytes()
    assert report["status"] == "prepared_images_only"
    assert report["ocr_performed"] is report["neural_model_run"] is False
    page = report["pages"][0]
    assert page["native_text_chars"] == 0 and page["vision_required"] is True
    assert page["expected_text"] is None and page["eligible_for_supervised_training"] is False
    with Image.open(out / page["page"]["path"]) as image:
        for tile in page["tiles"]:
            with (
                Image.open(out / tile["path"]) as crop,
                image.crop(tile["bbox_pixels_xyxy"]) as expected,
            ):
                assert ImageChops.difference(crop, expected).getbbox() is None
            assert ingest.file_hash(out / tile["path"]) == tile["sha256"]
    assert (out / "manifest.json").exists() and not (out / "FAILED.json").exists()


def hidden_text_pdf(source, destination):
    writer = pypdf.PdfWriter(clone_from=source)
    page = writer.pages[0]
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page["/Resources"][NameObject("/Font")] = DictionaryObject({NameObject("/F1"): writer._add_object(font)})
    stream = DecodedStreamObject()
    stream.set_data(page.get_contents().get_data() +
                    b"\nBT /F1 12 Tf 3 Tr 10 20 Td (WRONG HIDDEN 9999) Tj ET\n")
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(destination)
    writer.close()


def test_false_hidden_ocr_is_never_ground_truth(pdf_file, config, tmp_path):
    source = tmp_path / "wrong-hidden.pdf"
    hidden_text_pdf(pdf_file, source)
    report = ingest.prepare_pdf(source, tmp_path / "prepared", config)
    page = report["pages"][0]
    assert page["native_text_chars"] > 0
    assert page["native_text_state"] == "present_untrusted"
    assert "WRONG HIDDEN" not in json.dumps(report)
    assert page["expected_text"] is None
    assert report["native_text_policy"] == "never_used_as_input_or_ground_truth"


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_pdf_coordinate_roundtrip_with_crop_and_rotation(pdf_file, config, tmp_path, rotation):
    from pypdfium2 import raw

    altered = tmp_path / "cropped.pdf"
    writer = pypdf.PdfWriter(clone_from=pdf_file)
    writer.pages[0].cropbox = RectangleObject((10, 20, 350, 460))
    writer.pages[0].rotate(90)
    writer.write(altered)
    writer.close()
    report = ingest.prepare_pdf(altered, tmp_path / f"out{rotation}", config, rotation=rotation)
    record = report["pages"][0]
    w, h = record["render_canvas_pixels"]
    assert record["pdf_rotation"] == 90
    assert [w, h] == ([440, 340] if rotation in (0, 180) else [340, 440])
    with pdfium.PdfDocument(altered) as pdf, closing(pdf[0]) as page:
        for tile in record["tiles"]:
            x0, y0, x1, y1 = tile["bbox_pixels_xyxy"]
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            for (px, py), (x, y) in zip(tile["quad_pdf_canvas_units"], corners):
                dx, dy = ctypes.c_int(), ctypes.c_int()
                assert raw.FPDF_PageToDevice(page.raw, 0, 0, w, h, rotation // 90,
                                            px, py, ctypes.byref(dx), ctypes.byref(dy))
                assert abs(dx.value - x) <= 1 and abs(dy.value - y) <= 1
                assert 10 - 1e-6 <= px <= 350 + 1e-6
                assert 20 - 1e-6 <= py <= 460 + 1e-6


def test_blank_page_kept_and_multipage_scope(config, tmp_path):
    source = tmp_path / "blank.pdf"
    with Image.new("RGB", (200, 220), "white") as image:
        image.save(source, "PDF", save_all=True, append_images=[image, image], resolution=72)
    report = ingest.prepare_pdf(source, tmp_path / "out", config, start_page=2, page_count=1)
    assert report["total_document_pages"] == 3
    assert report["selected_pages"] == [2, 2] and not report["complete_document"]
    assert report["pages"][0]["page_number"] == 2
    quality = report["pages"][0]["quality"]
    assert quality["warnings"] and quality["auto_discard"] is False
    assert quality["readability"] == "not_assessed"


@pytest.mark.parametrize("overrides", [{"max_page_pixels": 10000}, {"max_total_pixels": 10000},
                                      {"max_tiles_per_page": 1}, {"max_output_bytes": 1024}])
def test_resource_limits_do_not_silently_downscale(pdf_file, config, tmp_path, overrides):
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        ingest.prepare_pdf(pdf_file, out, replace(config, **overrides))
    assert not (out / "manifest.json").exists() and (out / "FAILED.json").exists()
    assert not list(out.rglob("*.png"))


@pytest.mark.parametrize("args", [{"start_page": 0}, {"start_page": True}, {"page_count": 0},
                                  {"page_count": 999}, {"rotation": 45}, {"rotation": True},
                                  {"split": "all"}, {"family_id": ""}])
def test_invalid_request_rejected_early(pdf_file, config, tmp_path, args):
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        ingest.prepare_pdf(pdf_file, out, config, **args)
    assert not out.exists()


def test_out_of_range_and_encryption(pdf_file, config, tmp_path):
    with pytest.raises(ValueError):
        ingest.prepare_pdf(pdf_file, tmp_path / "range", config, start_page=2)
    writer = pypdf.PdfWriter(clone_from=pdf_file)
    writer.encrypt("secret-test-only")
    enc = tmp_path / "encrypted.pdf"
    writer.write(enc)
    writer.close()
    with pytest.raises(ValueError):
        ingest.prepare_pdf(enc, tmp_path / "enc", config)
    assert not (tmp_path / "enc/manifest.json").exists()


def test_corrupt_header_and_corrupt_body(config, tmp_path):
    for index, data in enumerate((b"not a pdf", b"%PDF-1.7\ncorrupt")):
        source = tmp_path / f"bad{index}.pdf"
        source.write_bytes(data)
        out = tmp_path / f"out{index}"
        with pytest.raises(ValueError):
            ingest.prepare_pdf(source, out, config)
        assert (out / "FAILED.json").exists() and not (out / "manifest.json").exists()


def test_no_overwrite_even_failed_folder(pdf_file, config, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ingest.prepare_pdf(pdf_file, out, config)
    assert (out / "keep.txt").read_text() == "keep"


def test_worker_timeout_leaves_no_completion(pdf_file, config, tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("worker", 1)
    monkeypatch.setattr(ingest.subprocess, "run", timeout)
    out = tmp_path / "out"
    with pytest.raises(subprocess.TimeoutExpired):
        ingest.prepare_pdf(pdf_file, out, config)
    assert (out / "FAILED.json").exists() and not (out / "manifest.json").exists()


def test_family_inherited_and_html_escaped(pdf_file, config, tmp_path):
    out = tmp_path / "out"
    report = ingest.prepare_pdf(pdf_file, out, config, family_id="same-family", split="test",
                                document_id="<script>alert(1)</script>")
    assert report["family_id"] == "same-family" and report["split"] == "test"
    assert report["family_review_required"] is False
    assert "<script>" not in (out / "review.html").read_text(encoding="utf-8")
    assert report["requires_visual_review"] is True  # Declared family is not validated truth.


def test_cli_no_torch_and_path_with_unicode(pdf_file, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    subprocess.run([sys.executable, "-c", ("import h2lm.scans.ingest; import sys; "
                     "assert 'torch' not in sys.modules")], env=env, check=True)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("dpi: 72\n", encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "h2lm.scans.cli", "--pdf", str(pdf_file),
                             "--output", str(tmp_path / "out"), "--config", str(cfg)],
                            env=env, check=False, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_source_change_is_not_accepted(pdf_file, config, tmp_path, monkeypatch):
    original_run = ingest.subprocess.run
    def modified(*args, **kwargs):
        result = original_run(*args, **kwargs)
        with pdf_file.open("ab") as stream:
            stream.write(b"\nchanged")
        return result
    monkeypatch.setattr(ingest.subprocess, "run", modified)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="changed"):
        ingest.prepare_pdf(pdf_file, out, config)
    assert (out / "FAILED.json").exists() and not (out / "manifest.json").exists()
