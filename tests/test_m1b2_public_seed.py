import json
from pathlib import Path

import pytest

from h2lm.tokenization.benchmark import corpus_gate, load_plan
from h2lm.tokenization.corpus import Corpus, digest
from h2lm.tokenization.public_seed import (
    collect_seed,
    discover_pdf,
    extract_pdf,
    identifier,
    inspect_pages,
)
from h2lm.tokenization.seed_audit import (
    check_freeze,
    duplicate_audit,
    evaluate_selected_holdout,
    freeze_corpus,
)
from h2lm.tokenization.seed_provenance import official_url, validate_official_source

ROOT = Path(__file__).resolve().parents[1]


def page(prefix="mẫu"):
    return "Số 22/2023/QH15. Điều 1. Phạm vi áp dụng. Điều 2. Giải thích.\n" + " ".join(
        f"{prefix}{i} dữ liệu tiếng Việt" for i in range(250))


@pytest.mark.parametrize("url", [
    "http://congbao.chinhphu.vn/x", "https://example.com/a.pdf", "https://127.0.0.1/a.pdf",
    "https://congbao.chinhphu.vn.evil.test/a.pdf", "https://user@congbao.chinhphu.vn/a",
    "file:///tmp/a", "https://g7.cdnchinhphu.vn:8080/a.pdf", "https://g7.cdnchinhphu.vn/a#frag",
])
def test_reject_unapproved_destinations(url):
    with pytest.raises(ValueError):
        official_url(url)


def test_discover_unique_pdf_and_no_partial_volumes():
    url = "https://g7.cdnchinhphu.vn/data/2023_22-2023-QH15.pdf"
    html = f'<a href="{url}">pdf</a><a href="{url}">again</a>'.encode()
    assert discover_pdf(html, "https://congbao.chinhphu.vn/doc", "22/2023/QH15") == url
    assert identifier("NĐ-CP") == identifier("N D - C P")
    with pytest.raises(ValueError):
        discover_pdf(b"<html>No file</html>", "https://congbao.chinhphu.vn/doc", "22/2023/QH15")
    with pytest.raises(ValueError):
        discover_pdf(html + html.replace(b"data/", b"another/"),
                     "https://congbao.chinhphu.vn/doc", "22/2023/QH15")


def test_network_requires_opt_in(tmp_path):
    with pytest.raises(ValueError, match="Network disabled"):
        collect_seed(tmp_path / "missing.json", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_page_checks_preserve_exact_offsets():
    pages = [page(), page("bổsung")]
    report = inspect_pages(pages, "22/2023/QH15")
    text = "\n\n".join(pages)
    for source, record in zip(pages, report["page_offsets"]):
        assert text[record["char_start"]:record["char_end"]] == source
    assert report["visual_transcription_verified"] is False
    assert report["current_legal_validity_checked"] is False


@pytest.mark.parametrize("bad", [[page(), ""], [page() + "\ufffd"], [page() + "\ue001"],
                                [page() + "\x00"], [], ["scan"]])
def test_quarantine_not_silent_ocr(bad):
    with pytest.raises(ValueError):
        inspect_pages(bad, "22/2023/QH15")


def test_wrong_identity_fails():
    with pytest.raises(ValueError, match="identity"):
        inspect_pages([page()], "99/1900/QH00")


def test_real_pdf_blank_page_is_rejected(tmp_path):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(300, 400)
    path = tmp_path / "blank.pdf"
    with path.open("wb") as stream:
        writer.write(stream)
    # Run actual parser in a child, so its Linux address-space limit never affects pytest/torch.
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-m", "h2lm.tokenization.public_seed", "extract",
                             "--pdf", str(path), "--code", "22/2023/QH15",
                             "--output", str(tmp_path / "out")], capture_output=True,
                            check=False, timeout=30)
    assert result.returncode == 2
    assert not (tmp_path / "out").exists()


def test_non_pdf_refused(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(sys, "platform", "win32")  # Do not set RLIMIT_AS in the test process.
    path = tmp_path / "fake.pdf"
    path.write_bytes(b"<html>not a PDF</html>")
    with pytest.raises(ValueError, match="PDF"):
        extract_pdf(path, "22/2023/QH15", tmp_path / "out")


def test_page_copy_across_splits_is_detected():
    copied = " ".join(f"word{i}" for i in range(300))
    docs = [{"id": "a", "family_id": "a", "split": "train", "pages": [copied, page("traina")]},
            {"id": "b", "family_id": "b", "split": "test", "pages": [copied, page("testb")]}]
    report = duplicate_audit(docs)
    assert not report["passed"]
    assert any(f["left_unit"] == "page-1" for f in report["flags"])
    docs[1]["split"] = "train"
    assert duplicate_audit(docs)["passed"]
    docs[1].update(family_id="a", split="test")
    with pytest.raises(ValueError, match="family"):
        duplicate_audit(docs)


def test_seed_provenance_is_not_human_review():
    source = {"source_uri": "https://congbao.chinhphu.vn/doc", "sensitivity": "public",
              "verification": {"scope": "tokenizer-seed-only", "method": "pypdf-digital-text-v1",
                               "human_verified": False, "pdf_url": "https://g7.cdnchinhphu.vn/a.pdf",
                               "pdf_sha256": "a" * 64, "text_sha256": "b" * 64,
                               "pages_sha256": "c" * 64}}
    validate_official_source(source)
    source["verification"]["human_verified"] = True
    with pytest.raises(ValueError):
        validate_official_source(source)


def test_seed_cannot_pass_reviewed_pilot_gate():
    corpus = Corpus([], {"sources": [{"origin": "official_pdf_text"}]}, "a" * 64, 0)
    result = corpus_gate(corpus, {"mode": "pilot", "minimums": {}})
    assert not result["passed"]
    corpus.manifest["sources"][0]["origin"] = "synthetic_fixture"
    assert not corpus_gate(corpus, {"mode": "official_seed", "minimums": {}})["passed"]
    assert load_plan(ROOT / "configs/tokenizer/h2lm_comparison_pilot.yaml")["minimums"]["train_families"] == 30


def test_freeze_detects_holdout_change(tmp_path):
    for split, manifest in [("train", "development_manifest.json"), ("test", "holdout_manifest.json")]:
        path = tmp_path / f"{split}.jsonl"
        path.write_text('{"text":"not training data"}\n', encoding="utf-8")
        (tmp_path / manifest).write_text(json.dumps({"sources": [{"path": path.name,
                                                                "sha256": digest(path.read_bytes())}]}))
    freeze = tmp_path / "freeze.json"
    freeze_corpus(tmp_path, freeze)
    check_freeze(tmp_path, freeze)
    with pytest.raises(FileExistsError):
        freeze_corpus(tmp_path, freeze)
    (tmp_path / "test.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        check_freeze(tmp_path, freeze)


def test_frozen_selection_one_holdout_run(tmp_path):
    from h2lm.tokenization.benchmark import compare_tokenizers
    from h2lm.tokenization.tokenizer import H2Tokenizer

    # Existing public synthetic fixture exercises the freeze protocol, not legal quality.

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    fixture = ROOT / "data/tokenizer_sample"
    original = json.loads((fixture / "manifest.json").read_text())
    for src in original["sources"]:
        (prepared / src["path"]).write_bytes((fixture / src["path"]).read_bytes())
    for file, splits in [("development_manifest.json", {"train", "validation"}),
                         ("holdout_manifest.json", {"test"})]:
        (prepared / file).write_text(json.dumps({"schema_version": 1,
                     "sources": [s for s in original["sources"] if s["split"] in splits]}))
    freeze = tmp_path / "freeze.json"
    freeze_corpus(prepared, freeze)
    candidates = tmp_path / "candidates"
    comparison = compare_tokenizers(ROOT / "configs/tokenizer/h2lm_comparison_fixture.yaml",
                                    prepared / "development_manifest.json", candidates)
    result = evaluate_selected_holdout(prepared, freeze, candidates / "comparison-0001.json",
                                       candidates, tmp_path / "evaluation")
    assert result["records"] == result["exact_roundtrip_passed"] == 12
    assert result["selected_candidate"] == comparison["provisional_validation_choice"]
    assert H2Tokenizer.load(candidates / result["selected_candidate"]).vocab_size > 0
    with pytest.raises(FileExistsError):
        evaluate_selected_holdout(prepared, freeze, candidates / "comparison-0001.json",
                                  candidates, tmp_path / "evaluation")
