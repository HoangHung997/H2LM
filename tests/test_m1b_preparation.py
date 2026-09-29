import copy
import json
import runpy
from pathlib import Path

import pytest

from h2lm.tokenization.corpus import digest, load_corpus
from h2lm.tokenization.preparation import chunk_text, prepare_corpus
from h2lm.tokenization.tokenizer import protect

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def registry(tmp_path):
    create = runpy.run_path(str(ROOT / "scripts/corpus_benchmark_demo.py"))["create_fixture_registry"]
    path = create(tmp_path / "inputs")
    return path, json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_prepare_preserves_exact_source_and_separates_holdout(registry, tmp_path):
    path, raw = registry
    text = ("  Điều 01. Khoa\u0309n 2\r\n\tSố: 009/2020/TT-XX; 1.234,50; "
            "\ue000▁\n" * 15) + "   \r\n"
    data = text.encode("utf-8")
    doc = raw["documents"][0]
    (path.parent / doc["path"]).write_bytes(data)
    doc["sha256"] = digest(data)
    raw["chunk_bytes"] = 128
    save(path, raw)
    output = tmp_path / "prepared"
    result = prepare_corpus(path, output)
    dev = load_corpus(output / "development_manifest.json")
    holdout = load_corpus(output / "holdout_manifest.json")
    assert not dev.split("test")
    assert len(holdout.split("test")) == 12
    assert result["production_ready"] is False
    shard = output / "train/source-0000.jsonl"
    rows = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines()]
    assert "".join(r["text"] for r in rows) == text
    assert all(text[r["source_char_start"]:r["source_char_end"]] == r["text"] for r in rows)
    assert all(len(protect(r["text"]).encode("utf-8")) <= 128 for r in rows)
    assert all(r["raw_document_sha256"] == digest(data) for r in rows)


@pytest.mark.parametrize("mutation", ["approval", "hash", "outside", "sensitivity", "origin",
                                      "reviewer", "family", "document_id", "path", "split", "teacher"])
def test_registry_guards_fail_before_publish(registry, tmp_path, mutation):
    path, raw = registry
    doc = raw["documents"][0]
    if mutation == "approval":
        doc["approved"] = False
    elif mutation == "hash":
        doc["sha256"] = "0" * 64
    elif mutation == "outside":
        doc["path"] = "../private.txt"
    elif mutation == "sensitivity":
        doc["sensitivity"] = "unknown"
    elif mutation == "origin":
        doc["origin"] = "downloaded-unreviewed"
    elif mutation == "reviewer":
        doc.pop("reviewed_by")
    elif mutation == "family":
        raw["documents"][24]["family_id"] = doc["family_id"]
    elif mutation == "document_id":
        raw["documents"][1]["id"] = doc["id"]
    elif mutation == "path":
        raw["documents"][1]["path"] = doc["path"]
    elif mutation == "split":
        doc["split"] = "other"
    else:
        doc["origin"] = "teacher_verified"
    save(path, raw)
    output = tmp_path / "prepared"
    with pytest.raises((ValueError, TypeError, OSError)):
        prepare_corpus(path, output)
    assert not output.exists()


@pytest.mark.parametrize("text", ["abc\x00def", "abc\ufffddef", "   \n\t"])
def test_bad_extractions_are_not_silently_cleaned(registry, tmp_path, text):
    path, raw = registry
    doc = raw["documents"][0]
    data = text.encode("utf-8")
    (path.parent / doc["path"]).write_bytes(data)
    doc["sha256"] = digest(data)
    save(path, raw)
    with pytest.raises(ValueError):
        prepare_corpus(path, tmp_path / "out")


def test_duplicate_full_text_in_different_split_blocked(registry, tmp_path):
    path, raw = registry
    first, other = raw["documents"][0], raw["documents"][24]
    data = (path.parent / first["path"]).read_bytes()
    (path.parent / other["path"]).write_bytes(data)
    other["sha256"] = digest(data)
    save(path, raw)
    with pytest.raises(ValueError, match="document_text"):
        prepare_corpus(path, tmp_path / "out")


def test_no_overwrite_and_deterministic_preparation(registry, tmp_path):
    path, _ = registry
    one, two = tmp_path / "one", tmp_path / "two"
    prepare_corpus(path, one)
    prepare_corpus(path, two)
    for f in one.rglob("*"):
        if f.is_file():
            assert f.read_bytes() == (two / f.relative_to(one)).read_bytes()
    before = (one / "development_manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_corpus(path, one)
    assert before == (one / "development_manifest.json").read_bytes()


def test_teacher_provenance_is_propagated_not_generated(registry, tmp_path):
    path, raw = registry
    doc = raw["documents"][0]
    doc["origin"] = "teacher_verified"
    doc["teacher"] = {"provider": "fixture", "model": "fixture", "prompt_version": "v0",
                      "run_id": "fixture-only", "source_sha256": "1" * 64, "review_status": "verified"}
    expected = copy.deepcopy(doc["teacher"])
    save(path, raw)
    prepare_corpus(path, tmp_path / "out")
    manifest = json.loads((tmp_path / "out/development_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sources"][0]["teacher"] == expected


@pytest.mark.parametrize("max_bytes", [0, True, 63, 32769, "128"])
def test_chunk_budget_validation(max_bytes):
    with pytest.raises(ValueError):
        chunk_text("Tiếng Việt", max_bytes)


def test_chunk_whitespace_and_unicode():
    with pytest.raises(ValueError, match="Whitespace"):
        chunk_text("text" + " " * 300 + "end", 64)
    text = "a" * 64 + "   "
    chunks = chunk_text(text, 64)
    assert "".join(t for _, _, t in chunks) == text
    assert all(len(protect(t).encode("utf-8")) <= 64 for _, _, t in chunks)
