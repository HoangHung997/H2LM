import copy
import json
import os
import random
import subprocess
import sys
import unicodedata
from dataclasses import replace
from pathlib import Path

import pytest

from h2lm.tokenization.corpus import digest, load_corpus
from h2lm.tokenization.tokenizer import (
    SPECIALS, H2Tokenizer, TokenizerConfig, evaluate, protect, restore,
    train_tokenizer,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/tokenizer_sample/manifest.json"
CONFIG = ROOT / "configs/tokenizer/h2lm_tokenizer_dev.yaml"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    output = tmp_path_factory.mktemp("tokenizer") / "run"
    metadata = train_tokenizer(TokenizerConfig.load(CONFIG), MANIFEST, output)
    return H2Tokenizer.load(output), output, metadata


@pytest.mark.parametrize("text", [
    "", " ", "  ", "\t\r\n\n\f\v", "\x00ABC\x01\x7f",
    "  Tiếng Việt có dấu\t\r\nGiữ nguyên  ",
    unicodedata.normalize("NFD", "Tiếng Việt: điểm đ, hiệu lực."),
    "Điều 12a; khoản 2; điểm đ; MẪU 7654/2088/NĐ-TEST.",
    "Ngày 29/02/2088; 1.234.567,89; -0,002%; 00001.",
    "m² ≠ m2; ½ ≠ 1/2; Ⅳ ≠ IV; ㎏ ≠ kg",
    "▁\ue000\ue0000\ue0001▁▁", "<unk><s></s><pad><image><page><bbox>",
    "中文 日本語 한국어 Ελληνικά العربية 🧪🧭", "a\u00a0b\u200bc\u2028d\u2029e",
])
def test_exact_roundtrip(trained, text):
    tokenizer, _, _ = trained
    ids = tokenizer.encode(text)
    assert not set(ids).intersection(range(len(SPECIALS)))
    assert tokenizer.decode(ids) == text


def test_unicode_fuzz_roundtrip(trained):
    tokenizer, _, _ = trained
    rng = random.Random(29)
    for _ in range(100):
        codepoints = [rng.randrange(0x110000) for _ in range(20)]
        text = "".join(chr(cp) for cp in codepoints if not 0xD800 <= cp <= 0xDFFF)
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_codec_is_reversible():
    for text in ["▁", "\ue000", "\ue0000", "\ue0001", "▁\ue000\ue0001"]:
        assert restore(protect(text)) == text
    with pytest.raises(ValueError, match="escape"):
        restore("\ue000")


def test_special_ids_are_explicit(trained):
    tokenizer, _, _ = trained
    assert [tokenizer.special_id(piece) for piece in SPECIALS] == list(range(10))
    assert tokenizer.encode("mẫu", bos=True, eos=True)[0] == 1
    assert tokenizer.encode("mẫu", bos=True, eos=True)[-1] == 2
    assert tokenizer.decode(tokenizer.encode("mẫu", bos=True, eos=True)) == "mẫu"
    with pytest.raises(ValueError):
        tokenizer.special_id("missing")
    for ids in [[-1], [tokenizer.vocab_size], [True], [0]]:
        with pytest.raises(ValueError):
            tokenizer.decode(ids)


def test_heldout_reports_and_train_refusal(trained):
    tokenizer, _, _ = trained
    corpus = load_corpus(MANIFEST)
    for split in ["validation", "test"]:
        report = evaluate(tokenizer, corpus, split)
        assert report["records"] == 12
        assert report["exact_roundtrip_passed"] == 12
        assert report["unknown_tokens"] == 0
        assert report["bytes_per_token"] > 0
        assert all("text" not in case for case in report["cases"])
    with pytest.raises(ValueError, match="not the training"):
        evaluate(tokenizer, corpus, "train")


def test_artifact_guards(trained, tmp_path):
    tokenizer, folder, metadata = trained
    tokenizer.require_model_vocab(tokenizer.vocab_size)
    with pytest.raises(ValueError, match="vocab_size"):
        tokenizer.require_model_vocab(tokenizer.vocab_size + 1)
    with pytest.raises(FileExistsError):
        train_tokenizer(TokenizerConfig(), MANIFEST, folder)
    model = (folder / "tokenizer.model").read_bytes()
    for changed in [dict(metadata, codec_version="wrong"), dict(metadata, model_sha256="bad"),
                    dict(metadata, actual_vocab_size=1)]:
        with pytest.raises(ValueError):
            H2Tokenizer(model, changed)
    with pytest.raises(ValueError):
        H2Tokenizer(model + b"broken", metadata)


def test_repeated_training_identity(trained, tmp_path):
    _, folder, _ = trained
    other = tmp_path / "repeat"
    train_tokenizer(TokenizerConfig.load(CONFIG), MANIFEST, other)
    assert (other / "tokenizer.model").read_bytes() == (folder / "tokenizer.model").read_bytes()


@pytest.mark.parametrize("kwargs", [
    {"vocab_size": 100}, {"vocab_size": True}, {"vocab_size": "1024"},
    {"model_type": "pretrained"}, {"split_digits": "false"}, {"max_records": 0},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        TokenizerConfig(**kwargs)


def make_manifest(tmp_path, *, same_family=False, same_text=False, teacher=False):
    manifest = copy.deepcopy(json.loads(MANIFEST.read_text(encoding="utf-8")))
    rows_by_split = {}
    for source in manifest["sources"]:
        rows = [json.loads(line) for line in (MANIFEST.parent / source["path"])
                .read_text(encoding="utf-8").splitlines()]
        rows_by_split[source["split"]] = rows
    if same_family:
        rows_by_split["test"][0]["family_id"] = rows_by_split["train"][0]["family_id"]
    if same_text:
        rows_by_split["test"][0]["text"] = unicodedata.normalize(
            "NFD", rows_by_split["train"][0]["text"])
    for source in manifest["sources"]:
        raw = ("\n".join(json.dumps(row, ensure_ascii=False) for row in
                         rows_by_split[source["split"]]) + "\n").encode()
        (tmp_path / source["path"]).write_bytes(raw)
        source["sha256"] = digest(raw)
    if teacher:
        manifest["sources"][0]["origin"] = "teacher_verified"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


@pytest.mark.parametrize("option", ["same_family", "same_text"])
def test_leakage_rejected(tmp_path, option):
    path, _ = make_manifest(tmp_path, **{option: True})
    with pytest.raises(ValueError, match="Cross-split leakage"):
        load_corpus(path)


def test_unreviewed_teacher_rejected(tmp_path):
    path, _ = make_manifest(tmp_path, teacher=True)
    with pytest.raises(ValueError, match="verified provenance"):
        load_corpus(path)


@pytest.mark.parametrize("mutation,match", [
    ("checksum", "Checksum"), ("path", "inside"), ("approval", "approved"),
    ("source_id", "Duplicate source"), ("split", "Source split"),
])
def test_manifest_guards(tmp_path, mutation, match):
    path, manifest = make_manifest(tmp_path)
    source = manifest["sources"][0]
    if mutation == "checksum":
        source["sha256"] = "0" * 64
    if mutation == "path":
        source["path"] = "../secrets.jsonl"
    if mutation == "approval":
        source["approved"] = False
    if mutation == "source_id":
        source["id"] = manifest["sources"][1]["id"]
    if mutation == "split":
        source["split"] = "other"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_corpus(path)


def test_bounded_corpus_and_long_sentence(tmp_path):
    with pytest.raises(ValueError, match="byte budget"):
        load_corpus(MANIFEST, max_corpus_bytes=10)
    with pytest.raises(ValueError, match="record budget"):
        load_corpus(MANIFEST, max_records=1)
    with pytest.raises(ValueError, match="record too long"):
        train_tokenizer(TokenizerConfig(max_sentence_bytes=8), MANIFEST, tmp_path / "long")
    assert not (tmp_path / "long").exists()


def test_external_evaluation_checked_against_original_training(trained):
    tokenizer, _, _ = trained
    corpus = load_corpus(MANIFEST)
    # Even a new manifest cannot relabel a trained document as evaluation.
    corpus.records = [replace(corpus.split("train")[0], split="test", id="relabeled")]
    with pytest.raises(ValueError, match="original training data"):
        evaluate(tokenizer, corpus, "test")


def test_cli_and_no_torch_import(trained, tmp_path):
    _, folder, _ = trained
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    subprocess.run([sys.executable, "-c", "import sys; import h2lm.tokenization; "
                    "assert 'torch' not in sys.modules"], env=env, check=True)
    command = [sys.executable, "-m", "h2lm.tokenization.cli", "evaluate",
               "--tokenizer", str(folder), "--manifest", str(MANIFEST),
               "--report", str(tmp_path / "report.json")]
    assert subprocess.run(command, env=env, capture_output=True).returncode == 0
    assert subprocess.run(command, env=env, capture_output=True).returncode == 2


def test_unigram_candidate(tmp_path):
    folder = tmp_path / "unigram"
    train_tokenizer(TokenizerConfig(model_type="unigram"), MANIFEST, folder)
    tokenizer = H2Tokenizer.load(folder)
    report = evaluate(tokenizer, load_corpus(MANIFEST))
    assert report["exact_roundtrip_passed"] == report["records"]


def test_unknown_yaml_key_rejected(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("vocab_szie: 1234\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        TokenizerConfig.load(config)


def test_same_split_duplicate_removal(tmp_path):
    path, manifest = make_manifest(tmp_path)
    train_path = tmp_path / "train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    duplicate = dict(rows[0], id="duplicate-content-only")
    rows.append(duplicate)
    raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    train_path.write_bytes(raw)
    manifest["sources"][0]["sha256"] = digest(raw)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    corpus = load_corpus(path)
    assert corpus.duplicates_removed == 1
    assert len(corpus.split("train")) == 24
