from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from h2lm.scans.config import ScanConfig
from h2lm.scans.ingest import file_hash
from h2lm.scans.review_seed import build_review, validate_seed

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data/scan_seed_20260929/draft_labels.json"


@pytest.fixture
def seed():
    return json.loads(SEED.read_text(encoding="utf-8"))


def test_real_drafts_are_scoped_and_not_ground_truth(seed):
    validate_seed(seed)
    assert len(seed["documents"]) == 3
    assert sum(d["pages"] for d in seed["documents"]) == 10
    assert len(seed["regions"]) == 28
    assert len(seed["questions"]) == 12
    groups = {d["id"]: d["leakage_group"] for d in seed["documents"]}
    assert groups["dt391"] == groups["thach-bich"]
    assert all(d["split"] == "development_seed" for d in seed["documents"])


@pytest.mark.parametrize("field,value", [("page", 0), ("page", True), ("page", 99),
    ("bbox_1000", [-1, 0, 10, 20]), ("bbox_1000", [0, 0, 1001, 20]),
    ("bbox_1000", [1, 0, 1, 20]), ("bbox_1000", [0, 0, 1.5, 20]),
    ("review_status", "human_verified"), ("training_eligible", True),
    ("document_id", "unknown"), ("id", "../../evil"), ("target", None),
    ("status", "uncertain"), ("status", "invented")])
def test_invalid_region_rejected(seed, field, value):
    seed["regions"][0][field] = value
    with pytest.raises(ValueError):
        validate_seed(seed)


def test_permission_cannot_expand_or_be_missing(seed):
    for permission in ({}, {"public_sharing_allowed_by_user": True,
                           "applies_to_document_ids": ["da700"]}):
        bad = copy.deepcopy(seed)
        bad["permission"] = permission
        with pytest.raises(ValueError):
            validate_seed(bad)


@pytest.mark.parametrize("field,value", [("split", "test"), ("filename", "../test.pdf"),
    ("filename", "C:\\secret.pdf"), ("sha256", "bad"), ("pages", True),
    ("family_id", ""), ("leakage_group", "unknown/group")])
def test_bad_document_metadata_rejected(seed, field, value):
    seed["documents"][0][field] = value
    with pytest.raises(ValueError):
        validate_seed(seed)


def test_duplicate_ids_and_bad_question_links(seed):
    for key in ("documents", "regions", "questions"):
        bad = copy.deepcopy(seed)
        bad[key].append(copy.deepcopy(bad[key][0]))
        with pytest.raises(ValueError):
            validate_seed(bad)
    for evidence in (["unknown-region"], ["tb-date"], []):
        bad = copy.deepcopy(seed)
        bad["questions"][0]["evidence_ids"] = evidence
        with pytest.raises(ValueError):
            validate_seed(bad)


def test_dates_not_conflated_and_table_continuation_declared(seed):
    regions = {r["id"]: r for r in seed["regions"]}
    assert regions["tb-date"]["kind"] == "content_date"
    assert regions["tb-sign-date"]["kind"] == "signature_appearance"
    assert regions["da-table-first"]["page"] == 1
    assert regions["da-table-next"]["page"] == 2
    assert regions["dt-margin"]["target"] is None
    assert regions["dt-margin"]["status"] == "uncertain"


def test_changed_source_and_no_overwrite(seed, tmp_path):
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(seed), encoding="utf-8")
    for document in seed["documents"]:
        (tmp_path / document["filename"]).write_bytes(b"%PDF-invalid")
    with pytest.raises(ValueError, match="checksum"):
        build_review(path, tmp_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_review_smoke_on_synthetic_pdf(tmp_path, seed):
    from PIL import Image

    source = tmp_path / "fixture.pdf"
    with Image.new("RGB", (128, 160), "white") as image:
        image.save(source, "PDF", resolution=72)
    document = seed["documents"][0]
    document.update(filename="fixture.pdf", sha256=file_hash(source), pages=1)
    seed["documents"] = [document]
    seed["regions"] = [seed["regions"][0]]
    seed["questions"] = []
    seed["permission"]["applies_to_document_ids"] = [document["id"]]
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(seed), encoding="utf-8")
    config = ScanConfig(dpi=72, tile_size=128, overlap=20, preview_size=128)
    report = build_review(path, tmp_path, tmp_path / "out", config)
    assert report["training_eligible"] is False
    assert report["neural_training_run"] is False
    assert len(report["regions"]) == 1
    assert (tmp_path / "out/review.html").is_file()
    with pytest.raises(FileExistsError):
        build_review(path, tmp_path, tmp_path / "out", config)
