from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from h2lm.experiments.data import decode, dump, encode, load, sha
from h2lm.experiments.engine import (
    batch,
    make_model,
    predict,
    read_checkpoint,
    train,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    c = yaml.safe_load((ROOT / "configs/training/micro_vlm.yaml").read_text())
    c.update(steps=4, d_model=32, image_size=32, text_layers=1, max_new_tokens=4, save_every=2)
    return c


@pytest.fixture
def manifest(tmp_path):
    rows = []
    for i, shade in enumerate((0, 128, 255)):
        path = tmp_path / f"{i}.png"
        with Image.new("L", (32, 32), shade) as image:
            image.save(path)
        rows.append({"id": str(i), "group": str(i), "image": path.name, "sha256": sha(path),
                     "prompt": "Đọc?", "target": str(i),
                     "split": "validation" if i == 2 else "train",
                     "label_kind": "synthetic_programmatic", "production_approved": False,
                     "synthetic_fields": [i, i + 1]})
    path = tmp_path / "manifest.json"
    dump(path, {"schema_version": 1, "experiment_only": True, "records": rows})
    return path


@pytest.mark.parametrize("text", ["Điều 15, khoản 2", "123/QĐ-TCT", "đ ă â ơ ư\r\n", "🥇", ""])
def test_byte_codec_exact(text):
    assert decode(encode(text)) == text


@pytest.mark.parametrize("key,value", [("steps", 0), ("steps", True), ("steps", 99999),
    ("seconds", 601), ("threads", 0), ("image_size", 512), ("learning_rate", float("nan")),
    ("learning_rate", -1), ("d_model", 33), ("allow_silver", "yes"), ("patch_size", 13)])
def test_invalid_config(cfg, key, value):
    cfg[key] = value
    with pytest.raises(ValueError):
        validate(cfg)


def test_prompt_mask_padding_and_target_shift(manifest, cfg):
    rows = load(manifest, 32, 96)
    ids, pixels, labels = batch(rows)
    assert pixels.shape == (3, 1, 32, 32)
    for r, label in zip(rows, labels):
        assert torch.all(label[:len(r["prefix"])] == -100)
        assert label[len(r["prefix"])] == encode(r["target"])[0]
    assert torch.equal(ids[:, -1], torch.full((3,), 2))


def test_group_leakage_and_tamper(manifest):
    data = json.loads(manifest.read_text())
    original = copy.deepcopy(data)
    data["records"][2]["group"] = "0"
    dump(manifest, data)
    with pytest.raises(ValueError, match="leakage"):
        load(manifest, 32, 96)
    dump(manifest, original)
    (manifest.parent / "0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load(manifest, 32, 96)


def test_draft_opt_in_does_not_change_approval(manifest):
    data = json.loads(manifest.read_text())
    data["records"][0]["label_kind"] = "silver_ai_draft"
    dump(manifest, data)
    with pytest.raises(ValueError, match="explicitly"):
        load(manifest, 32, 96)
    assert not load(manifest, 32, 96, allow_silver=True)[0]["production_approved"]


def test_layers_independent_and_causality(cfg):
    cfg["text_layers"] = 2
    model = make_model(cfg).eval()
    a, b = model.decoder.layers
    assert not torch.equal(a.self_attn.in_proj_weight, b.self_attn.in_proj_weight)
    ids = torch.tensor([[1, 12, 16, 19, 22]])
    changed = ids.clone()
    changed[0, 3:] = 44
    image = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        original = model(ids, image)["logits"]
        future_changed = model(changed, image)["logits"]
    torch.testing.assert_close(original[:, :3], future_changed[:, :3], atol=1e-5, rtol=1e-5)


def test_resume_exact_equivalence(manifest, tmp_path, cfg):
    a, b = tmp_path / "full", tmp_path / "segmented"
    full = train(manifest, a, cfg)
    part = train(manifest, b, cfg, segment_steps=2)
    assert part["step"] == 2 and not part["completed_steps"]
    resumed = train(manifest, b, cfg, resume=True)
    assert resumed["step"] == full["step"] == 4
    for k, v in read_checkpoint(a)["model"].items():
        assert torch.equal(v, read_checkpoint(b)["model"][k]), k
    assert full["weights_changed"] and full["vision_patch_max_change"] > 0
    assert full["reload_matches"] and resumed["reload_matches"]
    assert not full["production_ready"]
    assert full["final_train_loss"] < full["initial_train_loss"]


def test_resume_mismatch_and_checkpoint_corruption(manifest, tmp_path, cfg):
    out = tmp_path / "run"
    train(manifest, out, cfg, segment_steps=2)
    changed = dict(cfg, seed=30)
    with pytest.raises(ValueError, match="mismatch"):
        train(manifest, out, changed, resume=True)
    ref = json.loads((out / "latest.json").read_text())
    (out / ref["file"]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        read_checkpoint(out)
    assert not (out / "RUNNING.lock").exists()


def test_no_overwrite_and_lock(manifest, tmp_path, cfg):
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(FileExistsError):
        train(manifest, out, cfg)
    (out / "RUNNING.lock").write_text("another writer")
    with pytest.raises(FileExistsError):
        train(manifest, out, cfg, resume=True)
    assert (out / "RUNNING.lock").read_text() == "another writer"


def test_prediction_has_no_target_or_filename_input(cfg):
    model = make_model(cfg)
    result = predict(model, torch.ones(1, 32, 32), "Đọc?", 3)
    assert set(result) == {"text", "valid_utf8", "eos"}
