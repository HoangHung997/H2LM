from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import pytest
import torch

from h2lm.scale.config import ScaleConfig, tiny_config
from h2lm.scale.fixture import decode, encode, example
from h2lm.scale.model import build, parameter_report
from h2lm.scale.optim import AuditedSGD
from h2lm.scale.runner import resource_gate, run, verify
from h2lm.scale.storage import load_into, save


def sample():
    item = example(0, tiny_config().max_pages, 32)
    return {k: v for k, v in item.items() if k != "provenance"}


def test_billion_count_is_unique_and_components_sum():
    report = parameter_report(build(ScaleConfig(), device="meta"))
    assert report["unique_parameters"] == 1_001_571_584
    assert report["trainable_parameters"] == report["unique_parameters"]
    assert report["aliases"] == 0 and report["parameter_tensors"] == 361
    assert report["components"] == {"vision": 85_543_680, "fusion": 8_485_632,
                                     "language": 907_542_272}
    assert sum(report["components"].values()) == report["unique_parameters"]
    assert not report["initialized_storage"]  # Meta count alone is never reported as training.
    assert report["bf16_weight_bytes"] == 2_003_143_168


@pytest.mark.parametrize("key,value", [("layers", 0), ("layers", True), ("layers", 33),
    ("heads", 13), ("kv_heads", 3), ("d_model", 1793), ("vision_dim", 767),
    ("resampler_tokens", 0), ("max_tiles", 99), ("max_total_tokens", 16),
    ("vocab_size", 265), ("max_image_side", 513), ("max_pages", 0), ("name", "")])
def test_invalid_configuration(key, value):
    with pytest.raises(ValueError):
        replace(ScaleConfig(), **{key: value})


def test_config_typo_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("layres: 26\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ScaleConfig.read(path)


@pytest.mark.parametrize("text", ["Số 1542/QĐ-TCT", "Điều 1\r\nKhoản 2", "ơ ư ă đ", "𠀀", ""])
def test_byte_codec_preserves_unicode(text):
    assert decode(encode(text)) == text


def test_initialize_and_every_module_has_gradient():
    torch.set_num_threads(2)
    model = build(tiny_config())
    assert not torch.equal(model.language.blocks[0].attention.q.weight,
                           model.language.blocks[1].attention.q.weight)
    out = model(**sample())
    out["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(float(p.grad.abs().max()) > 0 for p in model.parameters())
    assert parameter_report(model)["initialized_storage"]


def test_causal_text_no_future_leak():
    model = build(tiny_config()).eval()
    data = sample()
    original = model(**data)["logits"].detach()
    changed = data["input_ids"].clone()
    changed[:, -3:] = 60
    data["input_ids"] = changed
    result = model(**data)["logits"].detach()
    torch.testing.assert_close(original[:, :-3], result[:, :-3], rtol=1e-5, atol=1e-6)


def test_padded_tile_does_not_change_valid_predictions():
    model = build(tiny_config()).eval()
    data = sample()
    data["tile_mask"] = torch.tensor([[True, False]])
    first = model(**data)["logits"].detach()
    data["images"] = data["images"].clone()
    data["images"][:, 1] = torch.randn_like(data["images"][:, 1])
    second = model(**data)["logits"].detach()
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)


def test_geometry_and_valid_image_affect_outputs():
    model = build(tiny_config()).eval()
    data = sample()
    first = model(**data)["logits"].detach()
    data["geometry"] = data["geometry"].clone()
    data["geometry"][..., 4] += 0.2
    second = model(**data)["logits"].detach()
    assert float((first - second).abs().max()) > 1e-6
    data["images"] = torch.ones_like(data["images"])
    third = model(**data)["logits"].detach()
    assert float((second - third).abs().max()) > 1e-6


def test_text_only_and_padding_loss():
    model = build(tiny_config()).eval()
    ids = torch.tensor([[1, 25, 26, 2]])
    a = model(ids, labels=ids)
    padded = torch.tensor([[1, 25, 26, 2, 3, 3]])
    b = model(padded, labels=padded)
    torch.testing.assert_close(a["loss"], b["loss"])
    torch.testing.assert_close(a["logits"], b["logits"][:, :4], atol=1e-6, rtol=1e-5)
    assert a["vision_tokens"] == 0


@pytest.mark.parametrize("mutation", ["bad_id", "float_id", "geometry", "nonfinite", "empty_target",
                                     "left_padding", "context", "tile_mask", "image_shape"])
def test_bad_inputs(mutation):
    model = build(tiny_config())
    data = sample()
    if mutation == "bad_id":
        data["input_ids"][0, 0] = 266
    elif mutation == "float_id":
        data["input_ids"] = data["input_ids"].float()
    elif mutation == "geometry":
        data["geometry"][0, 0, 2] = -1
    elif mutation == "nonfinite":
        data["images"][0, 0, 0, 0, 0] = float("nan")
    elif mutation == "empty_target":
        data["labels"][:] = -100
    elif mutation == "left_padding":
        data["input_ids"][0, 0] = 3
    elif mutation == "context":
        data["input_ids"] = torch.ones(1, 129, dtype=torch.long)
    elif mutation == "tile_mask":
        data["tile_mask"] = torch.ones(1, 2)
    elif mutation == "image_shape":
        data["images"] = torch.ones(1, 2, 3, 33, 32)
    with pytest.raises(ValueError):
        model(**data)


@pytest.mark.parametrize("mapped", [False, True])
def test_streaming_sgd_equals_torch_sgd_without_momentum(mapped, tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(29)
    a = build(tiny_config())
    torch.manual_seed(29)
    b = build(tiny_config(), backing_file=tmp_path / "params.bin" if mapped else None)
    expected = torch.optim.SGD(a.parameters(), lr=0.003, foreach=False)
    actual = AuditedSGD(b, 0.003, True)
    try:
        for _ in range(2):
            expected.zero_grad(set_to_none=True)
            a(**sample())["loss"].backward()
            expected.step()
            record = actual.backward(b(**sample())["loss"])
            assert record["parameter_tensors_in_backward"] == len(list(b.parameters()))
            assert record["parameters_in_backward"] == parameter_report(b)["unique_parameters"]
        for name, p in a.named_parameters():
            assert torch.equal(p, dict(b.named_parameters())[name]), name
    finally:
        actual.close()


def test_mapped_ranges_disjoint_and_version_counters_independent(tmp_path):
    model = build(tiny_config(), backing_file=tmp_path / "params.bin")
    params = list(model.parameters())
    ranges = sorted((p.storage_offset(), p.storage_offset() + p.numel()) for p in params)
    assert ranges[0][0] == 0
    for (_, end), (start, _) in pairwise(ranges):
        assert end == start
    assert ranges[-1][1] == parameter_report(model)["unique_parameters"]
    version = params[1]._version
    with torch.no_grad():
        params[0].add_(1)
    assert params[1]._version == version


def test_nonfinite_disconnected_and_accumulation_fail():
    model = build(tiny_config())
    opt = AuditedSGD(model, 0.003)
    with pytest.raises(ValueError):
        opt.backward(torch.tensor(float("nan"), requires_grad=True))
    model.language.head.weight.grad = torch.ones_like(model.language.head.weight)
    with pytest.raises(ValueError, match="accumulation"):
        opt.backward(model(**sample())["loss"])
    model.language.head.weight.grad = None
    model.register_parameter("unused", torch.nn.Parameter(torch.ones(4)))
    opt = AuditedSGD(model, 0.003)
    with pytest.raises(ValueError, match="gradient"):
        opt.backward(model(**sample())["loss"])


def test_sharded_checkpoint_and_checksum_fail_closed(tmp_path):
    model = build(tiny_config(), dtype=torch.bfloat16, backing_file=tmp_path / "arena.bin")
    training = {"learning_rate": 0.003, "completed_steps": 0, "dtype": "bfloat16"}
    index = save(model, tmp_path / "checkpoint", training, shard_bytes=16 * 1024)
    assert len(index["shards"]) > 1
    restored = build(tiny_config(), dtype=torch.bfloat16)
    load_into(restored, tmp_path / "checkpoint")
    for name, p in model.named_parameters():
        assert torch.equal(p, dict(restored.named_parameters())[name])
    # Saved views must not serialize the entire mapped 1B arena once PER shard.
    assert sum(s["file_bytes"] for s in index["shards"]) < 2 * (tmp_path / "arena.bin").stat().st_size
    with pytest.raises(FileExistsError):
        save(model, tmp_path / "checkpoint", training)
    first = tmp_path / "checkpoint" / index["shards"][0]["file"]
    first.write_bytes(b"corruption")
    with pytest.raises(ValueError, match="mismatch"):
        load_into(restored, tmp_path / "checkpoint")


def test_checkpoint_metadata_mismatch(tmp_path):
    model = build(tiny_config())
    save(model, tmp_path / "checkpoint", {"learning_rate": 0.003})
    with pytest.raises(ValueError, match="mismatch"):
        load_into(build(replace(tiny_config(), layers=3)), tmp_path / "checkpoint")


def test_large_is_explicit_and_count_is_not_train(tmp_path):
    with pytest.raises(ValueError, match="allow-large"):
        resource_gate(ScaleConfig(), torch.float32, tmp_path, False, False, False)
    assert not (tmp_path / "index.json").exists()


def test_runner_resume_and_fresh_reload(tmp_path):
    torch.set_num_threads(2)
    cfg = tiny_config()
    full = run(cfg, tmp_path / "full", steps=4, streaming=True, mapped=True)
    part = run(cfg, tmp_path / "part", steps=2, streaming=True, mapped=True)
    resumed = run(cfg, tmp_path / "part", steps=2, streaming=True, mapped=True, resume=True)
    assert part["training"]["completed_steps"] == 2
    assert resumed["training"]["completed_steps"] == 4
    a = build(cfg)
    b = build(cfg)
    load_into(a, tmp_path / "full/checkpoint-0004")
    load_into(b, tmp_path / "part/checkpoint-0004")
    for name, p in a.named_parameters():
        assert torch.equal(p, dict(b.named_parameters())[name]), name
    assert full["parameters"]["unique_parameters"] == resumed["parameters"]["unique_parameters"]
    assert verify(tmp_path / "full", False)["matches_saved_prediction"]
    with pytest.raises(FileExistsError):
        run(cfg, tmp_path / "full", steps=1)
    with pytest.raises(ValueError, match="mismatch"):
        run(cfg, tmp_path / "part", steps=1, lr=0.004, resume=True)
