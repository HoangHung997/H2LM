import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from h2lm.tokenization import benchmark
from h2lm.tokenization.benchmark import (
    compare_tokenizers,
    development_corpus,
    load_plan,
    run_worker,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/tokenizer/h2lm_comparison_fixture.yaml"


@pytest.fixture
def development(tmp_path):
    target = tmp_path / "development"
    shutil.copytree(ROOT / "data/tokenizer_sample", target)
    raw = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    raw["sources"] = [source for source in raw["sources"] if source["split"] != "test"]
    manifest = target / "development_manifest.json"
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    # Comparison must work with no physical test shard available at all.
    (target / "test.jsonl").unlink()
    return manifest


@pytest.fixture
def comparison(development, tmp_path):
    folder = tmp_path / "run"
    report = compare_tokenizers(PLAN, development, folder)
    return development, folder, report


def test_comparison_real_subprocesses_four_candidates(comparison):
    _, folder, report = comparison
    assert report["all_candidates_passed"]
    assert len(report["candidates"]) == 4
    assert report["test_evaluated"] is False
    assert report["production_ready"] is False
    assert all(r["exact_roundtrip_passed"] == 12 and r["unknown_tokens"] == 0 for r in report["candidates"])
    assert all(r["actual_vocab_size"] <= r["requested_vocab_size"] for r in report["candidates"])
    assert all(r["categories"] for r in report["candidates"])
    assert (folder / "comparison-0001.json").exists()
    assert not (folder / "RUNNING.lock").exists()


def test_resume_reuses_verified_artifacts_without_retraining(comparison, monkeypatch):
    manifest, folder, initial = comparison
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed checkpoints must be reused")
    monkeypatch.setattr(benchmark, "run_worker", forbidden)
    report = compare_tokenizers(PLAN, manifest, folder, resume=True)
    assert all(r["reused_checkpoint"] for r in report["candidates"])
    assert report["provisional_validation_choice"] == initial["provisional_validation_choice"]
    assert (folder / "comparison-0002.json").exists()


def test_no_overwrite_resume_mismatch_and_lock(comparison, tmp_path):
    manifest, folder, _ = comparison
    with pytest.raises(FileExistsError):
        compare_tokenizers(PLAN, manifest, folder)
    changed = yaml.safe_load(PLAN.read_text())
    changed["candidates"][0]["vocab_size"] += 1
    plan = tmp_path / "changed.yaml"
    plan.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="Resume rejected"):
        compare_tokenizers(plan, manifest, folder, resume=True)
    (folder / "RUNNING.lock").write_text("pid=example")
    with pytest.raises(RuntimeError, match="locked"):
        compare_tokenizers(PLAN, manifest, folder, resume=True)
    assert (folder / "RUNNING.lock").exists()


def test_corrupt_checkpoint_rejected(comparison):
    manifest, folder, _ = comparison
    model = folder / "bpe-512/tokenizer.model"
    model.write_bytes(model.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="checksum"):
        compare_tokenizers(PLAN, manifest, folder, resume=True)
    assert not (folder / "RUNNING.lock").exists()


def test_test_sources_refused_before_reading(tmp_path):
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps({"schema_version": 1, "sources": [
        {"split": "test", "path": "does-not-exist.jsonl"}]}))
    with pytest.raises(ValueError, match="refuses test"):
        development_corpus(manifest)


@pytest.mark.parametrize("change", ["mode", "name", "duplicate", "budget", "gate", "extra", "vocab"])
def test_invalid_comparison_plan(tmp_path, change):
    plan = yaml.safe_load(PLAN.read_text())
    if change == "mode":
        plan["mode"] = "production"
    elif change == "name":
        plan["candidates"][0]["name"] = "../escape"
    elif change == "duplicate":
        plan["candidates"][1]["name"] = plan["candidates"][0]["name"]
    elif change == "budget":
        plan["timeout_seconds"] = 901
    elif change == "gate":
        plan["minimums"]["train_records"] = False
    elif change == "vocab":
        plan["candidates"][0]["vocab_size"] = 1000000
    else:
        plan["silently_ignored"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(plan))
    with pytest.raises((ValueError, TypeError)):
        load_plan(path)


def test_pilot_gate_does_not_call_fixture_real(development, tmp_path):
    plan = yaml.safe_load(PLAN.read_text())
    plan["mode"] = "pilot"
    path = tmp_path / "pilot.yaml"
    path.write_text(yaml.safe_dump(plan))
    with pytest.raises(ValueError, match="synthetic_fixture"):
        compare_tokenizers(path, development, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_failed_candidate_withholds_choice_and_can_resume(development, tmp_path, monkeypatch):
    original = benchmark.run_worker
    def interrupt(config, manifest, output, seconds):
        return "timeout" if output.name == "bpe-512" else original(config, manifest, output, seconds)
    monkeypatch.setattr(benchmark, "run_worker", interrupt)
    output = tmp_path / "interrupted"
    report = compare_tokenizers(PLAN, development, output)
    assert not report["all_candidates_passed"]
    assert report["provisional_validation_choice"] is None
    monkeypatch.setattr(benchmark, "run_worker", original)
    resumed = compare_tokenizers(PLAN, development, output, resume=True)
    assert resumed["all_candidates_passed"]
    assert sum(r["reused_checkpoint"] for r in resumed["candidates"]) == 3


def test_worker_timeout_is_handled(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("fixture", 1)
    monkeypatch.setattr(subprocess, "run", timeout)
    assert run_worker(tmp_path / "c", tmp_path / "m", tmp_path / "out", 1) == "timeout"


def test_cli_refuses_test_and_does_not_import_torch(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run([sys.executable, "-m", "h2lm.tokenization.benchmark", "--config", str(PLAN),
                             "--manifest", str(ROOT / "data/tokenizer_sample/manifest.json"),
                             "--output", str(tmp_path / "invalid")], capture_output=True,
                            env=env, check=False)
    assert result.returncode == 2
    assert b"refuses test" in result.stderr
    subprocess.run([sys.executable, "-c", ("import sys; import h2lm.tokenization.preparation; "
                                            "import h2lm.tokenization.benchmark; "
                                            "assert 'torch' not in sys.modules")], check=True, env=env)
