"""Bounded, resumable tokenizer comparisons. The sealed test split is not opened."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import sentencepiece as spm
import yaml

from .corpus import Corpus, digest, load_corpus
from .preparation import read_bounded, read_json, write_json
from .tokenizer import H2Tokenizer, TokenizerConfig, evaluate, source_identity


def load_plan(path: str | Path) -> dict[str, Any]:
    data = read_bounded(Path(path), 65536)
    plan = yaml.safe_load(data)
    if not isinstance(plan, dict):
        raise TypeError("Comparison config must be a mapping")
    allowed = {"schema_version", "mode", "timeout_seconds", "candidates", "minimums"}
    if set(plan) - allowed or type(plan.get("schema_version")) is not int or plan["schema_version"] != 1:
        raise ValueError("Unknown comparison config key/schema")
    if plan.get("mode") not in {"fixture", "pilot"}:
        raise ValueError("Comparison mode must be fixture or pilot")
    seconds = plan.get("timeout_seconds", 120)
    if type(seconds) is not int or not 1 <= seconds <= 900:
        raise ValueError("Each candidate budget must be 1..900 seconds")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 8:
        raise ValueError("Plan must have 1..8 candidates")
    names = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise TypeError("Candidate must be a mapping")
        config = TokenizerConfig(**candidate)
        if (not config.name.isascii() or len(config.name) > 64
                or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in config.name)
                or config.name in names):
            raise ValueError("Candidate names must be distinct lowercase ASCII path-safe identifiers")
        if config.vocab_size > 65536 or config.max_corpus_bytes > 64 * 1024 * 1024:
            raise ValueError("Candidate exceeds pilot resource budget")
        if config.max_records > 100000 or config.max_sentence_bytes > 32768:
            raise ValueError("Candidate exceeds record/sentence budget")
        names.add(config.name)
    minimums = plan.get("minimums", {})
    valid = {"train_records", "validation_records", "train_families", "validation_families",
             "train_characters", "validation_characters"}
    if not isinstance(minimums, dict) or set(minimums) - valid:
        raise ValueError("Invalid corpus minimums")
    if any(type(v) is not int or v < 1 for v in minimums.values()):
        raise ValueError("Corpus minimums must be positive integers")
    return {"schema_version": 1, "mode": plan["mode"], "timeout_seconds": seconds,
            "candidates": [asdict(TokenizerConfig(**c)) for c in candidates], "minimums": minimums}


def development_corpus(manifest: Path) -> Corpus:
    raw = read_json(manifest)
    sources = raw.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Development manifest must list sources")
    # Must check BEFORE load_corpus; otherwise even an unused test shard would be read.
    if any(not isinstance(s, dict) or s.get("split") not in {"train", "validation"}
           for s in sources):
        raise ValueError("Comparison refuses test sources; use development_manifest.json only")
    corpus = load_corpus(manifest)
    if not corpus.split("train") or not corpus.split("validation"):
        raise ValueError("Both training and validation splits are required")
    return corpus


def corpus_gate(corpus: Corpus, plan: dict[str, Any]) -> dict[str, Any]:
    counts = {}
    for split in ("train", "validation"):
        rows = corpus.split(split)
        counts[f"{split}_records"] = len(rows)
        counts[f"{split}_families"] = len({r.family_id for r in rows})
        counts[f"{split}_characters"] = sum(len(r.text) for r in rows)
    failures = [f"{key}: {counts[key]} < {minimum}" for key, minimum in plan["minimums"].items()
                if counts[key] < minimum]
    if plan["mode"] == "pilot" and any(s["origin"] == "synthetic_fixture"
                                          for s in corpus.manifest["sources"]):
        failures.append("Pilot mode cannot use synthetic_fixture sources")
    return {"passed": not failures, "counts": counts, "failures": failures,
            "note": "Minimums are project gates, not a certificate of corpus representativeness."}


def run_worker(config: Path, manifest: Path, output: Path, seconds: int) -> str:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    command = [sys.executable, "-m", "h2lm.tokenization.cli", "train", "--config", str(config),
               "--manifest", str(manifest), "--output", str(output)]
    try:
        result = subprocess.run(command, check=False, capture_output=True,
                                env=env, timeout=seconds)
    except subprocess.TimeoutExpired:
        return "timeout"
    if result.returncode != 0:
        # Local log, not uploaded by our production path. May contain sensitive metadata.
        output.parent.joinpath(output.name + ".error.log").write_bytes(result.stderr[:65536])
        return "training_failed"
    return "trained"


def comparison_row(tokenizer: H2Tokenizer, corpus: Corpus, config: dict[str, Any]) -> dict[str, Any]:
    if tokenizer.metadata["config"] != config:
        raise ValueError("Existing candidate config mismatch; do not reuse this run")
    if tokenizer.metadata["manifest_sha256"] != corpus.manifest_sha256:
        raise ValueError("Existing candidate trained on a different manifest")
    started = time.monotonic()
    report = evaluate(tokenizer, corpus, "validation")
    elapsed = time.monotonic() - started
    groups: dict[str, dict[str, Any]] = {}
    for row in report["cases"]:
        group = groups.setdefault(row["category"], {"records": 0, "characters": 0, "tokens": 0})
        for key in ("characters", "tokens"):
            group[key] += row[key]
        group["records"] += 1
    for group in groups.values():
        group["characters_per_token"] = group["characters"] / group["tokens"]
    passed = report["exact_roundtrip_passed"] == report["records"] and report["unknown_tokens"] == 0
    return {
        "name": config["name"], "model_type": config["model_type"], "status": "evaluated",
        "mechanics_passed": passed, "requested_vocab_size": config["vocab_size"],
        "actual_vocab_size": tokenizer.vocab_size, "model_sha256": tokenizer.metadata["model_sha256"],
        "validation_tokens": report["total_tokens"],
        "validation_records": report["records"], "exact_roundtrip_passed": report["exact_roundtrip_passed"],
        "unknown_tokens": report["unknown_tokens"], "characters_per_token": report["characters_per_token"],
        "bytes_per_token": report["bytes_per_token"], "categories": groups,
        "validation_seconds_informational_only": elapsed,
    }


def compare_tokenizers(config: str | Path, manifest: str | Path, output: str | Path,
                       *, resume: bool = False) -> dict[str, Any]:
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    plan = load_plan(config)
    corpus = development_corpus(manifest)
    gate = corpus_gate(corpus, plan)
    if not gate["passed"]:
        raise ValueError("Corpus gate failed: " + "; ".join(gate["failures"]))
    identity = source_identity()
    fingerprint = {
        "plan_sha256": digest(json.dumps(plan, sort_keys=True).encode("utf-8")),
        "manifest_sha256": corpus.manifest_sha256,
        "source_sha256": identity["tokenizer_source_sha256"],
        "sentencepiece_version": spm.__version__, "python_version": sys.version,
    }
    if output.exists():
        if not resume:
            raise FileExistsError("Comparison output exists; use --resume or a new path")
        if read_json(output / "run.json")["fingerprint"] != fingerprint:
            raise ValueError("Resume rejected: data, configuration or implementation changed")
    else:
        if resume:
            raise ValueError("Cannot resume a missing comparison")
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "run.json", {"schema_version": 1, "fingerprint": fingerprint,
                                        "plan": plan, **identity})
    lock = output / "RUNNING.lock"
    try:
        with lock.open("x", encoding="ascii") as stream:
            stream.write(f"pid={os.getpid()}\n")
    except FileExistsError as exc:
        raise RuntimeError("Run is locked; confirm worker stopped before manually removing stale lock") from exc
    results = []
    try:
        for candidate in plan["candidates"]:
            name = candidate["name"]
            folder = output / name
            candidate_config = output / (name + ".json")  # JSON is also valid YAML.
            if not candidate_config.exists():
                write_json(candidate_config, candidate)
            elif read_json(candidate_config) != candidate:
                raise ValueError("Saved candidate config changed")
            if folder.exists():
                # A partial/corrupt artifact is an error, never silently overwritten.
                tokenizer = H2Tokenizer.load(folder)
                reused = True
            else:
                state = run_worker(candidate_config, manifest, folder, plan["timeout_seconds"])
                if state != "trained":
                    results.append({"name": name, "status": state, "mechanics_passed": False})
                    continue
                tokenizer, reused = H2Tokenizer.load(folder), False
            row = comparison_row(tokenizer, corpus, candidate)
            row["reused_checkpoint"] = reused
            results.append(row)
        eligible = [r for r in results if r["mechanics_passed"]]
        all_passed = len(eligible) == len(plan["candidates"])
        choice = min(eligible, key=lambda r: (r["validation_tokens"], r["actual_vocab_size"], r["name"])) \
            if all_passed else None
        result = {
            "schema_version": 1, "kind": "tokenizer-comparison-not-ocr-or-legal-reasoning",
            "mode": plan["mode"], "fingerprint": fingerprint, "corpus_gate": gate,
            "all_candidates_passed": all_passed, "candidates": results,
            "provisional_validation_choice": choice["name"] if choice else None,
            "selection_rule": "Fewest validation tokens; ties: smaller actual vocabulary then name.",
            "test_evaluated": False, "production_ready": False,
            "note": "No model promotion. Real-corpus review, locked holdout and deployment tests remain.",
            **identity,
        }
        # Append-only attempt reports; no ambiguous overwrite after a resumed run.
        attempt = 1
        while (output / f"comparison-{attempt:04d}.json").exists():
            attempt += 1
        write_json(output / f"comparison-{attempt:04d}.json", result)
        return result
    finally:
        lock.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare tokenizer candidates locally; no GPU/API")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        result = compare_tokenizers(args.config, args.manifest, args.output, resume=args.resume)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result["all_candidates_passed"] else 1
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(f"H2LM compare error: {exc}".encode("ascii", "backslashreplace").decode(), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
