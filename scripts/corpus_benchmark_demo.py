"""Fixture-only end-to-end import/comparison. No documents or APIs are downloaded."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from h2lm.tokenization.benchmark import compare_tokenizers
from h2lm.tokenization.corpus import digest
from h2lm.tokenization.preparation import prepare_corpus, write_json

ROOT = Path(__file__).resolve().parents[1]


def create_fixture_registry(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=False)
    documents = []
    for split in ("train", "validation", "test"):
        path = ROOT / "data" / "tokenizer_sample" / f"{split}.jsonl"
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            row = json.loads(line)
            data = row["text"].encode("utf-8")
            filename = f"{split}-{index:03d}.txt"
            (folder / filename).write_bytes(data)
            documents.append({
                "id": row["id"], "family_id": row["family_id"], "path": filename,
                "sha256": digest(data), "split": split, "category": row["category"],
                "origin": "synthetic_fixture", "source_uri": "h2lm://synthetic/tokenizer-fixture",
                "rights_note": "Project-authored synthetic tests; NOT real legal provisions.",
                "approved": True, "sensitivity": "public", "reviewed_by": "H2LM fixture tests",
                "reviewed_at": "fixture-v1",
            })
    registry = folder / "registry.json"
    write_json(registry, {"schema_version": 1, "name": "synthetic-import-smoke",
                          "chunk_bytes": 2048, "documents": documents})
    return registry


def main() -> int:
    parser = argparse.ArgumentParser(description="H2LM M1-B1 corpus/benchmark fixture")
    parser.add_argument("--output")
    args = parser.parse_args()
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = Path(args.output) if args.output else ROOT / "artifacts" / "corpus-benchmark" / tag
    folder.mkdir(parents=True, exist_ok=False)
    registry = create_fixture_registry(folder / "input")
    preparation = prepare_corpus(registry, folder / "prepared")
    report = compare_tokenizers(ROOT / "configs/tokenizer/h2lm_comparison_fixture.yaml",
                                folder / "prepared/development_manifest.json", folder / "comparison")
    print("Synthetic fixture only: NOT a real legal corpus or trained H2LM neural model.")
    print("Prepared splits:", preparation["splits"])
    for row in report["candidates"]:
        print(row["name"], row["status"], "actual_vocab=", row.get("actual_vocab_size"),
              "validation_tokens=", row.get("validation_tokens"))
    print("Sealed test read during comparison:", report["test_evaluated"])
    print("Production ready:", report["production_ready"])
    print("Output:", str(folder).encode("ascii", "backslashreplace").decode())
    return 0 if report["all_candidates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
