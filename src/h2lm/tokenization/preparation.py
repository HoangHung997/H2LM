"""Import reviewed local UTF-8 documents. Never download or approve sources implicitly."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from .corpus import SPLITS, digest, load_corpus, required_string, text_digest
from .tokenizer import protect


def read_bounded(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("JSON file exceeds size limit")
    return raw


def read_json(path: Path, limit: int = 1024 * 1024) -> dict[str, Any]:
    obj = json.loads(read_bounded(path, limit).decode("utf-8"))
    if not isinstance(obj, dict):
        raise TypeError("JSON root must be an object")
    return obj


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def chunk_text(text: str, max_bytes: int) -> list[tuple[int, int, str]]:
    """Lossless code-point slices; budget includes the H2 metaspace escape codec.

    Offsets are Python Unicode character offsets, NOT PDF bboxes or byte offsets.
    Excessively long whitespace-only spans fail explicitly instead of being removed.
    """
    if type(max_bytes) is not int or not 64 <= max_bytes <= 32768:
        raise ValueError("chunk_bytes must be an integer from 64 to 32768")
    if not text.strip():
        raise ValueError("Document has no non-whitespace text")
    chunks: list[tuple[int, int, str]] = []
    start = size = 0
    for index, char in enumerate(text):
        width = len(protect(char).encode("utf-8"))
        if size + width > max_bytes:
            chunks.append((start, index, text[start:index]))
            start, size = index, 0
        size += width
    chunks.append((start, len(text), text[start:]))
    # Preserve trailing whitespace without publishing an empty training example.
    if len(chunks) > 1 and not chunks[-1][2].strip():
        a, _, previous = chunks[-2]
        _, end, tail = chunks[-1]
        # Move a short non-whitespace suffix into the final record, if it fits.
        cut = len(previous.rstrip()) - 1
        if cut <= 0 or len(protect(previous[cut:] + tail).encode("utf-8")) > max_bytes:
            raise ValueError("Whitespace-only chunk; review document or increase chunk_bytes")
        chunks[-2:] = [(a, a + cut, previous[:cut]),
                       (a + cut, end, previous[cut:] + tail)]
    if any(not part.strip() for _, _, part in chunks):
        raise ValueError("Whitespace-only chunk; review document or increase chunk_bytes")
    if "".join(part for _, _, part in chunks) != text:
        raise RuntimeError("Lossless chunk reconstruction failed")
    return chunks


def prepare_corpus(registry: str | Path, output: str | Path) -> dict[str, Any]:
    """Build development + sealed-holdout manifests from an explicitly reviewed registry.

    This is a bounded pilot importer, not a web crawler or billion-token streaming loader.
    Registry approval fields are declarations by the reviewer, not automatic legal clearance.
    """
    registry, output = Path(registry).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Output exists; use a new directory")
    registry_bytes = read_bounded(registry, 1024 * 1024)
    registry_hash = digest(registry_bytes)
    raw = json.loads(registry_bytes.decode("utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Registry must be an object")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError("Expected registry schema_version=1")
    required_string(raw, "name")
    chunk_bytes = raw.get("chunk_bytes", 2048)
    chunk_text("validation", chunk_bytes)
    docs = raw.get("documents")
    if not isinstance(docs, list) or not 3 <= len(docs) <= 1000:
        raise ValueError("Registry needs 3..1000 documents including all three splits")
    root = registry.parent
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    assignments: dict[tuple[str, str], str] = {}
    rows_by_doc, source_entries = [], []
    total_bytes = total_rows = 0
    inventory = []
    for index, doc in enumerate(docs):
        if not isinstance(doc, dict):
            raise TypeError("Document must be an object")
        document_id = required_string(doc, "id")
        family_id = required_string(doc, "family_id")
        if document_id in seen_ids:
            raise ValueError("Duplicate document id")
        seen_ids.add(document_id)
        split = doc.get("split")
        if not isinstance(split, str) or split not in SPLITS:
            raise ValueError("Each document needs an explicit train/validation/test split")
        if doc.get("approved") is not True:
            raise ValueError("Document is not approved; review it before import")
        if doc.get("sensitivity") not in {"public", "private"}:
            raise ValueError("Document must declare public/private sensitivity")
        for key in ("source_uri", "rights_note", "reviewed_by", "reviewed_at", "category"):
            required_string(doc, key)
        if doc.get("origin") not in {"synthetic_fixture", "human_verified", "teacher_verified"}:
            raise ValueError("Unsupported reviewed document origin")
        relative = Path(required_string(doc, "path"))
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root) or path.suffix.lower() != ".txt":
            raise ValueError("Document path must be a local .txt inside registry directory")
        if path in seen_paths:
            raise ValueError("Duplicate source path")
        seen_paths.add(path)
        with path.open("rb") as stream:
            data = stream.read(8 * 1024 * 1024 + 1)
        total_bytes += len(data)
        if len(data) > 8 * 1024 * 1024 or total_bytes > 32 * 1024 * 1024:
            raise ValueError("Document/corpus pilot byte budget exceeded")
        if digest(data) != required_string(doc, "sha256"):
            raise ValueError("Raw document checksum mismatch")
        text = data.decode("utf-8", errors="strict")
        if "\x00" in text or "\ufffd" in text:
            raise ValueError("NUL/replacement character detected; review extraction, do not repair silently")
        for key in (("family", family_id), ("document_text", text_digest(text))):
            previous = assignments.get(key)
            if previous is not None and previous != split:
                raise ValueError(f"Cross-split leakage: {key[0]}")
            assignments[key] = split
        chunks = chunk_text(text, chunk_bytes)
        total_rows += len(chunks)
        if total_rows > 100000:
            raise ValueError("Prepared record budget exceeded")
        source_id = f"source-{index:04d}"
        shard = f"{split}/{source_id}.jsonl"
        rows = [dict(id=f"{source_id}-{n:05d}", document_id=document_id,
                     family_id=family_id, category=doc["category"], text=part,
                     source_char_start=start, source_char_end=end,
                     raw_document_sha256=doc["sha256"])
                for n, (start, end, part) in enumerate(chunks)]
        encoded = ("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)).encode("utf-8")
        entry = {key: doc[key] for key in ("source_uri", "rights_note", "approved", "origin",
                                         "sensitivity", "reviewed_by", "reviewed_at")}
        if "teacher" in doc:
            entry["teacher"] = doc["teacher"]
        entry.update(id=source_id, path=shard, split=split, sha256=digest(encoded),
                     raw_document_sha256=doc["sha256"])
        source_entries.append(entry)
        rows_by_doc.append((shard, encoded))
        inventory.append(dict(document_id=document_id, family_id=family_id, split=split,
                              characters=len(text), utf8_bytes=len(data), records=len(rows),
                              raw_sha256=digest(data), sensitivity=doc["sensitivity"]))
    if {entry["split"] for entry in source_entries} != SPLITS:
        raise ValueError("Registry must contain train, validation and sealed test documents")
    # Use an isolated staging directory, validate with the existing M1-A loader first.
    if digest(read_bounded(registry, 1024 * 1024)) != registry_hash:
        raise ValueError("Registry changed during preparation")
    with tempfile.TemporaryDirectory(prefix="h2lm-corpus-") as temporary:
        stage = Path(temporary)
        for shard, encoded in rows_by_doc:
            target = stage / shard
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(encoded)
        full = {"schema_version": 1, "sources": source_entries}
        write_json(stage / "all.json", full)
        corpus = load_corpus(stage / "all.json")
        summary = {
            "schema_version": 1, "kind": "prepared-corpus-not-model-training",
            "name": raw["name"], "registry_sha256": registry_hash,
            "documents": inventory, "chunk_bytes": chunk_bytes,
            "duplicates_removed_by_loader": corpus.duplicates_removed,
            "contains_private_data": any(d["sensitivity"] == "private" for d in docs),
            "splits": {s: {"records": len(corpus.split(s)),
                           "families": len({r.family_id for r in corpus.split(s)}),
                           "characters": sum(len(r.text) for r in corpus.split(s))}
                       for s in sorted(SPLITS)},
            "production_ready": False,
        }
        output.mkdir(parents=True, exist_ok=False)
        for shard, encoded in rows_by_doc:
            target = output / shard
            target.parent.mkdir(exist_ok=True)
            with target.open("xb") as stream:
                stream.write(encoded)
        write_json(output / "preparation_report.json", summary)
        write_json(output / "holdout_manifest.json", {
            "schema_version": 1, "sources": [e for e in source_entries if e["split"] == "test"]})
        # Publish last. Benchmark receives no test paths and never reads test text.
        write_json(output / "development_manifest.json", {
            "schema_version": 1, "sources": [e for e in source_entries if e["split"] != "test"]})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare reviewed UTF-8 corpus locally; no API")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        report = prepare_corpus(args.registry, args.output)
        print(json.dumps({"splits": report["splits"], "production_ready": False}, indent=2))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"H2LM prepare error: {exc}".encode("ascii", "backslashreplace").decode(), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
