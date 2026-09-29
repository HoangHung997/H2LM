"""Local, checksummed corpus contract. No downloads and no teacher API calls."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPLITS = {"train", "validation", "test"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_digest(text: str) -> str:
    # Canonical-equivalent Unicode counts as leakage, even in exact-text mode.
    return digest(unicodedata.normalize("NFC", text).encode("utf-8"))


def required_string(obj: dict[str, Any], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Required non-empty string: {key}")
    return value


@dataclass(frozen=True)
class Record:
    id: str
    document_id: str
    family_id: str
    text: str
    split: str
    source_id: str
    category: str

    def fingerprints(self) -> dict[str, str]:
        return {
            "text": text_digest(self.text),
            "document": digest(self.document_id.encode("utf-8")),
            "family": digest(self.family_id.encode("utf-8")),
        }


@dataclass
class Corpus:
    records: list[Record]
    manifest: dict[str, Any]
    manifest_sha256: str
    duplicates_removed: int

    def split(self, name: str) -> list[Record]:
        if name not in SPLITS:
            raise ValueError("Unknown split")
        return [record for record in self.records if record.split == name]


def load_corpus(
    manifest_path: str | Path,
    *,
    max_corpus_bytes: int = 64 * 1024 * 1024,
    max_records: int = 100_000,
    max_record_bytes: int = 262_144,
) -> Corpus:
    """Bounded prototype loader; fails on leakage instead of fixing splits silently.

    Paths must be inside the manifest directory, including symlink resolution.
    Checksums cover raw bytes, before Unicode or line-ending processing.
    """
    manifest_path = Path(manifest_path).resolve()
    if manifest_path.stat().st_size > 1024 * 1024:
        raise ValueError("Manifest exceeds 1 MiB")
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest.decode("utf-8"))
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest.get("schema_version") != 1):
        raise ValueError("Expected corpus schema_version=1")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Manifest must contain sources")

    records: list[Record] = []
    ids: set[str] = set()
    source_ids: set[str] = set()
    source_paths: set[Path] = set()
    seen: dict[tuple[str, str], str] = {}
    total_bytes = total_records = duplicates = 0
    root = manifest_path.parent
    for source in sources:
        if not isinstance(source, dict):
            raise TypeError("Source must be an object")
        source_id = required_string(source, "id")
        if source_id in source_ids:
            raise ValueError("Duplicate source id")
        source_ids.add(source_id)
        split = source.get("split")
        if not isinstance(split, str) or split not in SPLITS:
            raise ValueError("Source split must be train, validation, or test")
        required_string(source, "source_uri")
        required_string(source, "rights_note")
        if source.get("approved") is not True:
            raise ValueError("Source must be explicitly approved")
        origin = source.get("origin")
        if not isinstance(origin, str) or origin not in {
                "synthetic_fixture", "human_verified", "teacher_verified"}:
            raise ValueError("Unsupported source origin")
        if origin == "teacher_verified":
            teacher = source.get("teacher")
            if not isinstance(teacher, dict) or teacher.get("review_status") != "verified":
                raise ValueError("Teacher output needs verified provenance")
            for key in ("provider", "model", "prompt_version", "run_id", "source_sha256"):
                required_string(teacher, key)
            value = teacher["source_sha256"]
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Invalid teacher source_sha256")

        relative = Path(required_string(source, "path"))
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("Source path must stay inside the manifest directory")
        if path.suffix.lower() != ".jsonl" or path in source_paths:
            raise ValueError("Expected distinct .jsonl source paths")
        source_paths.add(path)
        size = path.stat().st_size
        total_bytes += size
        if total_bytes > max_corpus_bytes:
            raise ValueError("Corpus byte budget exceeded")
        data = path.read_bytes()
        if len(data) != size or digest(data) != required_string(source, "sha256"):
            raise ValueError(f"Checksum mismatch for source {source_id}")
        source_record_count = 0
        for number, line in enumerate(data.splitlines(), 1):
            if not line.strip():
                continue
            total_records += 1
            source_record_count += 1
            if total_records > max_records or len(line) > max_record_bytes:
                raise ValueError("Corpus record budget exceeded")
            try:
                row = json.loads(line.decode("utf-8"))
                if not isinstance(row, dict):
                    raise TypeError("Record must be an object")
                fields = {key: required_string(row, key) for key in
                          ("id", "document_id", "family_id", "text", "category")}
                fields["text"].encode("utf-8", errors="strict")
            except (ValueError, TypeError, UnicodeError) as exc:
                raise ValueError(f"Invalid record at {source_id}:{number}") from exc
            if fields["id"] in ids:
                raise ValueError("Duplicate record id")
            ids.add(fields["id"])
            record = Record(**fields, split=split, source_id=source_id)
            fingerprints = record.fingerprints()
            duplicate_text = ("text", fingerprints["text"]) in seen
            for kind, value in fingerprints.items():
                previous = seen.get((kind, value))
                if previous is not None and previous != split:
                    raise ValueError(f"Cross-split leakage: {kind}")
                seen[(kind, value)] = split
            if duplicate_text:
                duplicates += 1
            else:
                records.append(record)
        if source_record_count == 0:
            raise ValueError("Empty source shard")
    if not records:
        raise ValueError("Corpus is empty")
    return Corpus(records, manifest, digest(raw_manifest), duplicates)
