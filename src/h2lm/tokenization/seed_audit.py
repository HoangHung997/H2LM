"""Mechanical checks for an official-public-text SEED, not a legal ground-truth dataset."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from .corpus import digest
from .preparation import read_json, write_json
from .seed_provenance import SCOPE
from .tokenizer import H2Tokenizer, evaluate



def shingles(text: str, width: int = 5) -> set[str]:
    words = re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())
    return {" ".join(words[i:i + width]) for i in range(max(0, len(words) - width + 1))}


def duplicate_audit(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact pairwise 5-word shingles; bounded seed audit, not corpus-scale MinHash.

    Compare whole documents AND individual pages to catch a copied chapter/page.
    Shared legal boilerplate is not automatically a near-duplicate: thresholds and
    minimum overlap are fixed and reported. Text itself is never changed.
    """
    if not 2 <= len(documents) <= 32:
        raise ValueError("Seed audit supports 2..32 documents")
    units = []
    assignments = {}
    for doc in documents:
        family, split = doc["family_id"], doc["split"]
        if family in assignments and assignments[family] != split:
            raise ValueError("Cross-split family")
        assignments[family] = split
        pages = doc["pages"]
        units.append((doc["id"], split, "document", shingles("\n".join(pages))))
        units.extend((doc["id"], split, f"page-{i + 1}", shingles(page))
                     for i, page in enumerate(pages))
    if len(units) > 1500:
        raise ValueError("Seed page budget exceeded")
    flags = []
    comparisons = 0
    for i, (left_id, left_split, left_unit, left) in enumerate(units):
        for right_id, right_split, right_unit, right in units[i + 1:]:
            if left_split == right_split or (left_unit == "document") != (right_unit == "document"):
                continue
            if min(len(left), len(right)) < 150:
                continue
            comparisons += 1
            shared = len(left & right)
            if shared < 150:
                continue
            jaccard = shared / len(left | right)
            containment = shared / min(len(left), len(right))
            if jaccard >= 0.85 or containment >= 0.95:
                flags.append({"left": left_id, "right": right_id, "left_unit": left_unit,
                              "right_unit": right_unit, "jaccard": jaccard,
                              "containment": containment, "shared_shingles": shared})
    return {"method": "exact-5-word-shingles-v1", "jaccard_threshold": 0.85,
            "containment_threshold": 0.95, "minimum_shared_shingles": 150,
            "comparisons": comparisons, "flags": flags, "passed": not flags,
            "limitations": "Does not detect paraphrases or short copied clauses; not a legal relation graph."}


def freeze_corpus(prepared: Path, output: Path) -> dict[str, Any]:
    """Freeze both manifests and every shard BEFORE selecting a tokenizer."""
    manifest_hashes, shard_hashes = {}, {}
    for name in ("development_manifest.json", "holdout_manifest.json"):
        path = prepared / name
        manifest_hashes[name] = digest(path.read_bytes())
        for source in read_json(path)["sources"]:
            relative = Path(source["path"])
            target = (prepared / relative).resolve()
            if relative.is_absolute() or not target.is_relative_to(prepared.resolve()):
                raise ValueError("Invalid frozen shard path")
            actual = digest(target.read_bytes())
            if actual != source["sha256"]:
                raise ValueError("Frozen shard checksum mismatch")
            shard_hashes[source["path"]] = actual
    value = {"schema_version": 1, "manifests": manifest_hashes, "shards": shard_hashes,
             "scope": SCOPE, "tamper_proof": False}
    write_json(output, value)
    return value


def check_freeze(prepared: Path, frozen: Path) -> None:
    value = read_json(frozen)
    if value.get("schema_version") != 1 or value.get("scope") != SCOPE:
        raise ValueError("Invalid corpus freeze")
    for name, expected in {**value["manifests"], **value["shards"]}.items():
        path = (prepared / name).resolve()
        if not path.is_relative_to(prepared.resolve()) or digest(path.read_bytes()) != expected:
            raise ValueError("Frozen corpus has changed")


def evaluate_selected_holdout(prepared: Path, frozen: Path, comparison: Path,
                              candidates: Path, output: Path) -> dict[str, Any]:
    """Record selection first; consume one holdout evaluation per run directory.

    Procedural separation, NOT protection against an owner who copies/resets directories.
    Test results may not be used to revise the choice for this frozen seed.
    """
    from .corpus import load_corpus

    check_freeze(prepared, frozen)
    result = read_json(comparison)
    if not result.get("all_candidates_passed") or result.get("test_evaluated") is not False:
        raise ValueError("Need an accepted validation-only comparison")
    selected = result["provisional_validation_choice"]
    rows = [row for row in result["candidates"] if row["name"] == selected]
    if len(rows) != 1:
        raise ValueError("Invalid selected candidate")
    candidate_path = (candidates / selected).resolve()
    if not candidate_path.is_relative_to(candidates.resolve()):
        raise ValueError("Invalid candidate path")
    tokenizer = H2Tokenizer.load(candidate_path)
    if (tokenizer.metadata["model_sha256"] != rows[0]["model_sha256"]
            or tokenizer.metadata["manifest_sha256"]
            != digest((prepared / "development_manifest.json").read_bytes())):
        raise ValueError("Selected artifact does not match comparison/corpus")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "selection.json", {
        "name": selected, "model_sha256": rows[0]["model_sha256"],
        "comparison_sha256": digest(comparison.read_bytes()),
        "freeze_sha256": digest(frozen.read_bytes()), "decision_before_test": True,
        "scope": SCOPE, "production_ready": False,
    })
    # Completion failure does not remove selection: no quiet test retries in this run.
    report = evaluate(tokenizer, load_corpus(prepared / "holdout_manifest.json"), "test")
    report["selected_candidate"] = selected
    report["production_ready"] = False
    write_json(output / "holdout_report.json", report)
    return report
