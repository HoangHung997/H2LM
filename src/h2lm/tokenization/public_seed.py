"""Acquire a bounded, explicitly selected public Gazette seed. No crawling, OCR or API teacher."""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.robotparser import RobotFileParser

from .benchmark import compare_tokenizers
from .corpus import digest
from .preparation import prepare_corpus, read_json, write_json
from .seed_audit import duplicate_audit, evaluate_selected_holdout, freeze_corpus
from .seed_provenance import ORIGIN, SCOPE, official_url
from .tokenizer import source_identity

AGENT = "H2LM-Research-Seed/0.1"
MAX_PDF = 16 * 1024 * 1024


class CheckedRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        official_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url: str, limit: int) -> bytes:
    official_url(url)
    request = Request(url, headers={"User-Agent": AGENT, "Accept-Encoding": "identity"})
    # No auth, cookies, user secrets or TLS bypass. Every redirect is allowlisted.
    with build_opener(CheckedRedirect()).open(request, timeout=30) as response:
        official_url(response.url)
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise ValueError("Compressed HTTP response refused")
        value = response.read(limit + 1)
    if len(value) > limit:
        raise ValueError("Download budget exceeded")
    return value


def check_robots(url: str, cache: dict[str, RobotFileParser]) -> None:
    parts = urlsplit(official_url(url))
    base = f"{parts.scheme}://{parts.netloc}"
    if base not in cache:
        parser = RobotFileParser(base + "/robots.txt")
        try:
            data = download(base + "/robots.txt", 256 * 1024)
            parser.parse(data.decode("utf-8", errors="strict").splitlines())
        except HTTPError as exc:
            if exc.code not in {404, 410}:
                raise
            parser.parse([])
        cache[base] = parser
    parser = cache[base]
    if not parser.can_fetch(AGENT, url):
        raise ValueError("Acquisition blocked by robots.txt")
    delay = parser.crawl_delay(AGENT) or 1
    if delay > 30:
        raise ValueError("Site crawl delay exceeds seed acquisition budget")
    time.sleep(delay)


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs.extend(value for key, value in attrs if key == "href" and value)


def identifier(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFD", text.casefold())
                  .replace("đ", "d"))


def discover_pdf(html: bytes, landing: str, code: str) -> str:
    parser = Links()
    parser.feed(html.decode("utf-8", errors="strict"))
    found = set()
    for link in parser.hrefs:
        link = urljoin(landing, link)
        path = unquote(urlsplit(link).path)
        if path.lower().endswith(".pdf") and identifier(code) in identifier(path.split("/")[-1]):
            # Source sometimes publishes http hrefs; request the identical resource via HTTPS only.
            if link.startswith("http://"):
                link = "https://" + link[7:]
            found.add(official_url(link))
    if len(found) != 1:
        raise ValueError("Need exactly one complete document PDF; no guessed links or partial volumes")
    return found.pop()


def inspect_pages(pages: list[str], code: str) -> dict:
    if not 1 <= len(pages) <= 200:
        raise ValueError("Seed supports 1..200 pages per selected document")
    text = "\n\n".join(pages)
    if len(text) > 4_000_000 or len(text) < 2000:
        raise ValueError("Extracted document outside seed character budget")
    if identifier(code) not in identifier("\n".join(pages[:2])):
        raise ValueError("Document identity anchor missing on first two pages")
    bad = [i + 1 for i, page in enumerate(pages) if len(page.strip()) < 40]
    if bad:
        raise ValueError(f"Low-text pages {bad}; quarantine instead of silently skipping/OCR")
    if any(c == "\ufffd" or unicodedata.category(c) in {"Co", "Cs"}
           or (unicodedata.category(c) == "Cc" and c not in "\n\r\t") for c in text):
        raise ValueError("Unsupported/corrupt extraction characters; needs review")
    if len(re.findall(r"(?i)điều\s+\d+", text)) < 2:
        raise ValueError("Expected Vietnamese article anchors missing")
    offsets, cursor = [], 0
    for i, page in enumerate(pages):
        offsets.append({"page": i + 1, "char_start": cursor, "char_end": cursor + len(page),
                        "text_sha256": digest(page.encode("utf-8"))})
        cursor += len(page) + 2
    return {"pages": len(pages), "characters": len(text), "page_offsets": offsets,
            "mechanical_gate_passed": True, "visual_transcription_verified": False,
            "ocr_performed": False, "layout_ground_truth": False,
            "current_legal_validity_checked": False}


def extract_pdf(pdf: Path, code: str, output: Path) -> dict:
    import pypdf

    # Limits reduce accidental resource exhaustion. Not a PDF-security sandbox.
    if sys.platform == "linux":
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    raw = pdf.read_bytes()
    if len(raw) > MAX_PDF or not raw.startswith(b"%PDF-"):
        raise ValueError("Expected bounded PDF bytes")
    reader = pypdf.PdfReader(io.BytesIO(raw), strict=True)
    if reader.is_encrypted or not 1 <= len(reader.pages) <= 200:
        raise ValueError("Encrypted or oversized document refused")
    pages = []
    for page in reader.pages:
        contents = page.get_contents()
        if contents is not None and len(contents.get_data()) > 16 * 1024 * 1024:
            raise ValueError("Oversized decompressed page stream")
        pages.append(page.extract_text() or "")
    report = inspect_pages(pages, code)
    report["parser_version"] = pypdf.__version__
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "pages.json", {"pages": pages})
    (output / "document.txt").write_bytes("\n\n".join(pages).encode("utf-8"))
    write_json(output / "extraction.json", report)
    return report


def collect_seed(catalog: Path, output: Path, *, allow_network: bool = False) -> dict:
    if not allow_network:
        raise ValueError("Network disabled: explicitly pass --allow-network for reviewed public URLs")
    specs = read_json(catalog)
    entries = specs.get("documents")
    if (specs.get("schema_version") != 1 or specs.get("scope") != SCOPE
            or not isinstance(entries, list) or not 3 <= len(entries) <= 32):
        raise ValueError("Invalid bounded seed catalog")
    ids, families = set(), {}
    for entry in entries:
        key = entry["id"]
        if (not re.fullmatch(r"[a-z0-9-]{1,60}", key) or key in ids
                or entry.get("acquisition_approved") is not True
                or entry.get("split") not in {"train", "validation", "test"}
                or entry.get("version_scope") != "as-published-not-consolidated"):
            raise ValueError("Catalog requires distinct IDs, explicit source approval and version scope")
        ids.add(key)
        official_url(entry["landing_url"])
        if urlsplit(entry["landing_url"]).hostname != "congbao.chinhphu.vn":
            raise ValueError("Landing must be the official Gazette")
        family = entry["family_id"]
        if family in families and families[family] != entry["split"]:
            raise ValueError("Cross-split source family")
        families[family] = entry["split"]
    if {e["split"] for e in entries} != {"train", "validation", "test"}:
        raise ValueError("Catalog needs all splits")
    output.mkdir(parents=True, exist_ok=False)
    (output / "catalog.json").write_bytes(catalog.read_bytes())
    robots: dict[str, RobotFileParser] = {}
    docs, audits, inventory = [], [], []
    total = 0
    for entry in entries:
        folder = output / entry["id"]
        folder.mkdir()
        landing = entry["landing_url"]
        check_robots(landing, robots)
        html = download(landing, 4 * 1024 * 1024)
        (folder / "landing.html").write_bytes(html)
        url = discover_pdf(html, landing, entry["code"])
        check_robots(url, robots)
        raw = download(url, MAX_PDF)
        total += len(raw)
        if total > 80 * 1024 * 1024:
            raise ValueError("Total seed download budget exceeded")
        pdf = folder / "source.pdf"
        pdf.write_bytes(raw)
        extracted = folder / "extracted"
        command = [sys.executable, "-m", "h2lm.tokenization.public_seed", "extract",
                   "--pdf", str(pdf), "--code", entry["code"], "--output", str(extracted)]
        # Each parser runs out-of-process; do not continue after an unverified/missing document.
        result = subprocess.run(command, capture_output=True, check=False, timeout=180)
        (folder / "extract.log").write_bytes(result.stderr[:65536])
        if result.returncode:
            raise ValueError(f"Extraction failed for {entry['id']}; see local extract.log")
        extraction = read_json(extracted / "extraction.json")
        pages = read_json(extracted / "pages.json", limit=16 * 1024 * 1024)["pages"]
        text = (extracted / "document.txt").read_bytes()
        verification = {"scope": SCOPE, "method": "pypdf-digital-text-v1", "human_verified": False,
                        "pdf_url": url, "pdf_sha256": digest(raw), "text_sha256": digest(text),
                        "pages_sha256": digest((extracted / "pages.json").read_bytes())}
        docs.append({"id": entry["id"], "family_id": entry["family_id"], "split": entry["split"],
                     "path": f"{entry['id']}/extracted/document.txt", "sha256": digest(text),
                     "category": entry["category"], "source_uri": landing,
                     "rights_note": specs["rights_note"], "approved": True, "sensitivity": "public",
                     "origin": ORIGIN, "reviewed_by": "automated-mechanical-gate-NOT-human",
                     "reviewed_at": datetime.now(timezone.utc).isoformat(), "verification": verification})
        audits.append({**entry, "pages": pages})
        inventory.append({**entry, **verification, **extraction,
                          "landing_sha256": digest(html), "pdf_bytes": len(raw)})
        print(f"SOURCE {entry['id']} pages={len(pages)} chars={len(text.decode('utf-8'))} url={url}", flush=True)
    report = {"scope": SCOPE, "catalog_sha256": digest(catalog.read_bytes()), "documents": inventory,
              "pdf_bytes": total, "production_ready": False, "neural_training_performed": False,
              **source_identity()}
    write_json(output / "acquisition_report.json", report)
    near = duplicate_audit(audits)
    write_json(output / "duplicate_audit.json", near)
    if not near["passed"]:
        raise ValueError("Near-duplicate cross-split pages/documents detected; review before training")
    write_json(output / "registry.json", {"schema_version": 1, "name": specs["name"],
                                         "chunk_bytes": 2048, "documents": docs})
    return report


def run_seed(catalog: Path, config: Path, output: Path, *, allow_network: bool) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    collected = collect_seed(catalog, output / "sources", allow_network=allow_network)
    prepared = output / "prepared"
    preparation = prepare_corpus(output / "sources/registry.json", prepared)
    frozen = output / "corpus-freeze.json"
    freeze_corpus(prepared, frozen)
    comparison = compare_tokenizers(config, prepared / "development_manifest.json", output / "comparison")
    if not comparison["all_candidates_passed"]:
        raise ValueError("Tokenizer candidate failed: inspect comparison report, do not lower gates")
    holdout = evaluate_selected_holdout(prepared, frozen, output / "comparison/comparison-0001.json",
                                        output / "comparison", output / "final-evaluation")
    summary = {"kind": "official-public-pdf-tokenizer-SEED-not-trained-neural-model",
               "documents": len(collected["documents"]),
               "pages": sum(d["pages"] for d in collected["documents"]),
               "splits": preparation["splits"], "candidates": comparison["candidates"],
               "choice_from_validation": comparison["provisional_validation_choice"],
               "holdout_records": holdout["records"], "holdout_exact": holdout["exact_roundtrip_passed"],
               "holdout_unknown": holdout["unknown_tokens"], "holdout_tokens": holdout["total_tokens"],
               "freeze_sha256": digest(frozen.read_bytes()), "production_ready": False,
               "visual_transcription_verified": False, "legal_validity_verified": False,
               **source_identity()}
    write_json(output / "summary.json", summary)
    print("H2LM_PUBLIC_SEED_SUMMARY " + json.dumps(summary, ensure_ascii=True), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--pdf", type=Path, required=True)
    extract.add_argument("--code", required=True)
    extract.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--catalog", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "extract":
            extract_pdf(args.pdf, args.code, args.output)
        else:
            run_seed(args.catalog, args.config, args.output, allow_network=args.allow_network)
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"H2LM public seed error: {exc}".encode("ascii", "backslashreplace").decode(), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
