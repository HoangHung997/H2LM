"""Provenance for official digital text, restricted to tokenizer seed experiments."""
from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

ORIGIN = "official_pdf_text"
SCOPE = "tokenizer-seed-only"
HOSTS = {"congbao.chinhphu.vn", "g7.cdnchinhphu.vn"}


def official_url(url: str) -> str:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in HOSTS or parsed.username
            or parsed.password or parsed.port not in {None, 443} or parsed.fragment):
        raise ValueError("Only the reviewed official HTTPS hosts are allowed")
    return url


def sha256_value(value: Any) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("Expected SHA256")
    return value


def validate_official_source(source: dict[str, Any]) -> None:
    """Never label extraction by a parser as human_verified/teacher_verified."""
    proof = source.get("verification")
    if (not isinstance(proof, dict) or proof.get("scope") != SCOPE
            or proof.get("method") != "pypdf-digital-text-v1"
            or proof.get("human_verified") is not False
            or source.get("sensitivity") != "public"):
        raise ValueError("Official extraction requires explicit SEED-only mechanical provenance")
    official_url(source["source_uri"])
    official_url(proof["pdf_url"])
    sha256_value(proof.get("pdf_sha256"))
    sha256_value(proof.get("text_sha256"))
    sha256_value(proof.get("pages_sha256"))

