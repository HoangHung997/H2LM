"""Train our own vocabulary, not a downloaded model's vocabulary or weights."""
from __future__ import annotations

import io
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm
import yaml

from .corpus import Corpus, digest, load_corpus

# Stable IDs form a versioned contract. Raw text never inserts control IDs.
SPECIALS = ("<unk>", "<s>", "</s>", "<pad>", "<image>", "<page>",
            "<bbox>", "<answer>", "<evidence>", "<abstain>")
ESCAPE = "\ue000"
CODEC_VERSION = "h2-exact-text-v1"


def protect(text: str) -> str:
    """Escape SentencePiece's literal metaspace without losing source characters."""
    text.encode("utf-8", errors="strict")
    return text.replace(ESCAPE, ESCAPE + "0").replace("\u2581", ESCAPE + "1")


def restore(text: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == ESCAPE:
            index += 1
            if index == len(text) or text[index] not in "01":
                raise ValueError("Incomplete/invalid H2 text escape in decoded output")
            char = ESCAPE if text[index] == "0" else "\u2581"
        result.append(char)
        index += 1
    return "".join(result)


@dataclass(frozen=True)
class TokenizerConfig:
    name: str = "h2lm-tokenizer-dev-v1"
    vocab_size: int = 1024
    model_type: str = "bpe"
    split_digits: bool = True
    max_sentence_bytes: int = 4096
    max_corpus_bytes: int = 64 * 1024 * 1024
    max_records: int = 100_000

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Tokenizer name is required")
        for key in ("vocab_size", "max_sentence_bytes", "max_corpus_bytes", "max_records"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.vocab_size < 320:
            raise ValueError("vocab_size must be >=320 for byte fallback and special IDs")
        if self.model_type not in {"bpe", "unigram"}:
            raise ValueError("model_type must be bpe or unigram")
        if type(self.split_digits) is not bool:
            raise ValueError("split_digits must be boolean")

    @classmethod
    def load(cls, path: str | Path) -> TokenizerConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Tokenizer config must be a mapping")
        try:
            return cls(**raw)
        except TypeError as exc:
            raise ValueError("Unknown tokenizer config key or invalid type") from exc


def source_identity() -> dict[str, Any]:
    source_dir = Path(__file__).parent
    content = b"".join(path.name.encode() + b"\0" + path.read_bytes()
                       for path in sorted(source_dir.glob("*.py")))
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_dir, stderr=subprocess.DEVNULL,
            text=True, timeout=5,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=source_dir, stderr=subprocess.DEVNULL,
            text=True, timeout=5,
        ).strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    return {"code_revision": revision, "working_tree_dirty": dirty,
            "tokenizer_source_sha256": digest(content)}


class H2Tokenizer:
    """Exact text codec. Load the whole artifact directory, not .model alone."""

    def __init__(self, model: bytes, metadata: dict[str, Any]) -> None:
        if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
            raise ValueError("Unsupported tokenizer metadata schema")
        if metadata.get("codec_version") != CODEC_VERSION:
            raise ValueError("Unsupported H2 tokenizer codec version")
        if digest(model) != metadata.get("model_sha256"):
            raise ValueError("Tokenizer model checksum mismatch")
        fingerprints = metadata.get("training_fingerprints")
        if not isinstance(fingerprints, dict) or set(fingerprints) != {"text", "document", "family"}:
            raise ValueError("Missing training fingerprints")
        for values in fingerprints.values():
            if not isinstance(values, list) or not values or any(
                not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value) for value in values
            ):
                raise ValueError("Invalid training fingerprints")
        self.processor = spm.SentencePieceProcessor(model_proto=model)
        self.metadata = metadata
        if self.vocab_size != metadata.get("actual_vocab_size"):
            raise ValueError("Tokenizer vocabulary metadata mismatch")
        for index, piece in enumerate(SPECIALS):
            if self.processor.id_to_piece(index) != piece:
                raise ValueError("Incompatible special token IDs")

    @classmethod
    def load(cls, folder: str | Path) -> H2Tokenizer:
        folder = Path(folder)
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        tokenizer = cls((folder / "tokenizer.model").read_bytes(), metadata)
        if digest((folder / "source_manifest.json").read_bytes()) != metadata.get("manifest_sha256"):
            raise ValueError("Artifact source manifest checksum mismatch")
        return tokenizer

    @property
    def vocab_size(self) -> int:
        return self.processor.get_piece_size()

    def special_id(self, token: str) -> int:
        if token not in SPECIALS:
            raise ValueError("Unknown special token")
        return SPECIALS.index(token)

    def require_model_vocab(self, size: int) -> None:
        if size != self.vocab_size:
            raise ValueError(f"Model vocab_size must be {self.vocab_size}, received {size}")

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        ids = self.processor.encode(protect(text), out_type=int)
        if any(index < len(SPECIALS) for index in ids):
            raise ValueError("Unexpected control/unknown token in ordinary text")
        return ([1] if bos else []) + ids + ([2] if eos else [])

    def decode(self, ids: list[int]) -> str:
        if any(type(index) is not int or not 0 <= index < self.vocab_size for index in ids):
            raise ValueError("Token ID out of range")
        if 0 in ids:
            raise ValueError("Cannot faithfully decode an unknown token")
        return restore(self.processor.decode(ids))


def train_tokenizer(
    config: TokenizerConfig, manifest: str | Path, output: str | Path,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise FileExistsError("Output already exists; choose a new run directory")
    corpus = load_corpus(manifest, max_corpus_bytes=config.max_corpus_bytes,
                         max_records=config.max_records)
    training = corpus.split("train")
    if not training:
        raise ValueError("A training split is required")
    sentences = [protect(record.text) for record in training]
    if any(len(text.encode("utf-8")) > config.max_sentence_bytes for text in sentences):
        raise ValueError("Training record too long; split it explicitly without changing family_id")
    writer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(sentences), model_writer=writer,
        model_type=config.model_type, vocab_size=config.vocab_size,
        character_coverage=1.0, byte_fallback=True,
        normalization_rule_name="identity", add_dummy_prefix=False,
        remove_extra_whitespaces=False, split_digits=config.split_digits,
        allow_whitespace_only_pieces=True, hard_vocab_limit=False,
        max_sentence_length=config.max_sentence_bytes,
        num_threads=1, shuffle_input_sentence=False, input_sentence_size=0,
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        control_symbols=list(SPECIALS[4:]), minloglevel=2,
    )
    model = writer.getvalue()
    processor = spm.SentencePieceProcessor(model_proto=model)
    metadata = {
        "schema_version": 1, "codec_version": CODEC_VERSION,
        "name": config.name, "stage": "development-not-production",
        "config": asdict(config), "model_sha256": digest(model),
        "actual_vocab_size": processor.get_piece_size(),
        "special_tokens": dict(zip(SPECIALS, range(len(SPECIALS)))),
        "manifest_sha256": corpus.manifest_sha256,
        "training_records": len(training), "duplicates_removed": corpus.duplicates_removed,
        "training_fingerprints": {
            kind: sorted({record.fingerprints()[kind] for record in training})
            for kind in ("text", "document", "family")
        },
        "sentencepiece_version": spm.__version__, "python_version": platform.python_version(),
        **source_identity(),
    }
    tokenizer = H2Tokenizer(model, metadata)
    if any(tokenizer.decode(tokenizer.encode(row.text)) != row.text for row in training):
        raise ValueError("Training-text round trip failed; artifacts were not published")
    # Atomic create guard: never reuse another run, even after concurrent training.
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output / "tokenizer.model").write_bytes(model)
        (output / "tokenizer.vocab").write_text(
            "\n".join(f"{i}\t{processor.id_to_piece(i)}" for i in range(tokenizer.vocab_size))
            + "\n", encoding="utf-8",
        )
        manifest_bytes = Path(manifest).read_bytes()
        if digest(manifest_bytes) != corpus.manifest_sha256:
            raise ValueError("Manifest changed during training")
        (output / "source_manifest.json").write_bytes(manifest_bytes)
        # Metadata is the completion marker, written last.
        (output / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    except OSError:
        # Incomplete run stays for diagnosis; loader will reject missing metadata.
        raise
    return metadata


def evaluate(tokenizer: H2Tokenizer, corpus: Corpus, split: str = "validation") -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("Evaluate only validation or test, not the training split")
    records = corpus.split(split)
    if not records:
        raise ValueError("Evaluation split is empty")
    training = tokenizer.metadata["training_fingerprints"]
    known = {kind: set(values) for kind, values in training.items()}
    results = []
    for record in records:
        for kind, value in record.fingerprints().items():
            if value in known[kind]:
                raise ValueError(f"Evaluation leaks original training data: {kind}")
        ids = tokenizer.encode(record.text)
        results.append({
            "id": record.id, "category": record.category, "tokens": len(ids),
            "characters": len(record.text), "utf8_bytes": len(record.text.encode("utf-8")),
            "exact_roundtrip": tokenizer.decode(ids) == record.text,
            "unknown_tokens": ids.count(0),
        })
    total_tokens = sum(row["tokens"] for row in results)
    total_bytes = sum(row["utf8_bytes"] for row in results)
    total_chars = sum(row["characters"] for row in results)
    return {
        "schema_version": 1, "kind": "tokenizer-mechanics-not-legal-reasoning",
        "split": split, "records": len(results),
        "model_sha256": tokenizer.metadata["model_sha256"],
        "training_manifest_sha256": tokenizer.metadata["manifest_sha256"],
        "evaluation_manifest_sha256": corpus.manifest_sha256,
        "exact_roundtrip_passed": sum(row["exact_roundtrip"] for row in results),
        "unknown_tokens": sum(row["unknown_tokens"] for row in results),
        "total_tokens": total_tokens, "characters_per_token": total_chars / total_tokens,
        "bytes_per_token": total_bytes / total_tokens, "cases": results,
        **source_identity(),
    }
