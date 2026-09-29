"""python -m h2lm.tokenization.cli --help"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corpus import load_corpus
from .tokenizer import H2Tokenizer, TokenizerConfig, evaluate, train_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="H2LM CPU tokenizer tools; no model weights/API")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train a new vocabulary from approved local data")
    train.add_argument("--config", required=True)
    train.add_argument("--manifest", required=True)
    train.add_argument("--output", required=True)
    benchmark = commands.add_parser("evaluate", help="Evaluate held-out text reconstruction")
    benchmark.add_argument("--tokenizer", required=True)
    benchmark.add_argument("--manifest", required=True)
    benchmark.add_argument("--split", choices=["validation", "test"], default="validation")
    benchmark.add_argument("--report", required=True)
    inspect = commands.add_parser("inspect", help="Inspect local UTF-8 text and token IDs")
    inspect.add_argument("--tokenizer", required=True)
    inspect.add_argument("--text-file", required=True)
    args = parser.parse_args()
    try:
        if args.command == "train":
            metadata = train_tokenizer(TokenizerConfig.load(args.config), args.manifest, args.output)
            result = {key: metadata[key] for key in
                      ("stage", "actual_vocab_size", "training_records", "model_sha256")}
        elif args.command == "evaluate":
            report = Path(args.report)
            if report.exists():
                raise FileExistsError("Report exists; choose a new path")
            tokenizer = H2Tokenizer.load(args.tokenizer)
            result = evaluate(tokenizer, load_corpus(args.manifest), args.split)
            report.parent.mkdir(parents=True, exist_ok=True)
            with report.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            if result["exact_roundtrip_passed"] != result["records"] or result["unknown_tokens"]:
                print(json.dumps(result, ensure_ascii=True, indent=2))
                return 1
        else:
            tokenizer = H2Tokenizer.load(args.tokenizer)
            # Reading bytes avoids Python universal-newline conversion of CRLF.
            text = Path(args.text_file).read_bytes().decode("utf-8")
            ids = tokenizer.encode(text)
            result = {"tokens": len(ids), "ids": ids,
                      "exact_roundtrip": tokenizer.decode(ids) == text}
        # ASCII-safe console output also works with legacy Windows code pages.
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        message = f"H2LM tokenizer error: {exc}".encode("ascii", errors="backslashreplace").decode()
        print(message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
