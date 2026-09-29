"""Train a tiny, synthetic tokenizer and check both held-out fixture splits."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from h2lm.tokenization.corpus import load_corpus
from h2lm.tokenization.tokenizer import H2Tokenizer, TokenizerConfig, evaluate, train_tokenizer


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="H2LM M1 CPU demo; not a trained AI")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = args.output or root / "artifacts" / "tokenizer" / run_id
    manifest = root / "data/tokenizer_sample/manifest.json"
    metadata = train_tokenizer(
        TokenizerConfig.load(root / "configs/tokenizer/h2lm_tokenizer_dev.yaml"), manifest, output,
    )
    tokenizer = H2Tokenizer.load(output)
    for split in ("validation", "test"):
        report = evaluate(tokenizer, load_corpus(manifest), split)
        (output / f"{split}_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        if report["exact_roundtrip_passed"] != report["records"] or report["unknown_tokens"]:
            raise RuntimeError(f"{split} round-trip gate failed")
        print(f"{split}: {report['records']}/{report['records']} exact; "
              f"unknown={report['unknown_tokens']}; tokens={report['total_tokens']}")
    print(f"vocab={metadata['actual_vocab_size']}; train_records={metadata['training_records']}")
    print("Tokenizer fixture only. No PDF understanding or legal reasoning score.")
    print("Artifacts: " + str(output).encode("ascii", errors="backslashreplace").decode())


if __name__ == "__main__":
    main()
