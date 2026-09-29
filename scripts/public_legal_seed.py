"""Explicit public download + tokenizer run. No private files or teacher API are used."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from h2lm.tokenization.public_seed import run_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    name = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8]
    output = args.output or root / "artifacts/public-legal-seed" / name
    run_seed(root / "data/public_legal_seed/catalog.json",
             root / "configs/tokenizer/h2lm_comparison_official_seed.yaml",
             output, allow_network=args.allow_network)


if __name__ == "__main__":
    main()
