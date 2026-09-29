from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .data import prepare
from .engine import train


def main() -> None:
    parser = argparse.ArgumentParser(description="H2LM bounded neural micro-fit, not production training")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/training/micro_vlm.yaml"))
    parser.add_argument("--real-review", type=Path)
    parser.add_argument("--allow-silver", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--segment-steps", type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.allow_silver:
        cfg["allow_silver"] = True
    manifest = args.output / "dataset/manifest.json"
    if not args.resume:
        if args.output.exists():
            parser.error("Output exists; use a new path or explicit --resume")
        prepare(manifest.parent, args.real_review, cfg["allow_silver"])
    summary = train(manifest, args.output / "run", cfg, args.resume, args.segment_steps)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("evaluation", "contract")}, indent=2))
    print("Evaluation counts:", {k: (v["exact"], v["count"], v["blank_exact"])
                                  for k, v in summary["evaluation"].items()})


if __name__ == "__main__":
    main()
