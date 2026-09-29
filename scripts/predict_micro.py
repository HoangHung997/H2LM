"""Inspect an experimental checkpoint on one image crop; not a PDF/legal assistant."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from h2lm.experiments.data import image_tensor
from h2lm.experiments.engine import make_model, predict, read_checkpoint, source_hash, validate


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental image-to-answer neural inference")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    state = read_checkpoint(args.run)
    cfg = state["contract"]["config"]
    validate(cfg)
    if state["contract"]["source_sha256"] != source_hash():
        raise ValueError("Use the exact model source distributed with this checkpoint")
    if args.image.stat().st_size > 8_000_000 or len(args.question.encode("utf-8")) > 48:
        raise ValueError("Input is outside the micro-experiment limits")
    torch.set_num_threads(cfg["threads"])
    model = make_model(cfg)
    model.load_state_dict(state["model"])
    result = predict(model, image_tensor(args.image, cfg["image_size"]), args.question,
                     cfg["max_new_tokens"])
    print(json.dumps({"experimental_only": True, "production_ready": False,
                      "prediction": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
