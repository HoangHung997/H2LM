"""Free-running inference on one field crop. No target, filename semantics or grammar repair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from h2lm.experiments.crop2_data import image_tensor
from h2lm.experiments.crop2_model import CropVLM
from h2lm.experiments.crop2_train import CropConfig, predict, read_checkpoint, source_hash


def main():
    parser = argparse.ArgumentParser(description='H2LM EXP-02 field crop, not whole-PDF reasoning')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--question', required=True, choices=['Số?', 'Ngày?'])
    args = parser.parse_args()
    state = read_checkpoint(args.run)
    if state['contract']['source_sha256'] != source_hash():
        raise ValueError('Source does not match checkpoint')
    cfg = CropConfig(**state['contract']['config'])
    torch.set_num_threads(cfg.threads)
    model = CropVLM(cfg.d_model, cfg.height, cfg.width)
    model.load_state_dict(state['model'])
    output = predict(model, image_tensor(args.image, cfg.height, cfg.width)[None], args.question)
    print(json.dumps({'production_ready': False, 'prediction': output[0]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
