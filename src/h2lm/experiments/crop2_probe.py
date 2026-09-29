"""Real development probe only: six previously inspected AI drafts, no fitting or promotion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .crop2_data import image_tensor
from .crop2_model import CropVLM
from .crop2_train import CropConfig, evaluate, read_checkpoint, source_hash
from .data import dump, sha

EXPECTED_SOURCES = {
    'da700': '48187f3d4d13869e51c0dc97bfd53c75c1618a4ed8ba76a79a678e6451ba20b6',
    'dt391': '3e9997051dbc2cfaefbcefbaef09956127271cadd772da400c4dacf8007f3bda',
    'thach-bich': 'd451efdbbf6b91b990f7d64d46a72b0a1ef3a58747dcfb73ece91ac6454429cc',
}
SELECTED = {f'{prefix}-{field}' for prefix in ('da','dt','tb') for field in ('number','date')}


def load_probe(review: Path, cfg: CropConfig) -> list[dict]:
    manifest = review / 'draft_review.json'
    if manifest.stat().st_size > 2_000_000 or (review/'FAILED.json').exists():
        raise ValueError('Failed/oversized review bundle')
    data = json.loads(manifest.read_text(encoding='utf-8'))
    permission = data.get('permission', {})
    if (permission.get('public_sharing_allowed_by_user') is not True
            or set(permission.get('applies_to_document_ids', [])) != set(EXPECTED_SOURCES)):
        raise ValueError('Real probe permission scope mismatch')
    rows, seen = [], set()
    for region in data['regions']:
        if region['id'] not in SELECTED:
            continue
        if region['id'] in seen:
            raise ValueError('Duplicate selected real region')
        seen.add(region['id'])
        if (region.get('status') != 'readable' or region.get('review_status') != 'assistant_visual_draft'
                or region.get('training_eligible') is not False
                or EXPECTED_SOURCES.get(region['document_id']) != region['source_sha256']):
            raise ValueError('Real probe source/provenance mismatch')
        image = (review / region['crop_path']).resolve()
        if (not image.is_relative_to(review.resolve()) or image.stat().st_size > 8_000_000
                or sha(image) != region['crop_sha256']):
            raise ValueError('Real crop path/hash mismatch')
        rows.append({'id': region['id'], 'sha256': region['crop_sha256'],
                     'source_sha256': region['source_sha256'], 'target': region['target'],
                     'prompt': 'Số?' if region['id'].endswith('number') else 'Ngày?',
                     'tensor': image_tensor(image, cfg.height, cfg.width),
                     'category': 'real_number' if region['id'].endswith('number') else 'real_date'})
    if seen != SELECTED:
        raise ValueError('Incomplete six-region probe')
    return rows


def run_probe(run: Path, review: Path, report: Path) -> dict:
    if report.exists():
        raise FileExistsError('Probe report already exists; no silent overwrite')
    state = read_checkpoint(run)
    if state['contract']['source_sha256'] != source_hash():
        raise ValueError('Use exact source matching the checkpoint')
    cfg = CropConfig(**state['contract']['config'])
    torch.set_num_threads(cfg.threads)
    model = CropVLM(cfg.d_model, cfg.height, cfg.width)
    model.load_state_dict(state['model'])
    rows = load_probe(review, cfg)
    result = {'scope': 'previously_inspected_real_development_probe',
              'checkpoint_sha256': sha(run/json.loads((run/'latest.json').read_text())['file']),
              'training_real_samples': len(state['contract'].get('adaptation', {}).get('fit_region_ids', [])),
              'fit_region_ids': state['contract'].get('adaptation', {}).get('fit_region_ids', []),
              'checkpoint_steps': state['step'], 'label_kind': 'unverified_AI_draft_reference',
              'independent_benchmark': False, 'production_ready': False,
              'evaluation': evaluate(model, rows)}
    dump(report, result)
    return result


def main():
    parser = argparse.ArgumentParser(description='EXP-02 real-scan development probe, not a benchmark')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(args.run, args.review, args.report)
    print(json.dumps({k:v for k,v in result['evaluation'].items() if k not in ('examples','by_category')}, indent=2))


if __name__ == '__main__':
    main()
