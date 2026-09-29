"""Optional bounded silver-label adaptation, never an independent real-scan benchmark."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .crop2_data import load
from .crop2_model import CropVLM
from .crop2_probe import load_probe
from .crop2_train import CropConfig, checkpoint, evaluate, read_checkpoint, source_hash
from .data import dump, encode, sha
from .engine import batch

FIT_IDS = {'dt-number', 'dt-date', 'tb-number', 'tb-date'}
PROBE_IDS = {'da-number', 'da-date'}


def adapt(parent: Path, manifest: Path, review: Path, out: Path, *, allow_silver: bool = False,
          steps: int = 300) -> dict:
    if allow_silver is not True or type(steps) is not int or not 1 <= steps <= 500:
        raise ValueError('Explicit experimental silver-label opt-in and bounded steps required')
    state = read_checkpoint(parent)
    if state['contract']['source_sha256'] != source_hash() or 'adaptation' in state['contract']:
        raise ValueError('Use the exact synthetic-only EXP-02 parent checkpoint')
    if sha(manifest) != state['contract']['dataset_sha256']:
        raise ValueError('Parent synthetic dataset mismatch')
    cfg = CropConfig(**state['contract']['config'])
    if state['step'] != cfg.steps:
        raise ValueError('Parent training must finish before adaptation')
    synthetic = load(manifest, cfg.height, cfg.width, ('train',))
    real = load_probe(review, cfg)
    fit = [r for r in real if r['id'] in FIT_IDS]
    if len(fit) != 4 or {r['id'] for r in real if r['id'] not in FIT_IDS} != PROBE_IDS:
        raise ValueError('Fit/probe region partition mismatch')
    for row in fit:
        prefix = [1, *encode(row['prompt']), 7]
        target = [*encode(row['target']), 2]
        row['ids'], row['labels'] = prefix+target, [-100]*len(prefix)+target
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(71)
    model = CropVLM(cfg.d_model, cfg.height, cfg.width)
    model.load_state_dict(state['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    sampler = torch.Generator().manual_seed(71)
    started = time.monotonic()
    try:
        for step in range(1, steps+1):
            if time.monotonic()-started > 120:
                raise TimeoutError('Bounded adaptation exceeded its time allowance')
            rows = [synthetic[int(i)] for i in torch.randint(len(synthetic), (12,), generator=sampler)] + fit
            ids, pixels, labels = batch(rows)
            pixels[-4:] = (pixels[-4:] * float(torch.rand((), generator=sampler)*0.1+0.9)).clamp(-1,1)
            optimizer.zero_grad(set_to_none=True)
            model.train()
            logits, ctc = model(ids[:, :-1], pixels)
            ce = F.cross_entropy(logits.flatten(0,1), labels[:, 1:].flatten(), ignore_index=-100)
            aux = F.ctc_loss(ctc[:12].log_softmax(-1).transpose(0,1),
                torch.tensor([v for r in rows[:12] for v in r['answer_ids']]),
                torch.full((12,), ctc.shape[1]), torch.tensor([len(r['answer_ids']) for r in rows[:12]]),
                blank=0, zero_infinity=False)
            loss = ce+0.3*aux
            if not torch.isfinite(loss):
                raise ValueError('Non-finite adaptation loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
        parent_ref = json.loads((parent/'latest.json').read_text())
        provenance = {'steps': steps, 'learning_rate': 0.0001, 'seed': 71,
                      'fit_region_ids': sorted(FIT_IDS), 'probe_region_ids': sorted(PROBE_IDS),
                      'parent_checkpoint_sha256': parent_ref['sha256'],
                      'review_sha256': sha(review/'draft_review.json'), 'source_sha256': sha(Path(__file__)),
                      'label_kind': 'unverified_AI_silver', 'production_approved': False}
        contract = {**state['contract'], 'adaptation': provenance}
        checkpoint(out, {'architecture': state['architecture'], 'step': state['step']+steps,
                        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'contract': contract, 'sampler_rng': sampler.get_state(),
                        'torch_rng': torch.get_rng_state()})
        result = {'stage': 'silver_adaptation', 'provenance': provenance,
                  'training_seconds': time.monotonic()-started, 'production_ready': False,
                  'independent_benchmark': False, 'adaptation_resume_supported': False,
                  'real': evaluate(model, real),
                  'synthetic_validation': evaluate(model, load(manifest, cfg.height, cfg.width, ('validation',)))}
        for name, selected in [('fit', FIT_IDS), ('probe', PROBE_IDS)]:
            examples = [r for r in result['real']['examples'] if r['id'] in selected]
            result[name] = {'count': len(examples), 'exact': sum(r['exact'] for r in examples)}
        dump(out/'adaptation_report.json', result)
        return result
    except (Exception, KeyboardInterrupt):
        dump(out/'FAILED.json', {'complete': False, 'production_ready': False})
        raise


def main():
    parser = argparse.ArgumentParser(description='Optional silver experiment; not gold-label approval')
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-silver', action='store_true')
    args = parser.parse_args()
    report = adapt(args.parent, args.manifest, args.review, args.output, allow_silver=args.allow_silver)
    print(json.dumps({'fit': report['fit'], 'probe': report['probe'],
                      'production_ready': False}, indent=2))


if __name__ == '__main__':
    main()
