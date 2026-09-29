"""EXP-02 bounded joint image/language learning. CTC is auxiliary, never an OCR text input."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F

from .crop2_data import load, prepare
from .crop2_model import CropVLM
from .data import decode, dump, encode, sha
from .engine import batch, distance


@dataclass(frozen=True)
class CropConfig:
    steps: int = 1800
    batch_size: int = 16
    learning_rate: float = 0.001
    ctc_weight: float = 0.3
    seed: int = 29
    data_seed: int = 208
    threads: int = 2
    seconds: int = 480
    save_every: int = 300
    height: int = 48
    width: int = 320
    d_model: int = 96
    train_images: int = 1200
    validation_images: int = 96

    def __post_init__(self):
        bounds = {'steps': (1, 3000), 'batch_size': (1, 24), 'seed': (0, 100000),
                  'data_seed': (0, 100000), 'threads': (1, 4), 'seconds': (1, 600),
                  'save_every': (1, 1000), 'height': (32, 64), 'width': (128, 384),
                  'd_model': (32, 128), 'train_images': (8, 4096), 'validation_images': (8, 256)}
        for key, (lo, hi) in bounds.items():
            value = getattr(self, key)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f'Invalid bounded configuration: {key}')
        for key, lo, hi in [('learning_rate', 0.00001, 0.01), ('ctc_weight', 0, 1)]:
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'Invalid configuration: {key}')
        if self.width % 4 or self.height % 16 or self.d_model % 4:
            raise ValueError('Width/hidden size must divide by 4, height by 16')

    @classmethod
    def read(cls, path: Path):
        raw = yaml.safe_load(path.read_text(encoding='utf-8'))
        if not isinstance(raw, dict):
            raise TypeError('Config must be a mapping')
        if set(raw) - {f.name for f in fields(cls)}:
            raise ValueError('Unknown configuration key')
        return cls(**raw)


def source_hash() -> str:
    import hashlib
    root = Path(__file__).parent
    paths = sorted(root.glob('crop2_*.py')) + [root / 'data.py', root / 'engine.py',
            root.parent / 'config.py', root.parent / 'modeling/h2lm.py']
    return hashlib.sha256(b''.join(p.name.encode() + p.read_bytes() for p in paths)).hexdigest()


def code_revision() -> dict:
    try:
        root = Path(__file__).parent
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                           stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root,
                                            stderr=subprocess.DEVNULL, timeout=5).strip())
        return {'revision': revision, 'working_tree_dirty': dirty}
    except (OSError, subprocess.SubprocessError):
        return {'revision': None, 'working_tree_dirty': None}


def model_hash(model: torch.nn.Module) -> str:
    import hashlib
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(bytes(tensor.detach().cpu().contiguous().clone().untyped_storage()))
    return h.hexdigest()


def checkpoint(out: Path, state: dict) -> Path:
    path = out / f"step-{state['step']:06d}.pt"
    if path.exists() or path.with_suffix('.tmp').exists():
        raise FileExistsError('Checkpoint already exists; immutable step files are not overwritten')
    torch.save(state, path.with_suffix('.tmp'))
    os.replace(path.with_suffix('.tmp'), path)
    dump(out / 'latest.tmp', {'file': path.name, 'sha256': sha(path)})
    os.replace(out / 'latest.tmp', out / 'latest.json')
    return path


def read_checkpoint(out: Path) -> dict:
    ref = json.loads((out / 'latest.json').read_text(encoding='utf-8'))
    path = (out / ref['file']).resolve()
    if not path.is_relative_to(out.resolve()) or path.stat().st_size > 100_000_000:
        raise ValueError('Checkpoint path/size outside bounds')
    if sha(path) != ref['sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('architecture') != 'single-line-cross-attention-v2':
        raise ValueError('Wrong checkpoint architecture')
    return state


@torch.no_grad()
def predict(model: CropVLM, pixels: torch.Tensor, prompt: str, max_new: int = 32) -> list[dict]:
    """Only image tensor and question enter prediction. Answers, IDs and filenames do not."""
    prefix = [1, *encode(prompt), 7]
    if len(prefix) > 24 or type(max_new) is not int or not 1 <= max_new <= 32:
        raise ValueError('Prompt or generation limit is too long')
    if pixels.shape[0] > 32:
        raise ValueError('Prediction batch exceeds limit')
    model.eval()
    memory = model.memory(pixels)
    ids = torch.tensor([prefix] * pixels.shape[0], dtype=torch.long, device=pixels.device)
    generated = [[] for _ in range(pixels.shape[0])]
    finished = [False] * pixels.shape[0]
    for _ in range(max_new):
        logits = model.decode(ids, memory)[:, -1].clone()
        logits[:, :2] = -float('inf')
        logits[:, 3:10] = -float('inf')
        selected = logits.argmax(-1)
        for i, v in enumerate(selected.tolist()):
            if not finished[i]:
                if v == 2:
                    finished[i] = True
                else:
                    generated[i].append(v)
        ids = torch.cat([ids, selected[:, None]], dim=1)
        if all(finished):
            break
    results = []
    for values, ended in zip(generated, finished):
        try:
            text, valid = decode(values), True
        except UnicodeDecodeError:
            text = bytes(v - 10 for v in values).decode('utf-8', errors='replace')
            valid = False
        results.append({'text': text, 'valid_utf8': valid, 'eos': ended})
    return results


def matches(prediction: dict, target: str) -> bool:
    return prediction['valid_utf8'] and prediction['eos'] and prediction['text'] == target


def evaluate(model: CropVLM, rows: list[dict]) -> dict:
    """Greedy free generation, blank and wrong-image controls, no per-answer corrections."""
    examples = []
    for prompt in sorted({r['prompt'] for r in rows}):
        group = [r for r in rows if r['prompt'] == prompt]
        if len(group) < 2:
            raise ValueError('Need two different answers per prompt for counterfactual evaluation')
        alternatives = []
        for i, row in enumerate(group):
            ordered = group[i+1:] + group[:i]
            alternate = next((r for r in ordered if r['target'] != row['target']
                              and r['sha256'] != row['sha256']), None)
            if alternate is None:
                raise ValueError('Wrong-image control has no distinct target/image')
            alternatives.append(alternate)
        for start in range(0, len(group), 16):
            selected = group[start:start+16]
            swapped = alternatives[start:start+16]
            pixels = torch.stack([r['tensor'] for r in selected])
            normal = predict(model, pixels, prompt)
            blank = predict(model, torch.ones_like(pixels), prompt)
            wrong = predict(model, torch.stack([r['tensor'] for r in swapped]), prompt)
            for row, alt, p, b, w in zip(selected, swapped, normal, blank, wrong):
                examples.append({'id': row['id'], 'target': row['target'], 'prompt': prompt,
                                 'image_sha256': row['sha256'], 'prediction': p,
                                 'blank': b, 'wrong_image': w, 'wrong_image_id': alt['id'],
                                 'wrong_image_target': alt['target'],
                                 'exact': matches(p, row['target']),
                                 'blank_exact': matches(b, row['target']),
                                 'wrong_image_retains_original': matches(w, row['target']),
                                 'wrong_image_matches_replacement': matches(w, alt['target']),
                                 'category': row.get('category', 'real_silver'),
                                 'degraded': row.get('degraded'),
                                 'font_index': row.get('font_index')})
    def metrics(items):
        return {'count': len(items), 'exact': sum(r['exact'] for r in items),
                'blank_exact': sum(r['blank_exact'] for r in items),
                'wrong_image_retains_original': sum(r['wrong_image_retains_original'] for r in items),
                'wrong_image_matches_replacement': sum(r['wrong_image_matches_replacement'] for r in items),
                'answer_char_error_rate': sum(distance(r['target'], r['prediction']['text']) for r in items)
                / sum(len(r['target']) for r in items)}
    return {**metrics(examples), 'by_category': {k: metrics([r for r in examples if r['category'] == k])
            for k in sorted({r['category'] for r in examples})}, 'examples': examples,
            'independent_benchmark': False, 'scope': 'short_field_answers_not_full_page_OCR'}


@torch.no_grad()
def nll(model: CropVLM, rows: list[dict]) -> float:
    model.eval()
    total, tokens = 0.0, 0
    for start in range(0, len(rows), 16):
        ids, pixels, labels = batch(rows[start:start+16])
        logits, _ = model(ids[:, :-1], pixels)
        targets = labels[:, 1:]
        total += float(F.cross_entropy(logits.flatten(0,1), targets.flatten(),
                                       ignore_index=-100, reduction='sum'))
        tokens += int(targets.ne(-100).sum())
    return total/tokens


def train(manifest: Path, out: Path, cfg: CropConfig, *, resume: bool = False,
          segment_steps: int | None = None) -> dict:
    if segment_steps is not None and (type(segment_steps) is not int or not 1 <= segment_steps <= cfg.steps):
        raise ValueError('Invalid segment length')
    torch.set_num_threads(cfg.threads)
    rows = load(manifest, cfg.height, cfg.width, ('train',))
    contract = {'config': asdict(cfg), 'dataset_sha256': sha(manifest),
                'source_sha256': source_hash(), 'torch': str(torch.__version__),
                'python_major_minor': f'{sys.version_info.major}.{sys.version_info.minor}'}
    out.mkdir(parents=True, exist_ok=resume)
    lock = out / 'RUNNING.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    try:
        torch.manual_seed(cfg.seed)
        model = CropVLM(cfg.d_model, cfg.height, cfg.width)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        sampler = torch.Generator().manual_seed(cfg.seed)
        step, last_saved = 0, -1
        initial_hash = model_hash(model)
        initial_vision = model.vision[0].weight.detach().clone()
        initial_nll = nll(model, rows[:64])
        if resume:
            saved = read_checkpoint(out)
            if saved['contract'] != contract:
                raise ValueError('Resume config/data/source/library mismatch')
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            sampler.set_state(saved['sampler_rng'])
            torch.set_rng_state(saved['torch_rng'])
            step = last_saved = saved['step']
            initial_hash, initial_nll = saved['initial_hash'], saved['initial_nll']
            initial_vision = saved['initial_vision']
            if step >= cfg.steps:
                raise ValueError('Requested training is already complete; evaluate instead')
        else:
            dump(out/'contract.json', {**contract, 'git': code_revision()})
        stop = min(cfg.steps, step + (segment_steps or cfg.steps))
        started = time.monotonic()
        interrupted = False
        def save():
            return checkpoint(out, {'architecture': 'single-line-cross-attention-v2',
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                'contract': contract, 'sampler_rng': sampler.get_state(), 'torch_rng': torch.get_rng_state(),
                'initial_hash': initial_hash, 'initial_nll': initial_nll, 'initial_vision': initial_vision})
        try:
            while step < stop and time.monotonic()-started < cfg.seconds:
                chosen = torch.randint(len(rows), (cfg.batch_size,), generator=sampler)
                selected = [rows[int(i)] for i in chosen]
                ids, pixels, labels = batch(selected)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits, ctc = model(ids[:, :-1], pixels)
                ce = F.cross_entropy(logits.flatten(0,1), labels[:, 1:].flatten(), ignore_index=-100)
                aux = F.ctc_loss(ctc.log_softmax(-1).transpose(0,1),
                    torch.tensor([v for r in selected for v in r['answer_ids']]),
                    torch.full((len(selected),), ctc.shape[1]),
                    torch.tensor([len(r['answer_ids']) for r in selected]),
                    blank=0, zero_infinity=False)
                loss = ce + cfg.ctc_weight*aux
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite objective; checkpoint not advanced')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                if step % cfg.save_every == 0:
                    save()
                    last_saved = step
                    event = {'step': step, 'answer_loss': float(ce.detach()),
                             'ctc_loss': float(aux.detach()), 'gradient_norm': float(norm)}
                    print(json.dumps(event), flush=True)
                    with (out/'training.jsonl').open('a', encoding='utf-8') as log:
                        log.write(json.dumps(event)+'\n')
        except KeyboardInterrupt:
            interrupted = True
        if step != last_saved:
            save()
        summary = {'architecture': 'single-line-cross-attention-v2', 'step': step,
                   'requested_steps': cfg.steps, 'completed_steps': step == cfg.steps,
                   'interrupted': interrupted, 'training_seconds': time.monotonic()-started,
                   'parameters': sum(p.numel() for p in model.parameters()),
                   'initial_train64_nll': initial_nll, 'final_train64_nll': nll(model, rows[:64]),
                   'weights_changed': initial_hash != model_hash(model),
                   'vision_weight_max_change': float((model.vision[0].weight-initial_vision).abs().max().detach()),
                   'contract': contract, 'git': code_revision(), 'training_counts': {'synthetic': len(rows), 'real': 0},
                   'production_ready': False, 'independent_benchmark': False}
        if step == cfg.steps:
            validation = load(manifest, cfg.height, cfg.width, ('validation',))
            summary['evaluation'] = evaluate(model, validation)
        loaded = CropVLM(cfg.d_model, cfg.height, cfg.width)
        loaded.load_state_dict(read_checkpoint(out)['model'])
        first = rows[0]
        summary['reload_matches'] = (predict(model, first['tensor'][None], first['prompt']) ==
                                     predict(loaded, first['tensor'][None], first['prompt']))
        summary['checkpoint'] = json.loads((out/'latest.json').read_text())
        dump(out/f'report-{step:06d}.json', summary)
        return summary
    finally:
        lock.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description='EXP-02 neural field learning, not a general PDF model')
    parser.add_argument('--config', type=Path, default=Path('configs/training/crop_v2.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--segment-steps', type=int)
    args = parser.parse_args()
    cfg = CropConfig.read(args.config)
    manifest = args.output/'dataset/manifest.json'
    if not args.resume:
        if args.output.exists():
            parser.error('Output exists; choose a new directory or --resume')
        prepare(manifest.parent, cfg.train_images, cfg.validation_images, cfg.data_seed)
    report = train(manifest, args.output/'run', cfg, resume=args.resume, segment_steps=args.segment_steps)
    print(json.dumps({k:v for k,v in report.items() if k not in ('evaluation', 'contract')}, indent=2))
    if 'evaluation' in report:
        print('VALIDATION', json.dumps({k:v for k,v in report['evaluation'].items()
                                      if k not in ('examples','by_category')}))
    return 0 if report['completed_steps'] or args.segment_steps else 1


if __name__ == '__main__':
    raise SystemExit(main())
