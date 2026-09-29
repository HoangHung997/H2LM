"""CQ-01 finite full-scale curriculum experiment, NOT complete foundation pretraining."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F

from h2lm.scale.config import ScaleConfig, tiny_config
from h2lm.scale.fixture import decode, encode
from h2lm.scale.model import build, parameter_report
from h2lm.scale.optim import AuditedSGD
from h2lm.scale.runner import resource_gate
from h2lm.scale.storage import (
    digest_file,
    get_latest,
    index_at,
    load_into,
    save,
    set_latest,
    write_json,
)

from .data import TASKS, prepare, sample, validate


@dataclass(frozen=True)
class Plan:
    steps: int = 48
    seed: int = 404
    peak_lr: float = 0.0003
    floor_lr: float = 0.00003
    warmup: int = 6
    seconds: int = 900
    eval_pairs: int = 6
    max_new_tokens: int = 12
    threads: int = 2

    def __post_init__(self):
        for key, lo, hi in [('steps', 6, 128), ('seed', 0, 100000), ('warmup', 1, 16),
                           ('seconds', 1, 1200), ('eval_pairs', 1, 12),
                           ('max_new_tokens', 8, 32), ('threads', 1, 4)]:
            value = getattr(self, key)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f'Invalid bounded plan: {key}')
        for value in (self.peak_lr, self.floor_lr):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 0.003:
                raise ValueError('Invalid learning rate')
        if self.floor_lr > self.peak_lr or self.warmup >= self.steps:
            raise ValueError('Invalid schedule order')


def source_hash() -> str:
    root = Path(__file__).parent
    h = hashlib.sha256()
    for p in sorted(root.glob('*.py')):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def schedule(step: int, plan: Plan) -> float:
    if not 0 <= step < plan.steps:
        raise ValueError('Step outside plan')
    if step < plan.warmup:
        return plan.peak_lr*(step+1)/plan.warmup
    progress = (step-plan.warmup)/max(1, plan.steps-plan.warmup-1)
    return plan.floor_lr + (plan.peak_lr-plan.floor_lr)*(1+math.cos(math.pi*progress))/2


def select_training(data: dict, plan: Plan) -> list[dict]:
    """Deterministic task curriculum. Never uses validation/test target quality to sample."""
    groups = {task: [r for r in data['records'] if r['split'] == 'train' and r['task'] == task]
              for task in TASKS}
    if any(not rows for rows in groups.values()):
        raise ValueError('A curriculum task has no training examples')
    import random
    rng = random.Random(plan.seed)
    for rows in groups.values():
        rng.shuffle(rows)
    use = Counter()
    selected = []
    q = plan.steps//4
    for step in range(plan.steps):
        pool = ('read',) if step < q else (('read','locate','role') if step < 2*q else TASKS)
        task = pool[(step if step < q else step-q if step < 2*q else step-2*q) % len(pool)]
        selected.append(groups[task][use[task] % len(groups[task])])
        use[task] += 1
    return selected


def weighted_loss(logits: torch.Tensor, item: dict) -> torch.Tensor:
    """Answer/evidence matter more than repeated separators and EOS; all are still supervised."""
    labels = item['labels'][:,1:]
    values = F.cross_entropy(logits[:,:-1].float().flatten(0,1), labels.flatten(),
                             ignore_index=-100, reduction='none').view_as(labels)
    weights = torch.zeros_like(values)
    start = len(item['prefix'])-1
    target_ids = item['input_ids'][0, len(item['prefix']):].tolist()
    evidence = False
    for offset, token in enumerate(target_ids):
        if token == 10+ord('|'):
            weight, evidence = 0.1, True
        elif token in (2, 10+ord(',')):
            weight = 0.1
        else:
            weight = 0.5 if evidence else 1.0
        weights[0,start+offset] = weight
    if not weights.sum() > 0:
        raise ValueError('No supervised weights')
    return (values*weights).sum()/weights.sum()


@torch.no_grad()
def infer(model, images: torch.Tensor, geometry: torch.Tensor, prompt: str, limit: int) -> dict:
    """Only pixels, geometry and the question enter generation; no target, proof or filename."""
    ids = [1, *encode(prompt), 7]
    if len(ids)+limit > model.config.max_text_tokens:
        raise ValueError('Generation would exceed text budget')
    result, ended = [], False
    model.eval()
    for _ in range(limit):
        logits = model(torch.tensor([ids]), images, geometry)['logits'][0,-1].float()
        logits[:2], logits[3:10] = -torch.inf, -torch.inf
        if logits.numel() > 266:
            logits[266:] = -torch.inf
        token = int(logits.argmax())
        if token == 2:
            ended = True
            break
        result.append(token)
        ids.append(token)
    try:
        text, valid = decode(result), True
    except UnicodeDecodeError:
        text = bytes(t-10 for t in result).decode('utf-8', errors='replace')
        valid = False
    return {'text': text, 'valid_utf8': valid, 'eos': ended}


def score(pred: dict, proof: dict) -> dict:
    parts = pred['text'].split('|')
    valid = (pred['eos'] and pred['valid_utf8'] and len(parts) == 2
             and bool(parts[0]) and parts[1] in ('','1','2','1,2'))
    if not valid:
        return {'format_valid': False, 'answer_correct': False, 'evidence_correct': False,
                'joint_correct': False, 'abstained': False}
    answer, pages = parts
    expected_answer = '?' if proof['answer'] is None else proof['answer']
    answer_ok = answer == expected_answer
    evidence_ok = pages == ','.join(map(str, proof['pages']))
    return {'format_valid': True, 'answer_correct': answer_ok, 'evidence_correct': evidence_ok,
            'joint_correct': answer_ok and evidence_ok, 'abstained': answer == '?'}


def evaluate(model, data: dict, root: Path, dtype, plan: Plan) -> dict:
    """Bounded validation subset; held-out families are not evaluated by training."""
    candidates = [r for r in data['records'] if r['split'] == 'validation' and r['id'].endswith('-v0')]
    # One pair per task before adding a second; no cherry-picking successful predictions.
    selected = []
    for task in TASKS:
        selected += [r for r in candidates if r['task'] == task][:2]
    selected.sort(key=lambda r: (int(r['family'][1:]), TASKS.index(r['task'])))
    selected = selected[:plan.eval_pairs]
    examples = []
    for row in selected:
        original = sample(row, root, model.config.max_pages, dtype)
        alt = data['by_id'][row['counterfactual']]
        changed = sample(alt, root, model.config.max_pages, dtype)
        generated = infer(model, original['images'], original['geometry'], row['prompt'], plan.max_new_tokens)
        blank = infer(model, torch.ones_like(original['images']), original['geometry'], row['prompt'], plan.max_new_tokens)
        swapped = infer(model, changed['images'], changed['geometry'], row['prompt'], plan.max_new_tokens)
        examples.append({'id': row['id'], 'task': row['task'], 'prompt': row['prompt'],
                         'expected': row['target'], 'prediction': generated,
                         'score': score(generated,row['proof']), 'blank': blank,
                         'blank_score': score(blank,row['proof']), 'counterfactual_id': alt['id'],
                         'counterfactual_expected': alt['target'], 'counterfactual': swapped,
                         'counterfactual_score': score(swapped,alt['proof'])})
    return {'count': len(examples),
            'answer_correct': sum(e['score']['answer_correct'] for e in examples),
            'joint_correct': sum(e['score']['joint_correct'] for e in examples),
            'paired_joint_correct': sum(e['score']['joint_correct'] and e['counterfactual_score']['joint_correct'] for e in examples),
            'format_valid': sum(e['score']['format_valid'] for e in examples),
            'holdout_evaluated': False, 'independent_real_benchmark': False, 'examples': examples}


def run(manifest: Path, out: Path, cfg: ScaleConfig, plan: Plan, *, dtype_name: str = 'float32',
        mapped: bool = False, streaming: bool = False, allow_large: bool = False,
        resume: bool = False, segment_steps: int | None = None) -> dict:
    if dtype_name not in ('float32','bfloat16') or cfg.vocab_size != 266:
        raise ValueError('CQ-01 supports explicit byte-prototype FP32/BF16 only')
    if cfg.max_image_side < 256 or cfg.max_tiles < 2 or cfg.max_text_tokens < 128:
        raise ValueError('Model cannot receive the unchanged 256px curriculum images')
    if segment_steps is not None and (type(segment_steps) is not int or not 1 <= segment_steps <= plan.steps):
        raise ValueError('Invalid finite segment')
    data = validate(manifest)
    frozen = digest_file(manifest)
    sequence = select_training(data, plan)
    dtype = getattr(torch, dtype_name)
    resources = resource_gate(cfg, dtype, out, allow_large, mapped, streaming)
    contract = {'curriculum_source_sha256': source_hash(), 'manifest_sha256': frozen,
                'plan': asdict(plan), 'dtype': dtype_name, 'streaming': streaming,
                'torch': str(torch.__version__), 'model_config': asdict(cfg),
                'optimizer': 'SGD-no-momentum-with-warmup-cosine; not production AdamW',
                'sampling': [r['id'] for r in sequence]}
    torch.set_num_threads(plan.threads)
    out.mkdir(parents=True, exist_ok=resume)
    lock = out/'RUNNING.lock'
    with lock.open('x') as stream:
        stream.write('Discard partially updated RAM after failure; resume completed checkpoint only.')
    optimizer, backing = None, None
    started = time.monotonic()
    try:
        start = 0
        if resume:
            parent = get_latest(out)
            previous = index_at(parent)
            if previous['training'].get('curriculum_contract') != contract:
                raise ValueError('Resume curriculum/data/plan/model mismatch')
            start = previous['training']['completed_steps']
            if start >= plan.steps:
                raise ValueError('Plan already complete')
        torch.manual_seed(plan.seed)
        if mapped:
            backing = out/f'work-{start:04d}.bin'
        model = build(cfg, dtype=dtype, backing_file=backing, initialize=not resume)
        if resume:
            previous = load_into(model, parent)
            torch.set_rng_state(torch.tensor(previous['torch_rng'],dtype=torch.uint8))
        counts = parameter_report(model)
        print(json.dumps({'phase':'allocated', **counts}), flush=True)
        optimizer = AuditedSGD(model, schedule(start,plan), streaming)
        fixed = sample(sequence[0], manifest.parent, cfg.max_pages, dtype)
        kwargs = {k:v for k,v in fixed.items() if k not in ('prefix','supervised_tokens')}
        with torch.no_grad():
            before = float(model(**kwargs)['loss'])
        stop = min(plan.steps, start+(segment_steps or plan.steps))
        audits, seen, supervised = [], [], 0
        for step in range(start, stop):
            if time.monotonic()-started > plan.seconds:
                break
            row = sequence[step]
            item = sample(row,manifest.parent,cfg.max_pages,dtype)
            kwargs = {k:v for k,v in item.items() if k not in ('prefix','supervised_tokens','labels')}
            model.train()
            output = model(**kwargs)
            loss = weighted_loss(output['logits'],item)
            optimizer.lr = schedule(step,plan)
            audit = optimizer.backward(loss)
            audits.append({'step':step+1, 'task':row['task'], 'sample':row['id'],
                           'weighted_loss':float(loss.detach()), 'lr':optimizer.lr,
                           **{k:v for k,v in audit.items() if k != 'by_tensor'}})
            seen.append(row['id'])
            supervised += item['supervised_tokens']
            print(json.dumps(audits[-1]),flush=True)
            del output, loss
        optimizer.close()
        optimizer = None
        done = start+len(audits)
        if not audits:
            raise TimeoutError('No completed optimizer step')
        if digest_file(manifest) != frozen:
            raise ValueError('Manifest changed during training; discard RAM')
        fixed_kwargs = {k:v for k,v in fixed.items() if k not in ('prefix','supervised_tokens')}
        with torch.no_grad():
            after = float(model(**fixed_kwargs)['loss'])
        training = {'completed_steps':done, 'starting_step':start, 'steps_this_run':len(audits),
                    'learning_rate':schedule(done-1,plan), 'dtype':dtype_name,
                    'curriculum_contract':contract, 'observed_sample_ids':seen,
                    'supervised_answer_tokens_this_run':supervised, 'full_pretraining':False,
                    'precision_note':'BF16 can round away updates; audit reports actual changed elements',
                    'optimizer_momentum':0, 'global_clipping':False, 'gradient_accumulation':False}
        checkpoint = out/f'checkpoint-{done:04d}'
        index = save(model,checkpoint,training)
        set_latest(out,checkpoint)
        report = {'status':'completed_finite_plan' if done == plan.steps else 'partial_finite_plan',
                  'parameters':counts, 'training':training, 'audit':audits,
                  'first_train_example_nll_before':before, 'first_train_example_nll_after':after,
                  'checkpoint_shards':len(index['shards']), 'checkpoint_index_sha256':digest_file(checkpoint/'index.json'),
                  'checkpoint_weights_persisted_locally':True, 'resources':resources,
                  'production_ready':False, 'pretraining_complete':False, 'real_scan_quality_measured':False,
                  'elapsed_before_evaluation':time.monotonic()-started,
                  'runtime':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu'}}
        # Save training evidence BEFORE optional expensive evaluation; failures must not erase it.
        write_json(out/f'train-report-{done:04d}.json',report)
        if done == plan.steps:
            report['evaluation'] = evaluate(model,data,manifest.parent,dtype,plan)
        else:
            report['evaluation'] = None
        report['elapsed_total'] = time.monotonic()-started
        write_json(out/f'report-{done:04d}.json',report)
        return report
    except (Exception,KeyboardInterrupt) as error:
        write_json(out/f'FAILED-{time.time_ns()}.json',{'error':type(error).__name__,
                   'discard_mutated_RAM':True,'message':str(error),'production_ready':False})
        raise
    finally:
        if optimizer is not None:
            optimizer.close()
        lock.unlink(missing_ok=True)
        if backing is not None and platform.system() != 'Windows':
            backing.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description='CQ-01 evidence curriculum, finite training NOT quality certification')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path,default=Path('configs/training/curriculum_q1.yaml'))
    parser.add_argument('--dtype',choices=('float32','bfloat16'),default='float32')
    parser.add_argument('--allow-large',action='store_true')
    parser.add_argument('--mapped',action='store_true')
    parser.add_argument('--streaming',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--segment-steps',type=int)
    parser.add_argument('--tiny',action='store_true')
    args = parser.parse_args()
    from dataclasses import replace
    plan = Plan(**yaml.safe_load(args.plan.read_text(encoding='utf-8')))
    manifest = args.output/'dataset/manifest.json'
    if not args.resume:
        if args.output.exists():
            parser.error('Use a new output or explicit --resume')
        prepare(manifest.parent)
    cfg = replace(tiny_config(),max_image_side=256) if args.tiny else ScaleConfig()
    result = run(manifest,args.output/'run',cfg,plan,dtype_name=args.dtype,allow_large=args.allow_large,
                 mapped=args.mapped,streaming=args.streaming,resume=args.resume,segment_steps=args.segment_steps)
    print(json.dumps({k:v for k,v in result.items() if k not in ('audit','evaluation','training')},indent=2))
    if result['evaluation']:
        print('EVALUATION',json.dumps({k:v for k,v in result['evaluation'].items() if k != 'examples'}))
    return 0 if result['status'] == 'completed_finite_plan' or args.segment_steps else 1


if __name__ == '__main__':
    raise SystemExit(main())
