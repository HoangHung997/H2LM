from __future__ import annotations
import json
import shutil
from collections import Counter
from dataclasses import replace
import pytest
import torch
from h2lm.curriculum.data import TASKS, alternative, compact, prepare, sample, solve, validate
from h2lm.curriculum.train import Plan, infer, run, schedule, score, select_training, weighted_loss
from h2lm.scale.config import tiny_config
from h2lm.scale.model import build
from h2lm.scale.storage import get_latest, load_into

@pytest.fixture(scope='module')
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp('cq') / 'data'
    prepare(root, 12, 301)
    return root / 'manifest.json'

@pytest.fixture
def small_cfg():
    return replace(tiny_config(), max_image_side=256, patch_size=32, resampler_tokens=2, layers=1, vision_layers=1)

@pytest.mark.parametrize('key,value', [('steps', 0), ('steps', True), ('steps', 129), ('warmup', 99), ('seconds', 1201), ('max_new_tokens', 999), ('threads', 5), ('peak_lr', float('nan')), ('floor_lr', 0), ('eval_pairs', 0)])
def test_bounds(key, value):
    with pytest.raises(ValueError):
        Plan(**{key: value})


def test_schedule_limits_and_endpoint():
    p = Plan()
    values = [schedule(i, p) for i in range(p.steps)]
    assert values[p.warmup - 1] == p.peak_lr
    assert values[-1] == p.floor_lr
    assert max(values) == p.peak_lr and min(values) > 0
    with pytest.raises(ValueError):
        schedule(p.steps, p)


def test_oracle_math_and_scope():
    scene = {'codes': [875, 226], 'counts': [2, 5, 6, 3], 'role': 'GS', 'masked': False}
    assert compact(solve(scene, 'table_sum')) == '16|1,2'
    assert compact(solve(scene, 'compare')) == '1|1,2'
    assert compact(solve(scene, 'role')) == 'GS|1'
    assert compact(solve(scene, 'locate', 226)) == '2|2'
    scene['masked'] = True
    assert compact(solve(scene, 'read')) == '?|'
    assert compact(solve(scene, 'missing')) == '?|'


@pytest.mark.parametrize('task', TASKS)
def test_counterfactual_changes_answer_not_question(task):
    s = {'codes': [101, 809], 'counts': [2, 4, 3, 7], 'role': 'GS', 'masked': False}
    a = alternative(s, task)
    assert compact(solve(s, task, 101)) != compact(solve(a, task, 101))
    assert a is not s


def test_dataset_shape_scope_and_split(dataset):
    d = validate(dataset)
    assert len(d['records']) == 144
    assert Counter((r['split'] for r in d['records'])) == {'train': 96, 'validation': 24, 'holdout': 24}
    assert all((r['font_index'] == 2 for r in d['records'] if r['split'] == 'holdout'))
    assert all((r['font_index'] in (0, 1) for r in d['records'] if r['split'] != 'holdout'))
    sequence = select_training(d, Plan())
    assert len(sequence) == 48 and all((r['split'] == 'train' for r in sequence))
    assert set((r['task'] for r in sequence)) == set(TASKS)
    assert select_training(d, Plan()) == sequence
    assert all((r['task'] == 'read' for r in sequence[:12]))


def changed_manifest(dataset, tmp_path, mutate):
    root = tmp_path / 'copy'
    shutil.copytree(dataset.parent, root)
    p = root / 'manifest.json'
    d = json.loads(p.read_text())
    mutate(d)
    p.write_text(json.dumps(d), encoding='utf-8')
    return p


@pytest.mark.parametrize('what', ['answer', 'evidence', 'question', 'pair', 'approval', 'split', 'path', 'image'])
def test_corruption_rejected(dataset, tmp_path, what):

    def change(d):
        r = d['records'][0]
        if what == 'answer':
            r['target'] = '999|1'
        elif what == 'evidence':
            r['proof']['pages'] = [2]
        elif what == 'question':
            r['prompt'] = 'invented'
        elif what == 'pair':
            r['counterfactual'] = 'absent'
        elif what == 'approval':
            r['production_approved'] = True
        elif what == 'split':
            r['split'] = 'validation'
        elif what == 'path':
            r['pages'][0]['path'] = '../../secret.png'
        elif what == 'image':
            r['pages'][0]['sha256'] = '0' * 64
    path = changed_manifest(dataset, tmp_path, change)
    with pytest.raises(ValueError):
        validate(path)


def test_mask_and_loss_no_prompt_credit(dataset, small_cfg):
    d = validate(dataset)
    row = d['records'][0]
    item = sample(row, dataset.parent, 4)
    assert item['images'].shape == (1, 2, 3, 256, 256)
    assert (item['labels'][0, :len(item['prefix'])] == -100).all()
    model = build(small_cfg)
    torch.set_num_threads(2)
    out = model(**{k: v for k, v in item.items() if k not in ('prefix', 'supervised_tokens')})
    loss = weighted_loss(out['logits'], item)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.vision.patch.weight.grad is not None
    assert torch.count_nonzero(model.vision.patch.weight.grad) > 0


def test_score_requires_both_answer_and_evidence():
    p = {'answer': '4', 'pages': [1, 2], 'status': 'supported', 'operation': 'sum'}

    def s(text, eos=True):
        return score({'text': text, 'eos': eos, 'valid_utf8': True}, p)
    assert s('4|1,2')['joint_correct']
    assert s('4|2')['answer_correct'] and (not s('4|2')['joint_correct'])
    assert not s('4|1,2', False)['format_valid']
    assert not s('4|1|2')['format_valid']
    assert s('?|')['abstained'] and (not s('?|')['joint_correct'])


def test_generation_interface_does_not_accept_reference(small_cfg):
    import inspect
    assert list(inspect.signature(infer).parameters) == ['model', 'images', 'geometry', 'prompt', 'limit']
    m = build(small_cfg)
    px = torch.zeros(1, 2, 3, 256, 256)
    g = torch.tensor([[[0, 0, 1, 1, 0], [0, 0, 1, 1, 0.25]]])
    pred = infer(m, px, g, 'Mã?', 8)
    assert set(pred) == {'text', 'eos', 'valid_utf8'}


def test_segment_resume_weights_equal(dataset, tmp_path, small_cfg):
    p = Plan(steps=6, warmup=1, eval_pairs=1, max_new_tokens=8)
    full = run(dataset, tmp_path / 'full', small_cfg, p)
    a = run(dataset, tmp_path / 'seg', small_cfg, p, segment_steps=3)
    b = run(dataset, tmp_path / 'seg', small_cfg, p, resume=True)
    assert a['training']['completed_steps'] == 3 and b['training']['completed_steps'] == 6
    assert full['training']['steps_this_run'] == 6
    m1, m2 = (build(small_cfg), build(small_cfg))
    load_into(m1, get_latest(tmp_path / 'full'))
    load_into(m2, get_latest(tmp_path / 'seg'))
    assert all((torch.equal(a, b) for a, b in zip(m1.parameters(), m2.parameters())))
    assert full['evaluation']['holdout_evaluated'] is False
    assert full['production_ready'] is False
    assert b['training']['curriculum_contract']['manifest_sha256'] == full['training']['curriculum_contract']['manifest_sha256']
    with pytest.raises(ValueError, match='complete'):
        run(dataset, tmp_path / 'seg', small_cfg, p, resume=True)


def test_resume_mismatch_writer_guard(dataset, tmp_path, small_cfg):
    p = Plan(steps=6, warmup=1, eval_pairs=1, max_new_tokens=8)
    run(dataset, tmp_path / 'run', small_cfg, p, segment_steps=2)
    with pytest.raises(ValueError, match='mismatch'):
        run(dataset, tmp_path / 'run', small_cfg, replace(p, seed=1), resume=True)
    with pytest.raises(FileExistsError):
        run(dataset, tmp_path / 'run', small_cfg, p)
    (tmp_path / 'run/RUNNING.lock').write_text('keep')
    with pytest.raises(FileExistsError):
        run(dataset, tmp_path / 'run', small_cfg, p, resume=True)
    assert (tmp_path / 'run/RUNNING.lock').read_text() == 'keep'
