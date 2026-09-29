from __future__ import annotations

import copy
import json
import shutil
from dataclasses import replace

import pytest
import torch
from PIL import Image

from h2lm.experiments import crop2_data
from h2lm.experiments.crop2_data import load, prepare, tensor_from_image
from h2lm.experiments.crop2_model import CropVLM
from h2lm.experiments.crop2_train import (
    CropConfig,
    evaluate,
    predict,
    read_checkpoint,
    train,
)
from h2lm.experiments.data import dump
from h2lm.experiments.engine import batch


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp('dataset')/'generated'
    prepare(out, count=8, validation_count=8, seed=7)
    return out/'manifest.json'


@pytest.fixture
def cfg():
    return CropConfig(steps=4, batch_size=4, height=32, width=128, d_model=32,
                      train_images=8, validation_images=8, save_every=2)


@pytest.mark.parametrize('key,value', [
    ('steps',0),('steps',True),('steps',3001),('seconds',601),('d_model',33),('width',130),
    ('height',33),('train_images',4097),('validation_images',0),('threads',5),
    ('learning_rate',float('nan')),('ctc_weight',float('inf')),('ctc_weight',-1),
])
def test_config_limits(key,value):
    with pytest.raises(ValueError):
        CropConfig(**{key:value})


def test_config_unknown(tmp_path):
    path=tmp_path/'bad.yaml'
    path.write_text('unknown: true\n')
    with pytest.raises(ValueError):
        CropConfig.read(path)


def test_generator_disjoint_and_reproducible(data,tmp_path):
    original=json.loads(data.read_text())
    repeated=prepare(tmp_path/'repeat',8,8,7)
    assert original==repeated
    train_values={r['target'] for r in original['records'] if r['split']=='train'}
    val_values={r['target'] for r in original['records'] if r['split']=='validation'}
    assert not train_values & val_values
    assert not train_values & crop2_data.REAL_RESERVED
    assert len(original['fonts'])>=2
    assert all(r['training_approval']=='experiment_only' for r in original['records'])
    with pytest.raises(FileExistsError):
        prepare(tmp_path/'repeat',8,8,7)


def test_aspect_ratio_and_source_preserved(tmp_path):
    with Image.new('L',(400,40),0) as im:
        original=im.tobytes()
        tensor=tensor_from_image(im,48,320)
        assert im.tobytes()==original
    assert tensor.shape==(1,48,320)
    # 10:1 line fits as 320x32, without square stretching or cropping.
    ink=(tensor[0]<0).sum(dim=1)
    assert int((ink==320).sum())==32
    assert torch.isfinite(tensor).all()


def test_dataset_detects_cross_split_value_and_image(data,tmp_path):
    source=json.loads(data.read_text())
    for key in ('target','sha256'):
        value=copy.deepcopy(source)
        value['records'][8][key]=value['records'][0][key]
        folder=tmp_path/key
        shutil.copytree(data.parent,folder)
        path=folder/'manifest.json'
        dump(path,value)
        with pytest.raises(ValueError, match='leakage'):
            load(path,32,128,('train',))


def test_validation_pixels_never_read_while_training(data,monkeypatch):
    read=[]
    original=crop2_data.image_tensor
    def spy(path,height,width):
        read.append(path.name)
        return original(path,height,width)
    monkeypatch.setattr(crop2_data,'image_tensor',spy)
    rows=load(data,32,128,('train',))
    assert len(rows)==8
    assert set(read)=={f'{i:05}.png' for i in range(8)}
    assert all(r['split']=='train' for r in rows)


def test_prompts_masked_and_input_causal(data):
    torch.set_num_threads(2)
    torch.manual_seed(29)
    rows=load(data,32,128,('train',))
    ids,px,labels=batch(rows[:2])
    for i,row in enumerate(rows[:2]):
        assert (labels[i,:len(row['prefix'])]==-100).all()
    model=CropVLM(32,32,128).eval()
    changed=ids.clone()
    changed[:,10:]=44
    with torch.no_grad():
        a=model(ids,px)[0]
        b=model(changed,px)[0]
    torch.testing.assert_close(a[:,:10],b[:,:10],rtol=1e-5,atol=1e-5)
    assert not torch.equal(model.layers[0].self_attn.in_proj_weight,
                           model.layers[1].self_attn.in_proj_weight)


def test_generation_never_calls_ctc_head(monkeypatch):
    model=CropVLM(32,32,128)
    def forbidden(*args,**kwargs):
        raise AssertionError('CTC is not an input or decoder at inference')
    monkeypatch.setattr(model.ctc_head,'forward',forbidden)
    output=predict(model,torch.ones(1,1,32,128),'Số?',max_new=2)
    assert set(output[0])=={'text','eos','valid_utf8'}


def test_generation_limits():
    model=CropVLM(32,32,128)
    with pytest.raises(ValueError):
        predict(model,torch.ones(1,1,32,128),'x'*40)
    with pytest.raises(ValueError):
        predict(model,torch.ones(1,1,32,128),'Số?',max_new=999)
    with pytest.raises(ValueError):
        model.memory(torch.ones(1,1,128,128))
    with pytest.raises(ValueError):
        model.memory(torch.full((1,1,32,128),float('nan')))


def test_resume_same_weights_and_checkpoint_integrity(data,tmp_path,cfg):
    full=train(data,tmp_path/'full',cfg)
    part=train(data,tmp_path/'split',cfg,segment_steps=2)
    assert not part['completed_steps']
    resumed=train(data,tmp_path/'split',cfg,resume=True)
    a,b=read_checkpoint(tmp_path/'full'),read_checkpoint(tmp_path/'split')
    assert full['weights_changed'] and full['vision_weight_max_change']>0
    assert full['reload_matches'] and resumed['reload_matches']
    for key,tensor in a['model'].items():
        assert torch.equal(tensor,b['model'][key]),key
    assert not full['production_ready']
    assert full['training_counts']['real']==0
    with pytest.raises(ValueError,match='already complete'):
        train(data,tmp_path/'split',cfg,resume=True)
    pointer=json.loads((tmp_path/'full/latest.json').read_text())
    (tmp_path/'full'/pointer['file']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        read_checkpoint(tmp_path/'full')


def test_mismatch_and_single_writer(data,tmp_path,cfg):
    train(data,tmp_path/'run',cfg,segment_steps=2)
    with pytest.raises(ValueError,match='mismatch'):
        train(data,tmp_path/'run',replace(cfg,seed=9),resume=True)
    assert not (tmp_path/'run/RUNNING.lock').exists()
    with pytest.raises(FileExistsError):
        train(data,tmp_path/'run',cfg)
    (tmp_path/'run/RUNNING.lock').write_text('existing writer')
    with pytest.raises(FileExistsError):
        train(data,tmp_path/'run',cfg,resume=True)
    assert (tmp_path/'run/RUNNING.lock').read_text()=='existing writer'


def test_controls_use_different_answers(data):
    rows=load(data,32,128,('validation',))
    report=evaluate(CropVLM(32,32,128),rows)
    assert report['count']==8
    assert not report['independent_benchmark']
    assert all(r['target']!=r['wrong_image_target'] for r in report['examples'])


def test_real_probe_permission_cannot_be_implied(tmp_path,cfg):
    from h2lm.experiments.crop2_probe import load_probe
    dump(tmp_path/'draft_review.json',{'permission': {},'regions': []})
    with pytest.raises(ValueError,match='permission'):
        load_probe(tmp_path,cfg)


def test_silver_adaptation_opt_in_before_read(tmp_path):
    from h2lm.experiments.scan_adapt import adapt
    with pytest.raises(ValueError,match='opt-in'):
        adapt(tmp_path,tmp_path,tmp_path,tmp_path/'out')
    with pytest.raises(ValueError):
        adapt(tmp_path,tmp_path,tmp_path,tmp_path/'out',allow_silver=True,steps=999)
    assert not (tmp_path/'out').exists()


def test_probe_reports_adaptation_provenance(tmp_path,monkeypatch,cfg):
    from dataclasses import asdict

    from h2lm.experiments import crop2_probe
    from h2lm.experiments.crop2_train import source_hash
    model=CropVLM(cfg.d_model,cfg.height,cfg.width)
    fit_ids=['dt-number','dt-date','tb-number','tb-date']
    state={'contract':{'config':asdict(cfg),'source_sha256':source_hash(),
                      'adaptation':{'fit_region_ids':fit_ids}},
           'model':model.state_dict(),'step':304}
    (tmp_path/'fake.pt').write_bytes(b'unit-fixture-only')
    dump(tmp_path/'latest.json',{'file':'fake.pt'})
    monkeypatch.setattr(crop2_probe,'read_checkpoint',lambda _:state)
    monkeypatch.setattr(crop2_probe,'load_probe',lambda *args:[])
    monkeypatch.setattr(crop2_probe,'evaluate',lambda *args:{'count':6})
    report=crop2_probe.run_probe(tmp_path,tmp_path,tmp_path/'report.json')
    assert report['training_real_samples']==4 and report['fit_region_ids']==fit_ids
    assert report['checkpoint_steps']==304 and not report['independent_benchmark']
