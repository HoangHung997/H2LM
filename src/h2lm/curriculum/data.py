"""Generate paired visual tasks from explicit scene facts; no OCR/teacher/API or hidden labels."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from h2lm.scale.fixture import encode
from h2lm.scale.storage import digest_file, write_json

TASKS = ('read', 'locate', 'compare', 'table_sum', 'role', 'missing')
SPLITS = ('train', 'validation', 'holdout')


def font_paths() -> list[Path]:
    roots = ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
             '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
             '/usr/share/fonts/truetype/liberation2/LiberationSerif-Italic.ttf',
             '/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf',
             'C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/times.ttf',
             'C:/Windows/Fonts/timesi.ttf']
    found, hashes = [], set()
    for name in roots:
        path = Path(name)
        if path.is_file() and digest_file(path) not in hashes:
            hashes.add(digest_file(path))
            found.append(path)
    if len(found) < 3:
        raise ValueError('Need three local Vietnamese-capable fonts; none are distributed')
    return found[:3]


def solve(scene: dict, task: str, query_code: int | None = None) -> dict:
    """Ground-truth oracle for FICTIONAL generated scenes, not a legal decision engine."""
    if task not in TASKS or set(scene) != {'codes', 'counts', 'role', 'masked'}:
        raise ValueError('Unsupported task or scene schema')
    codes, counts = scene['codes'], scene['counts']
    if (not isinstance(codes, list) or len(codes) != 2 or codes[0] == codes[1]
            or any(type(v) is not int or not 100 <= v <= 999 for v in codes)
            or not isinstance(counts, list) or len(counts) != 4
            or any(type(v) is not int or not 1 <= v <= 10 for v in counts)
            or scene['role'] not in ('TC', 'GS') or type(scene['masked']) is not bool):
        raise ValueError('Malformed generated scene')
    if task in ('read', 'missing'):
        if scene['masked']:
            return {'answer': None, 'pages': [], 'status': 'missing_evidence', 'operation': 'abstain'}
        answer, pages, operation = str(codes[0]), [1], 'copy'
    elif task == 'locate':
        if query_code not in codes or scene['masked']:
            raise ValueError('Location query must refer to a visible generated code')
        page = codes.index(query_code) + 1
        answer, pages, operation = str(page), [page], 'locate'
    elif task == 'compare':
        answer, pages, operation = str(1 if codes[0] > codes[1] else 2), [1, 2], 'compare'
    elif task == 'table_sum':
        answer, pages, operation = str(sum(counts)), [1, 2], 'sum'
    else:
        answer, pages, operation = scene['role'], [1], 'copy_role'
    return {'answer': answer, 'pages': pages, 'status': 'supported', 'operation': operation}


def compact(proof: dict) -> str:
    return ('?' if proof['answer'] is None else proof['answer']) + '|' + ','.join(map(str, proof['pages']))


def question(task: str, query_code: int) -> str:
    return {'read': 'Đọc mã trang 1.', 'missing': 'Đọc mã trang 1.',
            'locate': f'Mã {query_code} ở trang nào?', 'compare': 'Trang có mã lớn hơn?',
            'table_sum': 'Tổng số lượng hai trang?',
            'role': 'Vai trò A? TC=thi công, GS=giám sát.'}[task]


def alternative(scene: dict, task: str) -> dict:
    new = copy.deepcopy(scene)
    if task == 'read':
        new['codes'][0] = 100 + ((new['codes'][0] - 100 + 1) % 900)
        if new['codes'][0] == new['codes'][1]:
            new['codes'][0] = 100 + ((new['codes'][0] - 100 + 1) % 900)
    elif task in ('locate', 'compare'):
        new['codes'].reverse()
    elif task == 'table_sum':
        new['counts'][3] += 1
    elif task == 'role':
        new['role'] = 'GS' if new['role'] == 'TC' else 'TC'
    else:
        new['masked'] = not new['masked']
    return new


def render(scene: dict, task: str, page: int, font: Path, degraded: bool) -> tuple[Image.Image, dict]:
    """256px two-page cards; no downsampling, clipping or synthetic image-file labels."""
    im = Image.new('RGB', (256, 256), 'white')
    draw = ImageDraw.Draw(im)
    small = ImageFont.truetype(str(font), 16)
    regular = ImageFont.truetype(str(font), 22)
    large = ImageFont.truetype(str(font), 32)
    draw.text((12, 10), f'Trang {page}', font=small, fill='black')
    box = [12, 50, 244, 218]
    if task in ('role', 'table_sum'):
        draw.text((140, 10), f"HS:{scene['codes'][page-1]}", font=small, fill='black')
    if task == 'table_sum':
        values = scene['counts'][(page-1)*2:page*2]
        draw.text((16, 48), 'Mục     Số lượng', font=regular, fill='black')
        for row, number in enumerate(values):
            y = 90 + row*55
            draw.rectangle((12, y, 244, y+48), outline='black', width=1)
            draw.text((24, y+4), f'{chr(65+(page-1)*2+row)}       {number}', font=large, fill='black')
    elif task == 'role':
        draw.text((14, 62), 'Đơn vị A' if page == 1 else 'Thông tin phụ', font=regular, fill='black')
        text = ('Thi công' if scene['role'] == 'TC' else 'Giám sát') if page == 1 else 'Mẫu giả lập'
        draw.text((14, 118), text, font=large, fill='black')
    else:
        draw.text((16, 50), 'Mã hồ sơ', font=regular, fill='black')
        if not (scene['masked'] and page == 1):
            draw.text((30, 105), str(scene['codes'][page-1]), font=large, fill='black')
        else:
            # Value is never drawn into the pixels. It cannot survive a hidden text layer.
            draw.rectangle((16, 95, 234, 167), fill=(190, 190, 190))
            draw.line((16, 95, 234, 167), fill=(150, 150, 150), width=2)
    draw.text((12, 233), 'MẪU GIẢ LẬP', font=small, fill='black')
    if degraded:
        softened = im.filter(ImageFilter.GaussianBlur(0.35))
        im.close()
        im = ImageEnhance.Contrast(softened).enhance(0.80)
        softened.close()
        buf = io.BytesIO()
        im.save(buf, 'JPEG', quality=62)
        im.close()
        with Image.open(io.BytesIO(buf.getvalue())) as jpeg:
            im = jpeg.convert('RGB')
    return im, {'page': page, 'bbox_pixels_xyxy': box, 'source_kind': 'programmatic_card'}


def prepare(out: Path, families: int = 96, seed: int = 301) -> dict:
    if type(families) is not int or not 12 <= families <= 192 or families % 6:
        raise ValueError('families must be a multiple of 6 in [12,192]')
    if type(seed) is not int or not 0 <= seed <= 100000:
        raise ValueError('Invalid dataset seed')
    out.mkdir(parents=True, exist_ok=False)
    (out/'images').mkdir()
    fonts = font_paths()
    rng = random.Random(seed)
    records = []
    code_pool = rng.sample(range(100, 1000), families*2)
    for family in range(families):
        split = 'train' if family < families*2//3 else ('validation' if family < families*5//6 else 'holdout')
        font_index = 2 if split == 'holdout' else rng.randrange(2)
        # Codes are family-specific; all variants/tasks remain in one split.
        codes = code_pool[family*2:family*2+2]
        rng.shuffle(codes)
        scene = {'codes': codes, 'counts': [rng.randint(1, 9) for _ in range(4)],
                 'role': rng.choice(['TC', 'GS']), 'masked': False}
        for task in TASKS:
            original = copy.deepcopy(scene)
            if task == 'missing':
                original['masked'] = bool(family % 2)
            query_code = scene['codes'][0]
            prompt = question(task, query_code)
            for variant, facts in enumerate((original, alternative(original, task))):
                identifier = f'g{family:03}-{task}-v{variant}'
                proof = solve(facts, task, query_code)
                pages = []
                for page in (1, 2):
                    image, box = render(facts, task, page, fonts[font_index], bool(family % 2))
                    path = out/'images'/f'{identifier}-p{page}.png'
                    with image:
                        image.save(path)
                    pages.append({'path': path.relative_to(out).as_posix(), 'sha256': digest_file(path), **box})
                records.append({'id': identifier, 'family': f'g{family:03}', 'split': split,
                                'task': task, 'prompt': prompt, 'query_code': query_code,
                                'scene': facts, 'proof': proof, 'target': compact(proof),
                                'counterfactual': f'g{family:03}-{task}-v{1-variant}',
                                'pages': pages, 'font_index': font_index, 'degraded': bool(family % 2),
                                'label_origin': 'programmatic_scene_oracle', 'production_approved': False})
    manifest = {'schema_version': 1, 'dataset_kind': 'synthetic_curriculum_v1',
                'seed': seed, 'families': families, 'records': records,
                'fonts': [{'name': p.name, 'sha256': digest_file(p)} for p in fonts],
                'independent_real_benchmark': False,
                'holdout_rule': 'last sixth of families, distinct font; synthetic only; no training use',
                'purpose': 'Read then ground then combine evidence; NOT real laws or scan authenticity'}
    write_json(out/'manifest.json', manifest)
    validate(out/'manifest.json')
    return manifest


def validate(path: Path) -> dict:
    if path.stat().st_size > 20_000_000:
        raise ValueError('Manifest exceeds bound')
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema_version') != 1 or data.get('dataset_kind') != 'synthetic_curriculum_v1':
        raise ValueError('Unsupported curriculum schema')
    rows = data['records']
    if not 144 <= len(rows) <= 2304:
        raise ValueError('Invalid bounded record count')
    by_id, groups, documents = {}, {}, {}
    for row in rows:
        if row['id'] in by_id or row['split'] not in SPLITS:
            raise ValueError('Duplicate record or invalid split')
        if (row.get('production_approved') is not False
                or row.get('label_origin') != 'programmatic_scene_oracle'):
            raise ValueError('Do not promote experimental programmatic labels to real/human gold')
        if groups.setdefault(row['family'], row['split']) != row['split']:
            raise ValueError('Family leakage')
        expected = solve(row['scene'], row['task'], row['query_code'])
        if row['proof'] != expected or row['target'] != compact(expected):
            raise ValueError('Target/evidence disagrees with independent scene oracle')
        if row['prompt'] != question(row['task'], row['query_code']):
            raise ValueError('Question and scene disagree')
        if len(row['pages']) != 2 or len(encode(row['prompt']))+len(encode(row['target']))+3 > 128:
            raise ValueError('Invalid page count or text budget')
        fingerprint = hashlib.sha256(''.join(p['sha256'] for p in row['pages']).encode()).hexdigest()
        if documents.setdefault(fingerprint, row['split']) != row['split']:
            raise ValueError('Full document pixels leaked across splits')
        for page, p in enumerate(row['pages'], 1):
            image = (path.parent/p['path']).resolve()
            if (p['page'] != page or not image.is_relative_to(path.parent.resolve())
                    or not image.is_file() or image.stat().st_size > 2_000_000
                    or digest_file(image) != p['sha256']):
                raise ValueError('Invalid page/hash/path')
        by_id[row['id']] = row
    for row in rows:
        other = by_id.get(row['counterfactual'])
        if (not other or other['family'] != row['family'] or other['split'] != row['split']
                or other['task'] != row['task'] or other['prompt'] != row['prompt']
                or other['counterfactual'] != row['id'] or other['target'] == row['target']):
            raise ValueError('Invalid image counterfactual pair')
    data['by_id'] = by_id
    return data


def sample(row: dict, root: Path, max_pages: int, dtype: torch.dtype = torch.float32) -> dict:
    images = []
    for page in row['pages']:
        path = (root/page['path']).resolve()
        if (not path.is_relative_to(root.resolve()) or path.stat().st_size > 2_000_000
                or digest_file(path) != page['sha256']):
            raise ValueError('Image changed since freeze')
        with Image.open(path) as image:
            if image.size != (256, 256) or image.mode != 'RGB':
                raise ValueError('Unexpected image representation; no implicit resizing')
            t = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(256,256,3)
            images.append(t.permute(2,0,1).float()/127.5-1)
    prefix = [1, *encode(row['prompt']), 7]
    answer = [*encode(row['target']), 2]
    return {'input_ids': torch.tensor([prefix+answer]),
            'labels': torch.tensor([[-100]*len(prefix)+answer]),
            'images': torch.stack(images)[None].to(dtype),
            'geometry': torch.tensor([[[0,0,1,1,0],[0,0,1,1,1/max_pages]]]),
            'prefix': prefix, 'supervised_tokens': len(answer)}
