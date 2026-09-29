"""EXP-02 generated Vietnamese fields; content-disjoint synthetic splits, no external OCR."""
from __future__ import annotations

import io
import json
import random
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from .data import dump, encode, sha

# Already inspected real development fields. Never render these as synthetic training answers.
REAL_RESERVED = {'1542/QĐ-TCT', '28/08/2026', '3464/QĐ-BQP', '01/08/2023',
                 '3154/QĐ-BQP', '16/06/2026'}


def fonts() -> list[Path]:
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSerif-Italic.ttf',
        'C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/times.ttf',
        'C:/Windows/Fonts/timesi.ttf',
    ]
    found = [Path(p) for p in candidates if Path(p).is_file()]
    if len(found) < 2:
        raise ValueError('Two local Vietnamese-capable fonts are required; fonts are not bundled')
    return found


@lru_cache(maxsize=128)
def get_font(path: str, size: int):
    return ImageFont.truetype(path, size)


def render(text: str, font_path: Path, seed: int, degraded: bool) -> Image.Image:
    rng = random.Random(seed)
    font = get_font(str(font_path), rng.randint(23, 30))
    left, top, right, bottom = font.getbbox(text)
    pad = rng.randint(4, 10)
    image = Image.new('L', (right-left + pad*2, bottom-top + pad*2), 255)
    ImageDraw.Draw(image).text((pad-left, pad-top), text, font=font, fill=rng.randint(0, 55))
    if degraded:
        # Mild bounded degradation keeps glyphs present; these are not physical scans.
        rotated = image.rotate(rng.uniform(-1.3, 1.3), Image.Resampling.BICUBIC,
                               expand=True, fillcolor=255)
        image.close()
        image = rotated.filter(ImageFilter.GaussianBlur(rng.uniform(0.0, 0.45)))
        rotated.close()
        buf = io.BytesIO()
        image.save(buf, 'JPEG', quality=rng.randint(45, 85))
        image.close()
        with Image.open(io.BytesIO(buf.getvalue())) as jpg:
            image = jpg.convert('L')
    return image


def tensor_from_image(image: Image.Image, height: int, width: int) -> torch.Tensor:
    if image.width * image.height > 16_000_000:
        raise ValueError('Image exceeds pixel limit')
    with (
        image.convert('L') as gray,
        ImageOps.contain(gray, (width, height)) as fitted,
        Image.new('L', (width, height), 255) as canvas,
    ):
        canvas.paste(fitted, ((width-fitted.width)//2, (height-fitted.height)//2))
        return torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8).float().view(
            1, height, width) / 127.5 - 1.0


def image_tensor(path: Path, height: int, width: int) -> torch.Tensor:
    if path.stat().st_size > 8_000_000:
        raise ValueError('Image exceeds file size limit')
    with Image.open(path) as image:
        return tensor_from_image(image, height, width)


def prepare(output: Path, count: int = 1200, validation_count: int = 96, seed: int = 208) -> dict:
    if any(type(v) is not int for v in (count, validation_count, seed)):
        raise ValueError('Counts and seed must be integers')
    if not 8 <= count <= 4096 or not 8 <= validation_count <= 256 or not 0 <= seed < 2**31:
        raise ValueError('Bounded dataset size/seed required')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    choices = fonts()
    rng = random.Random(seed)
    seen = set(REAL_RESERVED)
    records = []
    for i in range(count + validation_count):
        split = 'train' if i < count else 'validation'
        category = 'number' if i % 2 == 0 else 'date'
        while True:
            if category == 'number':
                target = f'{rng.randint(10,9999)}/QĐ-{rng.choice(["TCT", "BQP"])}'
            else:
                target = f'{rng.randint(1,28):02}/{rng.randint(1,12):02}/{rng.randint(2000,2029)}'
            if target not in seen:
                seen.add(target)
                break
        prompt = 'Số?' if category == 'number' else 'Ngày?'
        # Label is a copy of the visible value, never a fact about a real legal instrument.
        text = rng.choice(['', 'Số: ']) + target if category == 'number' else 'Ngày: ' + target
        font_index = rng.randrange(len(choices))
        degraded = bool(i % 3)
        render_seed = rng.randrange(2**31)
        path = output / 'images' / f'{i:05}.png'  # Filename carries no answer.
        with render(text, choices[font_index], render_seed, degraded) as image:
            image.save(path)
        records.append({'id': f'generated-{i:05}', 'split': split, 'group': target,
                        'category': category, 'prompt': prompt, 'target': target,
                        'visible_text': text, 'image': path.relative_to(output).as_posix(),
                        'sha256': sha(path), 'font_index': font_index, 'degraded': degraded,
                        'render_seed': render_seed, 'origin': 'synthetic_programmatic',
                        'training_approval': 'experiment_only'})
    manifest = {'schema_version': 2, 'seed': seed, 'scope': 'single_line_field_experiment',
                'production_ready': False, 'independent_benchmark': False,
                'fonts': [{'name': p.name, 'sha256': sha(p)} for p in choices], 'records': records}
    dump(output / 'manifest.json', manifest)
    return manifest


def load(path: Path, height: int, width: int, splits: tuple[str, ...]) -> list[dict]:
    if path.stat().st_size > 6_000_000 or not set(splits) <= {'train', 'validation'}:
        raise ValueError('Invalid manifest size or requested splits')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 2 or manifest.get('production_ready') is not False:
        raise ValueError('Unsupported experiment manifest')
    if not 16 <= len(manifest['records']) <= 4352:
        raise ValueError('Invalid manifest record count')
    seen, groups, hashes, results = set(), {}, {}, []
    for record in manifest['records']:
        if record['id'] in seen or record['split'] not in ('train', 'validation'):
            raise ValueError('Duplicate ID or invalid split')
        seen.add(record['id'])
        if record['origin'] != 'synthetic_programmatic' or record['training_approval'] != 'experiment_only':
            raise ValueError('Only programmatic experiment labels are supported')
        if not isinstance(record['target'], str) or not record['target'] or len(encode(record['target'])) > 32:
            raise ValueError('Invalid target length')
        if record['target'] in REAL_RESERVED:
            raise ValueError('Real development values cannot enter generated training')
        for key, mapping in ((record['target'], groups), (record['sha256'], hashes)):
            if key in mapping and mapping[key] != record['split']:
                raise ValueError('Target/image leakage across splits')
            mapping[key] = record['split']
        if record['split'] not in splits:
            continue  # No evaluation image content loaded into the training cache.
        image = (path.parent / record['image']).resolve()
        if (not image.is_relative_to(path.parent.resolve()) or not image.is_file()
                or image.stat().st_size > 8_000_000 or sha(image) != record['sha256']):
            raise ValueError('Image hash or path mismatch')
        prompt = [1, *encode(record['prompt']), 7]
        answer = [*encode(record['target']), 2]
        if len(prompt) + len(answer) > 64:
            raise ValueError('Sample too long; no silent truncation')
        ctc_length = len(answer)-1 + sum(a == b for a,b in zip(answer[:-2], answer[1:-1]))
        if ctc_length > width//4:
            raise ValueError('CTC alignment length impossible')
        results.append({**record, 'tensor': image_tensor(image, height, width),
                        'ids': prompt+answer, 'labels': [-100]*len(prompt)+answer,
                        'prefix': prompt, 'answer_ids': answer[:-1]})
    if not results:
        raise ValueError('Requested split is empty')
    return results
