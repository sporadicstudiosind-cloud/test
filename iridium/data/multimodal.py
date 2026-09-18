"""Explicit, bounded JSONL supervision for interleaved media and tool turns.

Each row: {id, split, source, license, turns:[{role, content:[parts]}]}.
Parts: text; image/video/audio with path; field/geometry with .npy path;
action with payload. Only assistant output is supervised. Tool results are
context. This trains Iridium's own heads, not a wrapper around another model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..codecs.media import FORMAT, audio_span, image_span, video_span
from ..codecs.spans import Sample, Span, text_span
from ..runtime.chat import BOS, EOS, ROLE_TOKEN, _control
from ..training.datasets import Corpus
from ..training.tasks import Item


def media_part(part, cfg, root, supervised):
    kind = part['type']
    if kind == 'text':
        return text_span(part['text'], supervised=supervised, offset=16)
    if kind == 'action':
        payload = np.asarray(part['payload'], dtype=np.float32)
        if payload.ndim != 2 or payload.shape[1] != 1 + cfg.action_scalars or not np.isfinite(payload).all():
            raise ValueError('action payload must be finite [N, 1+action_scalars]')
        if np.any(payload[:, 0] != np.floor(payload[:, 0])) or np.any(payload[:, 0] < 0) or np.any(payload[:, 0] >= cfg.action_ops):
            raise ValueError('action opcode out of range')
        return Span('action', payload, supervised=supervised, observed=not supervised)
    path = (root / part['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('media path must exist inside the manifest directory')
    if kind == 'audio':
        return audio_span(path, cfg, part.get('start', 0), part.get('duration', 4), supervised)
    if kind == 'image' and part.get('native_resolution', False):
        from ..codecs.high_resolution import image_tiles
        return list(image_tiles(path, cfg, part.get('tile_size', 128), supervised=supervised))
    if kind == 'image':
        return image_span(path, cfg, part.get('size', 64), supervised)
    if kind == 'video' and part.get('native_resolution', False):
        from ..codecs.high_resolution import video_tiles
        return list(video_tiles(path, cfg, part.get('start', 0), part.get('duration', 2),
                                part.get('fps', 4), part.get('tile_size', 128), supervised=supervised))
    if kind == 'video':
        return video_span(path, cfg, part.get('start', 0), part.get('duration', 2),
                          part.get('fps', 4), part.get('size', 64), supervised)
    if kind in ('field', 'geometry', 'quantity'):
        values = np.load(path, allow_pickle=False)
        if values.ndim != 2 or values.shape[-1] != cfg.continuous_dims()[kind] or not np.isfinite(values).all():
            raise ValueError('numeric media must be finite [tokens, codec width]')
        return Span(kind, values.astype(np.float32), grid=tuple(part['grid']) if 'grid' in part else None,
                    supervised=supervised, observed=not supervised, atomic=False)
    raise ValueError(f'unsupported media type {kind}')


def load_manifest(path, cfg, split='train', max_items=2000, max_tokens=1024):
    path = Path(path).resolve()
    root = path.parent
    items, seen = [], set()
    with path.open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('split') != split:
                continue
            if not all(row.get(key) for key in ('id', 'license', 'source')):
                raise ValueError(f'line {line_number}: id, source, license required')
            if row['id'] in seen:
                raise ValueError(f'duplicate example id {row["id"]}')
            seen.add(row['id'])
            spans = [_control(BOS)]
            for turn in row['turns']:
                role = turn['role']
                spans.append(_control(ROLE_TOKEN[role]))
                supervised = role == 'assistant'
                for part in turn['content']:
                    value = media_part(part, cfg, root, supervised)
                    spans.extend(value if isinstance(value, list) else [value])
                if supervised:
                    spans.append(_control(EOS, True))
            sample = Sample(spans, meta={'family': row.get('family', 'multimodal'), 'subject': row.get('subject'), 'format': FORMAT,
                                         'source': row['source'], 'license': row['license'], 'id': row['id'],
                                         'revision': row.get('revision'),
                                         'speaker_id': row.get('speaker_id'), 'chapter_id': row.get('chapter_id')})
            if len(sample) > max_tokens:
                raise ValueError(f'{row["id"]}: {len(sample)} tokens > {max_tokens}; explicitly shorten media/text')
            if not any(s.supervised and len(s) for s in spans[1:]):
                raise ValueError('example has no assistant supervision')
            items.append(Item(sample, row.get('family', 'multimodal')))
            if len(items) >= max_items:
                break
    if not items:
        raise ValueError(f'no examples for split={split!r}')
    return Corpus(items, split)
