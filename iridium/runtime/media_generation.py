"""Export native Iridium image/audio/video outputs using explicit small layouts."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from ..codecs.media import write_audio, write_image, write_video
from ..codecs.spans import Sample, text_span
from .chat import BOS, USER, ASSISTANT, _control
from .generate import generate


def generate_media(model, prompt, modality, path, *, size=64, frames=8, fps=4,
                   audio_seconds=1.0, seed=0, context=()):
    """Output layout is caller-declared; content is predicted by the same core.

    ``context`` can contain observed media spans for conditional generation.
    Train with the same layout declaration in the user's text. This is a native
    low-resolution baseline, with quality entirely dependent on paired training.
    """
    cfg = model.cfg.codecs
    if modality not in ('image', 'audio', 'video'):
        raise ValueError('expected image, audio or video')
    if not isinstance(size, int) or not 8 <= size <= 256 or size % cfg.image_patch:
        raise ValueError('size must be 8..256 and divisible by image_patch')
    if modality == 'image':
        count = (size // cfg.image_patch) ** 2
        layout = f'Output image RGB {size}x{size}.'
    elif modality == 'video':
        if not isinstance(frames, int) or not 1 <= frames <= 64 or frames % cfg.video_patch_t:
            raise ValueError('frames must be 1..64 and divisible by video_patch_t')
        if not np.isfinite(fps) or not 0 < fps <= 30:
            raise ValueError('fps must be in (0,30]')
        count = frames // cfg.video_patch_t * (size // cfg.image_patch) ** 2
        layout = f'Output video RGB {size}x{size}, {frames} frames at {fps} fps.'
    else:
        if not np.isfinite(audio_seconds) or not 0 < audio_seconds <= 8:
            raise ValueError('audio_seconds must be in (0,8]')
        sample_count = round(audio_seconds * 16000)
        width = cfg.audio_mels * cfg.audio_frames
        count = (sample_count + width - 1) // width
        layout = f'Output mono PCM 16000 Hz, {sample_count} samples.'
    if any(s.supervised or not s.observed for s in context):
        raise ValueError('conditioning context must contain unsupervised observed spans')
    sample = Sample([_control(BOS), _control(USER), text_span(prompt + '\n' + layout, False, 16),
                     *context, _control(ASSISTANT)])
    result = generate(model, sample, max_new_tokens=count, allow_continuous=True,
                      force_modality=modality, seed=seed)
    tokens = np.stack([value.reshape(-1) for name, value in result.continuous if name == modality])
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if modality == 'image':
        return write_image(tokens, path, cfg, size)
    if modality == 'video':
        return write_video(tokens, path, cfg, frames, fps, size)
    return write_audio(tokens, path, sample_count)
