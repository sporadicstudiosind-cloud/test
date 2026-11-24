"""Versioned raw-media adapters for the single Iridium model; no external models.

RGB pixels and mono PCM are normalized to [-1, 1]. PCM patches are invertible;
unlike mel magnitudes they do not require a separately trained vocoder. This is
a small-model baseline, not a pretrained perceptual encoder or neural codec.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from .spans import Span, patchify, unpatchify

FORMAT = "iridium-rgb-pcm-v1"
SAMPLE_RATE = 16000


def ffmpeg(args, timeout=180):
    exe = shutil.which('ffmpeg')
    if not exe:
        raise RuntimeError('ffmpeg is required for media I/O; install the system executable')
    result = subprocess.run([exe, '-hide_banner', '-loglevel', 'error', '-nostdin',
                             *map(str, args)], capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace')[-3000:])
    return result.stdout


def probe(path):
    exe = shutil.which('ffprobe')
    if not exe:
        raise RuntimeError('ffprobe is required (distributed with ffmpeg)')
    result = subprocess.run([exe, '-v', 'error', '-show_format', '-show_streams',
                             '-of', 'json', str(Path(path).resolve())],
                            capture_output=True, text=True, check=True, timeout=30)
    return json.loads(result.stdout)


def audio_span(path, cfg, start=0.0, duration=4.0, supervised=False):
    if not math.isfinite(start) or not math.isfinite(duration) or start < 0 or not 0 < duration <= 30:
        raise ValueError('audio start >= 0 and 0 < duration <= 30 seconds required')
    raw = ffmpeg(['-ss', start, '-i', Path(path).resolve(), '-t', duration,
                  '-vn', '-ac', 1, '-ar', SAMPLE_RATE, '-f', 'f32le', 'pipe:1'])
    pcm = np.frombuffer(raw, dtype='<f4').copy()
    if not pcm.size or not np.isfinite(pcm).all():
        raise ValueError('audio selection is empty or non-finite')
    width = cfg.audio_mels * cfg.audio_frames  # historical field names, fixed tensor width
    padded = np.pad(np.clip(pcm, -1, 1), (0, (-pcm.size) % width))
    return Span('audio', padded.reshape(-1, width), supervised=supervised,
                observed=not supervised, atomic=False,
                meta={'format': FORMAT, 'sample_rate': SAMPLE_RATE,
                      'samples': pcm.size, 'start': start,
                      'coordinates': np.stack((start + np.arange(len(padded)//width) * width / SAMPLE_RATE,
                                               np.zeros(len(padded)//width), np.zeros(len(padded)//width)), -1)})


def write_audio(tokens, path, sample_count=None):
    pcm = np.asarray(tokens, dtype=np.float32).reshape(-1)
    if not np.isfinite(pcm).all() or not pcm.size:
        raise ValueError('audio must be nonempty and finite')
    if sample_count is not None:
        if not 0 < sample_count <= pcm.size:
            raise ValueError('invalid sample count')
        pcm = pcm[:sample_count]
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes((np.clip(pcm, -1, 1) * 32767).astype('<i2').tobytes())
    return Path(path)


def image_span(path, cfg, size=64, supervised=False):
    from PIL import Image, ImageOps
    if cfg.image_channels != 3 or size < cfg.image_patch or size > 512 or size % cfg.image_patch:
        raise ValueError('RGB size must be <= 512 and divisible by image_patch')
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        # Letterbox preserves aspect ratio; training and generation use this layout.
        image = ImageOps.pad(image, (size, size), color=(0, 0, 0))
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1
    tokens, grid = patchify(pixels, (cfg.image_patch, cfg.image_patch))
    return Span('image', tokens, grid=grid, supervised=supervised,
                observed=not supervised, atomic=False,
                meta={'format': FORMAT, 'size': size, 'coordinates': np.stack((
                    np.zeros(len(tokens)), (np.arange(len(tokens))//grid[1]+.5)*cfg.image_patch,
                    (np.arange(len(tokens))%grid[1]+.5)*cfg.image_patch), -1)})


def write_image(tokens, path, cfg, size=64):
    from PIL import Image
    grid = (size // cfg.image_patch,) * 2
    pixels = unpatchify(np.asarray(tokens), grid, 3, (cfg.image_patch,) * 2)
    if not np.isfinite(pixels).all():
        raise ValueError('non-finite pixels')
    Image.fromarray(np.clip((pixels.transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype('uint8')).save(path)
    return Path(path)


def video_span(path, cfg, start=0.0, duration=2.0, fps=4, size=64, supervised=False):
    if cfg.image_channels != 3 or size < cfg.image_patch or size > 256 or size % cfg.image_patch:
        raise ValueError('RGB video size must divide into image patches and be <= 256')
    if not all(math.isfinite(v) for v in (start, duration, fps)) or start < 0 or not 0 < duration <= 10 or not 0 < fps <= 30:
        raise ValueError('invalid video time range/fps')
    filters = (f'fps={fps},scale={size}:{size}:force_original_aspect_ratio=decrease,'
               f'pad={size}:{size}:(ow-iw)/2:(oh-ih)/2')
    raw = ffmpeg(['-ss', start, '-i', Path(path).resolve(), '-t', duration,
                  '-vf', filters, '-an', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1'])
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, size, size, 3)
    if not len(frames):
        raise ValueError('empty video selection')
    count = len(frames)
    if count % cfg.video_patch_t:
        frames = np.concatenate([frames, np.repeat(frames[-1:], (-count) % cfg.video_patch_t, axis=0)])
    pixels = frames.astype(np.float32).transpose(3, 0, 1, 2) / 127.5 - 1
    tokens, grid = patchify(pixels, (cfg.video_patch_t, cfg.image_patch, cfg.image_patch))
    return Span('video', tokens, grid=grid, supervised=supervised,
                observed=not supervised, atomic=False,
                meta={'format': FORMAT, 'fps': fps, 'frames': count, 'size': size, 'start': start})


def write_video(tokens, path, cfg, frames=8, fps=4, size=64):
    if size % 2 or size % cfg.image_patch or frames % cfg.video_patch_t or fps <= 0:
        raise ValueError('video output layout must divide into patches (and even pixels)')
    grid = (frames // cfg.video_patch_t, size // cfg.image_patch, size // cfg.image_patch)
    pixels = unpatchify(np.asarray(tokens), grid, 3, (cfg.video_patch_t, cfg.image_patch, cfg.image_patch))
    if not np.isfinite(pixels).all():
        raise ValueError('non-finite video pixels')
    raw = np.clip((pixels.transpose(1, 2, 3, 0) + 1) * 127.5, 0, 255).astype('uint8')
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / 'frames.rgb'
        source.write_bytes(raw.tobytes())
        ffmpeg(['-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{size}x{size}',
                '-r', fps, '-i', source, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', Path(path).resolve()])
    return Path(path)
