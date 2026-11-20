"""Native-resolution tiles and explicit spatial coordinates; no neural service."""
import numpy as np

from .media import FORMAT
from .spans import Span, Sample, patchify


def image_tiles(path, cfg, tile_size=128, max_pixels=67_108_864, supervised=False):
    """Yield bounded tiles without reducing source resolution (up to 8K square).

    Edge padding is recorded. Coordinates use original image pixel units.
    Consuming all tiles costs proportionally more compute than a thumbnail.
    """
    from PIL import Image, ImageOps
    patch = cfg.image_patch
    if cfg.image_channels != 3 or tile_size % patch or not patch <= tile_size <= 512:
        raise ValueError('tile size must be a patch multiple, at most 512')
    with Image.open(path) as raw:
        if raw.width * raw.height > max_pixels:
            raise ValueError('image exceeds the declared pixel budget')
        image = ImageOps.exif_transpose(raw).convert('RGB')
        width, height = image.size
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                actual_w, actual_h = min(tile_size, width-left), min(tile_size, height-top)
                tile = image.crop((left, top, left + actual_w, top + actual_h))
                # Pad only to patch boundaries, not to an entire empty tile.
                w, h = ((actual_w+patch-1)//patch)*patch, ((actual_h+patch-1)//patch)*patch
                canvas = Image.new('RGB', (w, h)); canvas.paste(tile, (0, 0))
                pixels = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)/127.5-1
                tokens, grid = patchify(pixels, (patch, patch))
                yy, xx = np.meshgrid(np.arange(grid[0]), np.arange(grid[1]), indexing='ij')
                coords = np.stack((np.zeros(xx.size), (top+(yy.ravel()+.5)*patch),
                                   (left+(xx.ravel()+.5)*patch)), -1).astype(np.float32)
                yield Span('image', tokens, grid=grid, atomic=False, supervised=supervised,
                           observed=not supervised, meta={'format': FORMAT,
                           'coordinates': coords, 'source_size': [width, height],
                           'crop': [left, top, actual_w, actual_h]})


def stream_image(session, path, tile_size=128):
    """Feed tile-by-tile without allocating the entire image token sequence."""
    from .spans import collate
    from .bank import TensorBatch
    model = session.model
    device = next(model.parameters()).device
    for tile in image_tiles(path, model.cfg.codecs, tile_size):
        batch = TensorBatch(collate([Sample([tile])], model.cfg.codecs.continuous_dims(),
                                    model.cfg.codecs.action_scalars), device=device)
        session.consume(batch)
    return session.last_hidden


def video_tiles(path, cfg, start=0., duration=2., fps=4., tile_size=128,
                max_pixels=67_108_864, supervised=False):
    """Stream full-resolution spatiotemporal patches in bounded frame groups.

    Explicit fps sampling preserves spatial resolution, not every source frame.
    A group holds at most video_patch_t source frames. This can still use large
    CPU memory for 8K input; pixels and duration have explicit budgets.
    """
    import math
    from .media import probe, ffmpeg
    if not all(math.isfinite(v) for v in (start, duration, fps)) or start < 0 or not 0 < duration <= 30 or not 0 < fps <= 30:
        raise ValueError('invalid video time or sampling budget')
    patch, temporal = cfg.image_patch, cfg.video_patch_t
    if cfg.image_channels != 3 or tile_size % patch or not patch <= tile_size <= 512:
        raise ValueError('invalid RGB tile size')
    streams = [s for s in probe(path)['streams'] if s['codec_type'] == 'video']
    if not streams:
        raise ValueError('no video stream')
    width, height = int(streams[0]['width']), int(streams[0]['height'])
    if width * height > max_pixels:
        raise ValueError('video exceeds pixel budget')
    count = max(1, math.ceil(duration * fps))
    for first in range(0, count, temporal):
        wanted = min(temporal, count - first)
        raw = ffmpeg(['-ss', start + first/fps, '-noautorotate', '-i', path,
                      '-vf', f'fps={fps}', '-frames:v', wanted, '-an',
                      '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1'])
        if not raw:
            break
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)
        real = len(frames)
        if real < temporal:
            frames = np.concatenate((frames, np.repeat(frames[-1:], temporal-real, axis=0)))
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                crop = frames[:, top:top+tile_size, left:left+tile_size]
                h, w = crop.shape[1:3]
                crop = np.pad(crop, ((0, 0), (0, (-h)%patch), (0, (-w)%patch), (0, 0)))
                pixels = crop.astype(np.float32).transpose(3, 0, 1, 2)/127.5-1
                tokens, grid = patchify(pixels, (temporal, patch, patch))
                yy, xx = np.meshgrid(np.arange(grid[1]), np.arange(grid[2]), indexing='ij')
                coords = np.stack((np.full(xx.size, start+first/fps),
                                   top+(yy.ravel()+.5)*patch, left+(xx.ravel()+.5)*patch), -1)
                yield Span('video', tokens, grid=grid, atomic=False, supervised=supervised,
                           observed=not supervised, meta={'format': FORMAT, 'coordinates': coords,
                           'source_size': [width, height], 'fps': fps, 'sampled_frames': real,
                           'rotation_applied': False, 'crop': [left, top, w, h]})


def stream_video(session, path, **options):
    from .spans import collate
    from .bank import TensorBatch
    model = session.model
    for tile in video_tiles(path, model.cfg.codecs, **options):
        batch = TensorBatch(collate([Sample([tile])], model.cfg.codecs.continuous_dims(),
                                    model.cfg.codecs.action_scalars),
                            device=next(model.parameters()).device)
        session.consume(batch)
    return session.last_hidden
