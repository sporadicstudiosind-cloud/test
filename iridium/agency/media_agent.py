"""Bounded Iridium tool loop and an executable clip/transcribe/subtitle workflow.

The policy and ASR both use the supplied Iridium checkpoint. ffmpeg is a media
utility, not another model. Tool calls are JSON with explicit argument schemas;
model text is never evaluated as Python, a shell command, or an ffmpeg filter.
"""
from __future__ import annotations

import inspect
import json
import math
import subprocess
from pathlib import Path

from ..codecs.media import FORMAT, audio_span, image_span, video_span, ffmpeg, probe
from ..codecs.spans import Sample, text_span
from ..runtime.chat import ASSISTANT, BOS, EOS, ROLE_TOKEN, STOP_IDS, _control
from ..runtime.generate import generate


def _finite_time(start, end):
    if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in (start, end)):
        raise ValueError('timestamps must be finite numbers')
    if start < 0 or end <= start or end - start > 120:
        raise ValueError('require 0 <= start < end and duration <= 120 seconds')


def _stamp(seconds):
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f'{hours:02}:{minutes:02}:{seconds:02},{ms:03}'


class MediaTools:
    def __init__(self, model, workspace):
        self.model = model
        self.root = Path(workspace).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts = {}
        self.counter = 0

    def register(self, path):
        """The caller explicitly grants read access to one source file."""
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        key = f'asset_{len(self.artifacts)}'
        self.artifacts[key] = path
        return key

    def _source(self, asset):
        if asset not in self.artifacts:
            raise ValueError(f'unknown asset {asset!r}; use an asset returned by a tool')
        return self.artifacts[asset]

    def _output(self, suffix):
        self.counter += 1
        path = self.root / f'media_{self.counter:04}{suffix}'
        while path.exists():
            self.counter += 1
            path = self.root / f'media_{self.counter:04}{suffix}'
        return path

    def _result(self, path, **extra):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError('media tool did not produce an artifact')
        return {'asset': self.register(path), 'path': str(path), **extra}

    def inspect_media(self, asset):
        data = probe(self._source(asset))
        return {'asset': asset, 'duration': data.get('format', {}).get('duration'),
                'streams': [{k: stream.get(k) for k in
                             ('codec_type', 'codec_name', 'width', 'height', 'sample_rate')}
                            for stream in data.get('streams', [])]}

    def describe(self, asset, modality='image'):
        source, cfg = self._source(asset), self.model.cfg.codecs
        if modality == 'image':
            media = image_span(source, cfg, size=32)
        elif modality == 'video':
            media = video_span(source, cfg, duration=1, fps=2, size=32)
        else:
            raise ValueError('describe accepts image or video; use transcribe for speech')
        sample = Sample([_control(BOS), _control(ROLE_TOKEN['user']),
                         text_span('Describe the supplied media.', False, 16), media, _control(ASSISTANT)])
        response = generate(self.model, sample, text_only=True, max_new_tokens=128, stop_ids=STOP_IDS)
        if response.stopped != 'eos' or not response.text.strip():
            raise RuntimeError('native description empty/truncated; paired caption training required')
        return {'asset': asset, 'description': response.text, 'modality': modality}

    def extract_audio(self, asset, start, end):
        _finite_time(start, end)
        path = self._output('.wav')
        ffmpeg(['-n', '-ss', start, '-i', self._source(asset), '-t', end-start,
                '-vn', '-ar', 16000, '-ac', 1, '-c:a', 'pcm_s16le', path])
        return self._result(path, start=start, duration=end-start)

    def trim_video(self, asset, start, end):
        _finite_time(start, end)
        path = self._output('.mp4')
        ffmpeg(['-n', '-ss', start, '-i', self._source(asset), '-t', end-start,
                '-map', '0:v:0', '-map', '0:a?', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                '-c:v', 'libx264', '-preset', 'veryfast', '-c:a', 'aac',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path])
        return self._result(path, duration=end-start)

    def transcribe(self, asset, chunk_seconds=3.0, max_new_tokens=160):
        """Native audio->text; timestamps are chunk bounds, NOT word alignment."""
        if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 512:
            raise ValueError('max_new_tokens must be 1..512')
        if not isinstance(chunk_seconds, (int, float)) or not math.isfinite(chunk_seconds) or not 0.25 <= chunk_seconds <= 8:
            raise ValueError('chunk_seconds must be 0.25..8')
        source = self._source(asset)
        duration = float(probe(source)['format']['duration'])
        if not math.isfinite(duration) or not 0 < duration <= 120:
            raise ValueError('clip audio to <= 120 seconds first')
        cfg = self.model.cfg
        prompt = 'Transcribe this audio exactly. Return only the spoken words.'
        available = cfg.max_seq_len - max_new_tokens - len(prompt.encode()) - 8
        chunk_seconds = min(chunk_seconds, available * cfg.codecs.audio_mels * cfg.codecs.audio_frames / 16000)
        if chunk_seconds < 0.25:
            raise ValueError('context budget too small for audio transcription')
        segments = []
        for index in range(math.ceil(duration / chunk_seconds)):
            start = index * chunk_seconds
            end = min(duration, start + chunk_seconds)
            audio = audio_span(source, cfg.codecs, start, end-start)
            sample = Sample([_control(BOS), _control(ROLE_TOKEN['user']),
                             text_span(prompt, False, 16), audio, _control(ASSISTANT)])
            result = generate(self.model, sample, max_new_tokens=max_new_tokens,
                              text_only=True, stop_ids=STOP_IDS)
            if not result.text.strip() or result.stopped != 'eos':
                raise RuntimeError('native ASR returned empty/truncated text; train paired speech/text data first')
            segments.append({'start': start, 'end': end, 'text': result.text.strip()})
        path = self._output('.json')
        path.write_text(json.dumps({'segments': segments, 'timing': 'chunk bounds',
                                    'source': asset, 'format': FORMAT}, indent=2), encoding='utf-8')
        return self._result(path, segments=segments, timing='chunk bounds; transcription quality is unverified')

    def subtitles(self, transcript):
        data = json.loads(self._source(transcript).read_text(encoding='utf-8'))
        previous, lines = 0.0, []
        for i, segment in enumerate(data['segments'], 1):
            start, end, text = segment['start'], segment['end'], segment['text']
            _finite_time(start, end)
            if start < previous or not isinstance(text, str) or not text.strip():
                raise ValueError('subtitle segments must be ordered, nonoverlapping and nonempty')
            # Strip markup to avoid interpreted subtitle tags; retain literal dialogue.
            text = text.replace('<', '').replace('>', '').replace('\r', ' ').replace('\n', ' ')
            lines.append(f'{i}\n{_stamp(start)} --> {_stamp(end)}\n{text}\n')
            previous = end
        if not lines:
            raise ValueError('empty transcript')
        path = self._output('.srt')
        path.write_text('\n'.join(lines), encoding='utf-8')
        return self._result(path)

    def compose(self, video, audio, subtitles):
        """Stitch replacement audio and an embedded, selectable MP4 subtitle track."""
        for asset in (video, audio):
            duration = float(probe(self._source(asset))['format']['duration'])
            if not math.isfinite(duration) or not 0 < duration <= 120:
                raise ValueError('composition inputs must be clips of <= 120 seconds')
        path = self._output('.mp4')
        ffmpeg(['-n', '-i', self._source(video), '-i', self._source(audio),
                '-i', self._source(subtitles), '-map', '0:v:0', '-map', '1:a:0',
                '-map', '2:s:0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                '-c:s', 'mov_text', '-disposition:s:0', 'default',
                '-movflags', '+faststart', '-shortest', path])
        return self._result(path, subtitle_mode='embedded selectable track')

    def concatenate(self, assets):
        """Normalize clips and concatenate video+audio using a fixed filter graph."""
        if not isinstance(assets, list) or not 2 <= len(assets) <= 8:
            raise ValueError('provide 2..8 clips with video and audio')
        args, graph, joins = ['-n'], [], []
        for i, asset in enumerate(assets):
            info = probe(self._source(asset))
            duration = float(info['format']['duration'])
            if not math.isfinite(duration) or not 0 < duration <= 120:
                raise ValueError('each input clip must be <= 120 seconds')
            kinds = {stream['codec_type'] for stream in info['streams']}
            if not {'video', 'audio'} <= kinds:
                raise ValueError('concatenation requires both video and audio in every clip')
            args += ['-i', self._source(asset)]
            graph += [f'[{i}:v]scale=640:360:force_original_aspect_ratio=decrease,'
                      f'pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,setpts=PTS-STARTPTS[v{i}]',
                      f'[{i}:a]aresample=48000,aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS[a{i}]']
            joins.append(f'[v{i}][a{i}]')
        graph.append(''.join(joins) + f'concat=n={len(assets)}:v=1:a=1[v][a]')
        path = self._output('.mp4')
        ffmpeg(args + ['-filter_complex', ';'.join(graph), '-map', '[v]', '-map', '[a]',
                       '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', path], timeout=600)
        return self._result(path, note='subtitle tracks are not concatenated; apply subtitles after stitching')

    def dispatch(self, call):
        if not isinstance(call, dict) or set(call) != {'tool', 'args'} or not isinstance(call['args'], dict):
            raise ValueError('expected exactly {"tool": name, "args": {...}}')
        allowed = ('inspect_media', 'describe', 'extract_audio', 'trim_video', 'transcribe',
                   'subtitles', 'compose', 'concatenate')
        if call['tool'] not in allowed:
            raise ValueError('unknown tool')
        fn = getattr(self, call['tool'])
        inspect.signature(fn).bind(**call['args'])
        return fn(**call['args'])

    def clip_with_subtitles(self, source, start, end):
        """Concrete requested workflow; ASR still requires a trained checkpoint."""
        asset = self.register(source)
        audio = self.extract_audio(asset, start, end)
        video = self.trim_video(asset, start, end)
        transcript = self.transcribe(audio['asset'])
        captions = self.subtitles(transcript['asset'])
        final = self.compose(video['asset'], audio['asset'], captions['asset'])
        return {'video': final, 'audio': audio, 'transcript': transcript, 'subtitles': captions}


TOOL_PROMPT = '\n'.join([
    'Reply with one JSON: {"tool":name,"args":{}} or {"final":asset_id}.',
    'Tools: inspect_media(asset), describe(asset,modality), extract_audio(asset,start,end),',
    'trim_video(asset,start,end), transcribe(asset), subtitles(transcript),',
    'compose(video,audio,subtitles), concatenate(assets). Times in seconds.',
    'Use registered asset IDs. Finish with a produced artifact. Wait for tool results.'
])



class MediaAgent:
    def __init__(self, model, workspace, max_steps=12):
        if not 1 <= max_steps <= 24:
            raise ValueError('max_steps must be 1..24')
        self.tools = MediaTools(model, workspace)
        self.model, self.max_steps = model, max_steps

    def run(self, request, sources):
        assets = [self.tools.register(path) for path in sources]
        spans = [_control(BOS), _control(ROLE_TOKEN['system']), text_span(TOOL_PROMPT, False, 16),
                 _control(ROLE_TOKEN['user']), text_span(request + '\nAssets: ' + json.dumps(assets), False, 16)]
        base_spans = list(spans)
        trace, produced = [], set()
        observations = []
        for _ in range(self.max_steps):
            # Keep the request/system plus bounded state; the complete audit is on disk.
            state = json.dumps({'outputs': observations, 'last': {k: v for k, v in trace[-1]['tool'].items() if k in ('asset', 'error', 'description')} if trace else {}}, separators=(',', ':'))
            spans = base_spans + [_control(ROLE_TOKEN['tool']), text_span(state, False, 16)]
            sample = Sample(spans + [_control(ASSISTANT)])
            budget = min(256, self.model.cfg.max_seq_len - len(sample))
            if budget < 32:
                raise RuntimeError('agent context exhausted; no completion claimed')
            response = generate(self.model, sample, max_new_tokens=budget,
                                text_only=True, stop_ids=STOP_IDS)
            spans += [_control(ASSISTANT), text_span(response.text, False, 16), _control(EOS)]
            try:
                if response.stopped != 'eos':
                    raise ValueError('truncated tool call')
                call = json.loads(response.text)
                if isinstance(call, dict) and set(call) == {'final'}:
                    if call['final'] not in produced:
                        raise ValueError('final must name an artifact actually produced by a tool')
                    path = self.tools._source(call['final'])
                    if not path.is_file():
                        raise ValueError('final artifact no longer exists')
                    return {'asset': call['final'], 'path': str(path), 'trace': trace}
                result = self.tools.dispatch(call)
                if 'asset' in result and 'path' in result:
                    produced.add(result['asset'])
                    observations.append([call['tool'], result['asset']])
            except (ValueError, TypeError, KeyError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
                result = {'error': str(exc)[:500]}
            trace.append({'assistant': response.text, 'tool': result})
            (self.tools.root / 'trace.json').write_text(json.dumps(trace, indent=2), encoding='utf-8')
            # Compact results keep full transcripts in artifacts, not in the policy context.
            compact = {k: v for k, v in result.items() if k not in ('segments', 'path')}
            spans += [_control(ROLE_TOKEN['tool']), text_span(json.dumps(compact), False, 16)]
        raise RuntimeError('agent step budget exhausted; see trace.json; no completion claimed')
