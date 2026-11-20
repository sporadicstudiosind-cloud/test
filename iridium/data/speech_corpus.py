"""Prepare real paired speech for native ASR and waveform generation.

Downloads occur only when explicitly called by a notebook user. No external ASR
or TTS network is involved; LibriSpeech supplies the reference transcript.
"""
import hashlib
import json
import math
from pathlib import Path


def prepare_librispeech(directory, cfg, count=128, max_tokens=1024,
                       subject='audio_speech_music', max_scanned=10000):
    from datasets import Audio, load_dataset
    from huggingface_hub import HfApi
    from ..codecs.media import probe, SAMPLE_RATE
    if count < 1 or max_tokens < 32:
        raise ValueError('positive count and useful token budget required')
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    name = 'openslr/librispeech_asr'
    revision = HfApi().dataset_info(name).sha
    data = load_dataset(name, 'all', split='train.clean.100', streaming=True,
                        revision=revision).cast_column('audio', Audio(decode=False))
    records = []
    for scanned, row in enumerate(data):
        if scanned >= max_scanned or len(records) >= count * 2:
            break
        encoded = row['audio'].get('bytes')
        if encoded is None:
            raise ValueError('dataset did not return embedded audio bytes; no silent substitute')
        identity = hashlib.sha256(encoded).hexdigest()
        path = root / (identity + '.flac')
        path.write_bytes(encoded)
        duration = float(probe(path)['format']['duration'])
        transcript = row['text']
        # Keep the COMPLETE utterance and its matching complete transcript.
        tokens = math.ceil(duration * SAMPLE_RATE / (cfg.audio_mels * cfg.audio_frames))
        if not math.isfinite(duration) or not 0 < duration <= 30 or tokens + len(transcript.encode()) + 100 > max_tokens:
            path.unlink()  # only the exact file written above in this output directory
            continue
        base = {'split': 'train', 'source': name + '/' + str(row['id']),
                'license': 'cc-by-4.0', 'revision': revision, 'subject': subject,
                'speaker_id': row['speaker_id'], 'chapter_id': row['chapter_id']}
        media = {'type': 'audio', 'path': path.name, 'duration': duration}
        records.append(dict(base, id=identity+'-asr', family='native_asr', turns=[
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Transcribe this audio.'}, media]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': transcript}]}]))
        records.append(dict(base, id=identity+'-speech', family='native_speech', turns=[
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Read aloud: ' + transcript}]},
            {'role': 'assistant', 'content': [media]}]))
    if not records:
        raise ValueError('no full speech utterances fit; increase token budget')
    manifest = root / 'speech-train.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
    return manifest
