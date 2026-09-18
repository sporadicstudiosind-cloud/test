# Paired media and tool supervision

Place a JSONL manifest next to the files it names. Each row requires `id`,
`split`, `source`, `license` and `turns`. Every media path must resolve inside
the manifest directory. No media is downloaded implicitly. Assign source
recordings/users/scenes to a split BEFORE extracting clips to avoid leakage.

```json
{"id":"speech-001","split":"train","source":"my-recording","license":"owned","turns":[{"role":"user","content":[{"type":"text","text":"Transcribe this audio exactly. Return only the spoken words."},{"type":"audio","path":"speech.wav","start":0,"duration":3}]},{"role":"assistant","content":[{"type":"text","text":"The actual words spoken in this recording."}]}]}
```

Replace the transcript with verified words and the source/license with the true
provenance. The template's text is NOT an actual transcription. Do not train it
against unrelated audio.

Supported content parts:

| Type | Required fields | Optional fields / representation |
| --- | --- | --- |
| text | text | UTF-8 bytes, offset 16 |
| audio | path | start=0, duration=4; mono 16 kHz PCM, normalized [-1,1] |
| image | path | size=64; RGB letterbox; size divisible by image_patch |
| video | path | start=0, duration=2, fps=4, size=64; temporal RGB patches |
| geometry | path | .npy `[tokens, point_features]`, no pickle |
| field | path | .npy `[tokens, field_channels*field_patch**2]`, optional grid |
| quantity | path | .npy `[tokens, 3+quantity_roles]` |
| action | payload | finite `[tokens, 1+action_scalars]`: opcode, operands |

Assistant media parts train generation; user/tool parts train conditioning.
Assistant media spans are non-atomic and unobserved to prevent future-target
leakage. Numeric field units/geometry coordinate frames must be explicit in the
text context or your dataset metadata. This loader does not infer units.

The audio representation is **iridium-rgb-pcm-v1**. Historical config fields
`audio_mels * audio_frames` determine the patch width, but these new adapters put
raw waveform samples in those slots, not mel magnitudes. Do not mix historical
mel datasets and PCM datasets in one run. At default width 256, one second of
audio needs 63 tokens. At image patch 8, a 64x64 image needs 64 tokens; an
8-frame 64x64 video at temporal patch 2 needs 256 tokens. Text/role markers also
consume context. Generation budget must fit in the same context window.

For native image/video/audio generation, use the same user-side layout suffix
as `generate_media`:

```text
A red cube
Output image RGB 64x64.
```

```text
Say hello
Output mono PCM 16000 Hz, 16000 samples.
```

```text
A cube turning
Output video RGB 64x64, 8 frames at 4 fps.
```

Supervise the corresponding assistant media part, followed by EOS (the loader
adds EOS). Supply genuinely paired data across modes, including audio-to-text,
text-to-audio, image/video-to-text, text-to-image/video, and mixed tasks. A head
that only receives input conditioning has not been trained to generate that modality.

For agent training use roles `system`, `user`, `assistant`, `tool`. Copy
`TOOL_PROMPT` from `agency/media_agent.py` into the system text. An assistant
call is text containing `{"tool":"extract_audio","args":{"asset":"asset_0","start":10,"end":20}}`.
The next tool turn contains the actual observation JSON. Subsequent assistant
turns choose the next tool, and finish with `{"final":"asset_4"}` only when the
artifact exists. Include trajectories using the runtime's compact state format
(`outputs` plus `last`), not only full-history demonstrations, since small-context
agent inference conditions on that compact state. Source asset IDs and success
observations must come from real runs. Include invalid-call recovery examples.

Train successful media workflows as well as captions/transcription. General
chat data does not teach this tool schema automatically. Native action opcodes
and JSON media tool calls are separate output interfaces; the action head remains
for the existing scene/UI vocabulary, while file-aware media tools use JSON text.

In the notebook set `MEDIA_MANIFEST`, then use `TRAIN_MODE='mixed'` or
`'media_only'`. The loader caps examples and rejects overlong samples; it does not
silently cut the answer or pool future output tokens. Dependencies are Pillow,
NumPy and system FFmpeg/ffprobe. No pretrained external model is a dependency.


## Research campaigns

For research-loop input, include a `family` label on each row and use disjoint
`train`, `selection`, and `audit` splits. `source` identifies the original recording
or document (not just its licensing category). The controller rejects overlapping
source IDs, example IDs and identical media file content across splits. See
`SELF_IMPROVEMENT.md` for evaluation coverage and promotion criteria.


## Controller revision additions

Rows may include `subject`, matching a configured `stacks.specializations` name.
That label supervises routing during training; the model does not need it during
inference. Dataset `revision`, `speaker_id` and `chapter_id` are preserved.

Image/video parts may use `"native_resolution": true, "tile_size": 128` to retain
source spatial resolution. The manifest loader still rejects examples larger
than `max_tokens`; small-device training should use appropriately sized crops.
Use `stream_image` / `stream_video` with `LongContextSession` for incremental
inference. Observed media coordinates enter the nonlinear modality encoders'
output; output-generation slots retain stream-position conditioning.

`prepare_librispeech` produces complete utterance/transcript pairs for native ASR
and native speech generation; it never substitutes an external speech model.
The notebook enables its bounded download by default. Internet, FFmpeg/ffprobe,
space, source availability and matching training budgets are required.

See `ARCHITECTURE_V2.md` for the complete mechanism and unvalidated boundaries.
