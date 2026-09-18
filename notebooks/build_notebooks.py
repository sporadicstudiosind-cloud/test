"""Generate all four notebooks; this script only writes JSON, never trains/tests."""
from pathlib import Path
import json
import textwrap

HERE = Path(__file__).resolve().parent
TARGETS = {
    'kaggle': 'iridium_studio_kaggle.ipynb',
    'colab': 'iridium_studio.ipynb',
    'jupyter': 'iridium_studio_jupyter.ipynb',
    'colab_legacy': 'train_iridium_colab.ipynb',
}

def cell(kind, text):
    out = {'cell_type': kind, 'metadata': {}, 'source': textwrap.dedent(text).strip().splitlines(True)}
    if kind == 'code':
        out.update(execution_count=None, outputs=[])
    return out

def notebooks(target):
    cells = []
    def md(s): cells.append(cell('markdown', s))
    def code(s): cells.append(cell('code', s))
    md(f'''# Iridium Studio — {target.replace('_legacy', '')}
    Shared core → sparse routed superstacks → refinement → native modality heads.

    **Start a fresh kernel/session after the old `torch._dynamo` failure.**
    This revision defaults to eager AdamW without torch.optim/Dynamo imports,
    fp32 master weights plus automatic GPU mixed precision, conservative batches,
    and resumable optimizer checkpoints. It does not upgrade your installed torch.

    Upload this revision to GitHub before using the clone cell, or attach its ZIP
    and set `PROJECT_ZIP`. Kaggle: enable Internet for GitHub/pip/text datasets and
    select a GPU. Two T4s have separate VRAM; this notebook uses ONE GPU.

    No tests or training were run to validate this revision. The notebook is
    prepared for your run; hardware/runtime/data availability still determine success.
    There are no automatic tests, evaluations, servers or interactive loops in Run All.
    Training only runs when you execute its cell.
    ''')
    md('## 1. Load the updated source')
    code(f'''
    import os, sys, subprocess, zipfile
    from pathlib import Path
    PROJECT_ZIP = ''  # e.g. /kaggle/input/iridium-update/iridium-update.zip
    REPO_URL = 'https://github.com/sporadicstudiosind-cloud/test.git'
    BRANCH = 'claude/gallant-faraday-lhycva'
    BASE = Path('/kaggle/working') if '{target}' == 'kaggle' else Path.cwd()
    ROOT = BASE / 'iridium-updated'
    if 'iridium' in sys.modules:
        raise RuntimeError('Restart the kernel before changing source revisions')
    if not ROOT.exists():
        if PROJECT_ZIP:
            staging = BASE / 'iridium-upload'
            staging.mkdir(exist_ok=True)
            with zipfile.ZipFile(PROJECT_ZIP) as archive:
                for member in archive.infolist():
                    if not (staging / member.filename).resolve().is_relative_to(staging.resolve()):
                        raise ValueError('unsafe archive path')
                archive.extractall(staging)
            candidates = list(staging.rglob('iridium/training/eager_adamw.py'))
            if len(candidates) != 1:
                raise RuntimeError('ZIP must contain exactly one updated repository')
            ROOT = candidates[0].parents[2]
        else:
            subprocess.run(['git', 'clone', '--depth', '1', '--branch', BRANCH, REPO_URL, str(ROOT)], check=True)
    if not (ROOT / 'iridium/training/eager_adamw.py').is_file():
        raise RuntimeError('Old checkout. Upload the update, then use a fresh ROOT directory')
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    # Preserve the host CUDA/PyTorch installation. Optional BNB is NOT installed by default.
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'numpy>=1.24,<3', 'pyyaml>=6,<7', 'psutil>=5,<8',
                    'Pillow>=10,<13', 'datasets>=3,<5'], check=True)
    print('source:', ROOT)
    import hashlib
    SOURCE_DIGEST = hashlib.sha256(b''.join(p.relative_to(ROOT).as_posix().encode() + p.read_bytes()
                                          for p in sorted((ROOT / 'iridium').rglob('*.py')))).hexdigest()
    print('source SHA256:', SOURCE_DIGEST)
    ''')
    md('''## Controller and memory revision
    Each cycle runs the full general core. It chooses core recurrence or subject
    dispatch; returned specialist states are integrated by the next full core cycle.
    Emission selects a completed core state. Fixed unrolling maintains causal caches.
    Consumer presets have shallower cores and deeper specialist stacks.

    A small neural compressor and learned read gate are trained inside the model.
    Normal decoding uses exact local KV. `LongContextSession` bounds KV, retaining
    lossy learned memory across windows, with a 1,048,576-token ingestion budget.
    This is NOT a demonstrated million-token recall capability. Native image tiling
    preserves source pixels; training those capabilities requires matching examples.
    See `docs/ARCHITECTURE_V2.md` for mechanisms, tradeoffs and unvalidated boundaries.
    ''')
    md('## 2. Hardware and training settings')
    code(f'''
    import json, math, torch, psutil
    from dataclasses import replace
    from iridium.config_builder import intelligence_preset
    from iridium.runtime.memory import plan_training
    print('Python:', sys.version.split()[0], 'torch:', torch.__version__, 'torch path:', torch.__file__)
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if '{target}' == 'kaggle' and DEVICE == 'cpu':
        raise RuntimeError('Enable a Kaggle GPU accelerator and restart the session')
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0), 'available GPUs:', torch.cuda.device_count())
    PRESET = 'consumer_tiny'  # consumer, workstation, research_large, frontier_design
    MAX_SEQ_LEN = 1024
    MICRO_BATCH = 1
    EFFECTIVE_BATCH = 16
    STEPS = 3000
    LR = 3e-4
    N_LOOPS = 3           # full core -> optional stacks -> core integration; bounded at 8
    OPTIMIZER = 'eager_adamw'  # native adamw/adafactor/sgd and explicit BNB options remain available
    PRECISION = 'auto'    # T4/P100: fp16+scaler; capable GPUs: bf16; CPU: fp32
    SEED = 0
    RESUME = ''          # trusted full training checkpoint .pt; increase STEPS to continue
    cfg = replace(intelligence_preset(PRESET), max_seq_len=MAX_SEQ_LEN)
    if not 1 <= N_LOOPS <= cfg.router.max_loops:
        raise ValueError('N_LOOPS exceeds configured budget')
    avail = torch.cuda.mem_get_info()[0] if DEVICE.startswith('cuda') else psutil.virtual_memory().available
    plan = plan_training(cfg.n_params, MICRO_BATCH, MAX_SEQ_LEN,
                         cfg.core.n_layers + cfg.stacks.n_layers * cfg.stacks.n_stacks,
                         cfg.core.n_query_heads, avail, optimizer_kind=OPTIMIZER,
                         d_model=max(cfg.core.d_model, cfg.stacks.d_model),
                         d_ff=max(cfg.core.d_ff, cfg.stacks.d_ff), n_loops=N_LOOPS)
    print(cfg.report().render())
    print(plan.render())
    if not plan.fits:
        raise RuntimeError('Conservative budget exceeded. Reduce PRESET or MAX_SEQ_LEN')
    MICRO_BATCH = min(MICRO_BATCH, plan.micro_batch)
    ACCUMULATE = math.ceil(EFFECTIVE_BATCH / MICRO_BATCH)
    print('micro batch:', MICRO_BATCH, 'accumulation:', ACCUMULATE,
          'effective batch:', MICRO_BATCH * ACCUMULATE)
    ''')
    md('''## 3. Training data
    Real text/chat, subject examples and paired speech need Internet. Custom media uses JSONL (see
    `docs/MULTIMODAL_DATA.md`). Text-only data does not train speech, image, video,
    geometry or tool-use competence. `MEDIA_MANIFEST` supports interleaved media,
    assistant outputs, and tool observations; no external AI model is used.
    Start with short speech segments/64px media. Split by source recording, not
    neighboring clips. Dataset attribution is recorded in the checkpoint.
    ''')
    code('''
    from iridium.training.datasets import build_corpus, Corpus, describe
    from iridium.data.multimodal import load_manifest
    from iridium.data.text_corpus import DEFAULT_MIX, licence_notice
    from iridium.data.chat_corpus import chat_licence_notice
    from iridium.codecs.media import FORMAT
    from iridium.data.subject_corpus import real_subject_corpus
    from iridium.data.speech_corpus import prepare_librispeech
    DOWNLOAD_SPEECH = True  # real paired native ASR + speech examples; requires ffmpeg/Internet
    SUBJECT_STEPS = 100
    USE_REAL_TEXT = True
    N_TRAIN_ITEMS = 8000
    MEDIA_MANIFEST = ''  # /kaggle/input/my-media/train.jsonl
    MEDIA_ITEMS = 2000
    TRAIN_MODE = 'mixed'  # mixed | media_only
    if TRAIN_MODE not in ('mixed', 'media_only'):
        raise ValueError('unknown TRAIN_MODE')
    if TRAIN_MODE == 'media_only' and not MEDIA_MANIFEST:
        raise ValueError('media_only requires MEDIA_MANIFEST')
    if TRAIN_MODE == 'media_only':
        train = Corpus([], 'train')
    else:
        mixture = {'channel_depth': .08, 'channel_intervention': .06,
                   'field_rollout': .04, 'scene_goal': .06, 'false_premise': .06}
        if USE_REAL_TEXT:
            mixture.update(chat=.45, text_lm=.25)
        train = build_corpus(N_TRAIN_ITEMS, seed=SEED, split='train', mixture=mixture,
                             text_window=min(256, MAX_SEQ_LEN-8))
        # Reject overlong conversations rather than truncate away the assistant target.
        before = len(train.items)
        train.items = [it for it in train.items if 1 < len(it.sample) <= MAX_SEQ_LEN]
        print('removed overlong examples:', before - len(train.items))
    if MEDIA_MANIFEST:
        media = load_manifest(MEDIA_MANIFEST, cfg.codecs, split='train',
                              max_items=MEDIA_ITEMS, max_tokens=MAX_SEQ_LEN)
        train.items.extend(media.items)
    else:
        print('No custom image/video/tool manifest attached. DOWNLOAD_SPEECH separately controls real speech pairs.')
    SUBJECT_PROVENANCE = []
    if TRAIN_MODE == 'mixed':
        subjects, SUBJECT_PROVENANCE = real_subject_corpus(cfg.stacks.specializations,
                                                          per_subject=256, max_tokens=MAX_SEQ_LEN)
        train.items.extend(subjects.items)
    SPEECH_MANIFEST = ''
    if DOWNLOAD_SPEECH:
        speech_subject = ('audio_speech_music' if 'audio_speech_music' in cfg.stacks.specializations
                          else 'language_reasoning_intent')
        SPEECH_MANIFEST = str(prepare_librispeech(ROOT / 'data/native-speech', cfg.codecs,
                              count=128, max_tokens=MAX_SEQ_LEN, subject=speech_subject))
        speech = load_manifest(SPEECH_MANIFEST, cfg.codecs, max_items=256, max_tokens=MAX_SEQ_LEN)
        train.items.extend(speech.items)
    if not train.items:
        raise ValueError('No usable training examples')
    print(describe(train))
    DATA_INFO = {'media_format': FORMAT, 'source_sha256': SOURCE_DIGEST,
                 'subject_datasets': SUBJECT_PROVENANCE, 'speech_manifest': SPEECH_MANIFEST,
                 'speech_manifest_sha256': hashlib.sha256(Path(SPEECH_MANIFEST).read_bytes()).hexdigest() if SPEECH_MANIFEST else '',
                 'media_manifest_sha256': hashlib.sha256(Path(MEDIA_MANIFEST).read_bytes()).hexdigest() if MEDIA_MANIFEST else '', 'families': train.counts(),
                 'media_manifest': MEDIA_MANIFEST,
                 'text_licences': licence_notice(DEFAULT_MIX) if USE_REAL_TEXT and TRAIN_MODE == 'mixed' else '',
                 'chat_licences': chat_licence_notice() if USE_REAL_TEXT and TRAIN_MODE == 'mixed' else '',
                 'media_examples': sum(it.sample.meta.get('format') == FORMAT for it in train.items),
                 'media_sources': sorted({(it.sample.meta.get('source',''), it.sample.meta.get('license',''))
                                          for it in train.items if it.sample.meta.get('format') == FORMAT})}
    ''')
    md('''## 4. Train and save
    Execute this cell when ready. Keep model weights in fp32; autocast selects
    operation precision. Forward/backward OOM retries discard the whole partial
    update and lower the micro-batch. An optimizer-update OOM stops because an
    interrupted update may have changed some weights. Resume from a checkpoint.
    Checkpoints include optimizer, scaler, step and RNG state; resume starts a
    new data shuffle and is not bit-for-bit replay. Gradient checkpointing stays
    disabled because this architecture's earlier implementation had parity problems.
    ''')
    code('''
    from iridium.model.iridium1 import Iridium1
    from iridium.training.trainer import Trainer, TrainConfig
    from iridium.training.losses import LossWeights
    from iridium.runtime.memory import free_memory
    # Release previous notebook references before allocating a replacement model.
    model = trainer = chat = media_tools = agent = None
    free_memory(verbose=False)
    torch.manual_seed(SEED)
    model = Iridium1(cfg).to(DEVICE)
    settings = TrainConfig(steps=STEPS, batch_size=MICRO_BATCH, accumulate=ACCUMULATE,
                           lr=LR, n_loops=N_LOOPS, optimizer=OPTIMIZER, precision=PRECISION,
                           max_length=MAX_SEQ_LEN, seed=SEED, log_every=25,
                           checkpoint_every=250, label=f'studio-{PRESET}')
    if not RESUME and SUBJECT_STEPS > 0:
        from iridium.training.subject_curriculum import specialize
        warmup = Trainer(model, train, replace(settings, steps=max(1, STEPS//3), label='general-warmup'),
                         LossWeights(), out_dir=ROOT / 'runs/general-warmup', device=DEVICE)
        warmup.data_info = DATA_INFO
        warmup.train()
        warmup.save('final')
        del warmup
        free_memory(verbose=False)
        DATA_INFO['specialization'] = specialize(model, train, settings, SUBJECT_STEPS,
                                                 ROOT / 'runs/subjects', device=DEVICE)
        free_memory(verbose=False)
    # Joint phase unfreezes the shared model. Each phase has a fresh optimizer.
    trainer = Trainer(model, train, settings, LossWeights(),
                      out_dir=ROOT / 'runs/studio', device=DEVICE)
    trainer.data_info = DATA_INFO
    if RESUME:
        trainer.resume(RESUME)
    trainer.train()
    path = trainer.save('final')
    print('Full training checkpoint:', path)
    ''')
    md('''Training includes a general warmup, labelled specialization and joint refinement.
    Real train-split GSM8K/MBPP/ARC data supplies subject routing targets; real LibriSpeech
    pairs supply native speech targets. Downloads pin and record dataset revisions.
    These starter subsets are not sufficient for broad expert or fluent multimodal skills.
    `SUBJECT_STEPS=0` skips warmup/specialization. Resuming skips those phases too.
    ''')
    md('''## 5. Export and optional use
    These cells define helpers only. Call them yourself after training. The
    inference export omits optimizer state and retains fp32 weights to avoid
    mismatched input/parameter dtypes. A trained checkpoint is required for
    meaningful text, native speech recognition, native media generation and
    agent planning; success is not inferred from a falling training loss.
    ''')
    code('''
    model.eval()
    export = ROOT / 'iridium-inference.pt'
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'manifest': trainer.manifest()}, export)
    from iridium.runtime.chat import ChatSession
    chat = ChatSession(model)
    def ask(message):
        return chat.send(message)
    print('Inference checkpoint:', export)
    # ask('Hello!')
    ''')
    code('''
    from iridium.agency.media_agent import MediaTools, MediaAgent
    from iridium.runtime.media_generation import generate_media
    media_tools = MediaTools(model, ROOT / 'media_outputs')
    agent = MediaAgent(model, ROOT / 'agent_outputs')
    # Native transcription + real editing: source seconds 10..20 become clip-relative 0..10.
    # result = media_tools.clip_with_subtitles('/kaggle/input/my-video/source.mp4', 10, 20)
    # Or ask the trained model to choose the sequence of tools:
    # result = agent.run('Extract seconds 10 to 20, transcribe, and subtitle the clip.',
    #                    ['/kaggle/input/my-video/source.mp4'])
    # Native output from Iridium's own trained flow heads:
    # generate_media(model, 'A red cube', 'image', ROOT / 'cube.png', size=64)
    # FFmpeg and ffprobe must be installed for audio/video I/O.
    ''')
    md('''## Reading the result
    The agent returns a real artifact path or an error; it does not claim success
    for a plan alone. Transcription captions use chunk timestamps, not word-level
    alignment. MP4 subtitles are selectable embedded tracks plus a separate SRT.
    See `docs/UPDATE_NOTES.md` for the Kaggle diagnosis, changes, limits and manual
    acceptance checklist. Larger sizes, reliable general intelligence, photorealistic
    synthesis and arbitrary application control are not established by this notebook.
    ''')
    md('''## 6. Optional bounded self-improvement campaign
    This is OFF by default. When you explicitly enable it, the trained current
    version proposes JSON experiments; fresh subprocesses train candidates one
    at a time. A fixed evaluator compares them on selection data and a separate
    promotion audit. A failing candidate never replaces the current version.

    Prepare three disjoint source splits: `train`, `selection`, `audit`, in
    JSONL manifests using the media schema. Add `family` (speech, captions, tools,
    etc.) and at least 16 held-out items per family/modality. No model-written
    Python is executed: code patches are saved for review. This Kaggle backend
    is configuration search, not a security sandbox for arbitrary code.
    ''')
    code('''
    RUN_RESEARCH = False
    RESEARCH_TRAIN = ''
    RESEARCH_SELECTION = ''
    RESEARCH_AUDIT = ''
    # None asks the current model. Weak initial models may fail to emit valid JSON.
    # Or explicitly supply candidates-1 human recipes, e.g.:
    # Controller models must keep gated_bank=False (the core performs integration).
    # PROPOSALS = [{'lr': 1e-4}, {'lr': 2e-4, 'n_loops': 2}]
    PROPOSALS = None
    if RUN_RESEARCH:
        if not all((RESEARCH_TRAIN, RESEARCH_SELECTION, RESEARCH_AUDIT)):
            raise ValueError('Attach all three manifests before enabling research')
        from iridium.research.controller import ResearchLoop
        from iridium.research.policy import ResearchPolicy
        import datetime
        initial_checkpoint = str(path)  # full trained checkpoint saved in section 4
        # Release notebook GPU ownership so each worker has the device to itself.
        model = trainer = chat = media_tools = agent = None
        free_memory(verbose=False)
        policy = ResearchPolicy(rounds=2, candidates=3, train_steps=100,
                                max_parameters=max(150_000_000, int(cfg.n_params * 1.01)))
        campaign = ROOT / 'research_runs' / datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        research = ResearchLoop(campaign, RESEARCH_TRAIN, RESEARCH_SELECTION,
                                RESEARCH_AUDIT, policy=policy, device=DEVICE)
        research.initialize(initial_checkpoint)
        result = research.run(proposals=PROPOSALS)
        print(result)
        # Version directories contain safe tensor weights.pt + model.json.
        # Reload the accepted model explicitly; the live model is never rewritten.
        from iridium.research.worker import load_version
        model, research_descriptor = load_version(result['current'], DEVICE)
        chat = ChatSession(model, n_loops=research_descriptor['n_loops'])
    else:
        print('Research disabled. No candidate training or evaluation was started.')
    ''')
    md('''## Bounded long context and high-resolution inputs
    Definitions only. Call after training. Use one session per document/user; original
    files or SourceArchive remain necessary for exact recovery after compression.
    ''')
    code('''
    from iridium.runtime.long_context import LongContextSession
    from iridium.runtime.source_archive import SourceArchive
    from iridium.codecs.high_resolution import stream_image, stream_video
    from iridium.runtime.chat import Turn, conversation_sample
    def new_long_session():
        model.eval()
        return LongContextSession(model, n_loops=N_LOOPS, window=MAX_SEQ_LEN)
    def ask_long(session, prompt, max_new_tokens=128):
        sample = conversation_sample([Turn('user', prompt)], False, True)
        return session.continue_text(sample, max_new_tokens)
    # session = new_long_session()
    # stream_image(session, '/path/to/large-image.png')
    # print(ask_long(session, 'Read the small labels and explain this diagram.'))
    # print(session.storage_report())
    ''')
    return cells

def main():
    for target, name in TARGETS.items():
        nb = {'cells': notebooks(target), 'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
            'language_info': {'name': 'python'}, 'accelerator': 'GPU'},
            'nbformat': 4, 'nbformat_minor': 0}
        (HERE / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
        print('Wrote', name)

if __name__ == '__main__':
    main()
