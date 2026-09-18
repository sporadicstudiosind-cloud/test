"""Trusted subprocess worker. Only bounded data/config is accepted, never Python."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..codecs.bank import TensorBatch, continuous_dims
from ..codecs.spans import Sample, collate, text_span
from ..config import IridiumConfig
from ..data.multimodal import load_manifest
from ..model.iridium1 import Iridium1
from ..training.datasets import Corpus
from ..training.losses import LossWeights
from ..training.trainer import TrainConfig, Trainer
from .policy import ResearchPolicy, validate_proposal


def load_version(directory, device='cpu'):
    directory = Path(directory)
    descriptor = json.loads((directory / 'model.json').read_text(encoding='utf-8'))
    cfg = IridiumConfig.from_dict(descriptor['config'])
    model = Iridium1(cfg)
    model.load_state_dict(torch.load(directory / 'weights.pt', map_location='cpu', weights_only=True))
    return model.to(device), descriptor


def save_version(model, directory, loops, provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, directory / 'weights.pt')
    (directory / 'model.json').write_text(json.dumps({'config': model.cfg.to_dict(),
        'n_loops': loops, 'provenance': provenance}, indent=2), encoding='utf-8')


def train_candidate(job):
    policy = ResearchPolicy(**job['policy'])
    torch.manual_seed(job['seed'])
    model, descriptor = load_version(job['parent'])
    proposal = validate_proposal(job['proposal'], job['families'], model.cfg.router.max_loops)
    if model.cfg.controller_mode and proposal.get("gated_bank", False):
        raise ValueError("controller architecture integrates in the core; gated_bank must remain false")
    cfg = replace(model.cfg, **{k: proposal[k] for k in ('qk_norm', 'gated_bank', 'loop_identity') if k in proposal})
    if cfg.n_params > policy.max_parameters:
        raise ValueError('candidate exceeds parameter budget')
    if cfg != model.cfg:
        replacement = Iridium1(cfg)
        missing, unexpected = replacement.load_state_dict(model.state_dict(), strict=False)
        if set(missing + unexpected) - {'bank_gate'}:
            raise ValueError('architecture mutation requires an unsupported weight migration')
        model = replacement
    loops = proposal.get('n_loops', descriptor['n_loops'])
    corpus = load_manifest(job['manifest'], cfg.codecs, split='train',
                           max_items=policy.max_items, max_tokens=cfg.max_seq_len)
    families = corpus.by_family()
    mixture = proposal.get('mixture', {})
    if mixture:
        rng = np.random.default_rng(job['seed'])
        names = sorted(families)
        weights = np.array([mixture.get(name, 1.) for name in names]); weights /= weights.sum()
        # Every candidate receives the same number of training examples/updates.
        corpus = Corpus([families[name][int(rng.integers(len(families[name])))]
                         for name in rng.choice(names, len(corpus), p=weights)], 'train')
    settings = TrainConfig(steps=policy.train_steps, lr=proposal['lr'], weight_decay=proposal['weight_decay'],
        batch_size=policy.micro_batch, accumulate=policy.accumulate, n_loops=loops,
        optimizer='eager_adamw', precision='auto', seed=job['seed'], oom_retry=False,
        label='research', log_every=max(1, policy.train_steps // 10))
    trainer = Trainer(model, corpus, settings, LossWeights(**proposal.get('loss_weights', {})),
                      device=job['device'])
    history = trainer.train()
    save_version(model, job['output'], loops, {'proposal': proposal, 'parent': job['parent'],
                 'steps': policy.train_steps, 'seed': job['seed']})
    (Path(job['output']) / 'training.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    return {'completed_steps': trainer.completed_steps, 'parameters': cfg.n_params}


@torch.no_grad()
def architecture_checks(model, loops, device):
    """Fixed finite-output, accounting, causality and cache-parity regression gates."""
    from ..runtime.decode import run_chunked, slice_batch
    length = min(model.cfg.max_seq_len, max(16, 2 * model.cfg.memory_stride + 7) if model.cfg.memory_slots else 16)
    sample = Sample([text_span(('Causal memory check. ' * (length // 21 + 2))[:length], supervised=False, offset=16)])
    batch = TensorBatch(collate([sample], continuous_dims(model.cfg.codecs)), device=device)
    whole = model(batch, n_loops=loops).hidden
    prefix = model(slice_batch(batch, 0, 6), n_loops=loops).hidden
    incremental = run_chunked(model, batch, chunk=1, n_loops=loops)
    checks = {'finite_hidden': bool(torch.isfinite(whole).all()),
              'parameter_accounting': sum(p.numel() for p in model.parameters()) == model.cfg.n_params,
              'causal_prefix': bool(torch.allclose(whole[:, :6], prefix, atol=2e-4, rtol=2e-3)),
              'cached_decoding': bool(torch.allclose(whole, incremental, atol=2e-4, rtol=2e-3))}
    if not all(checks.values()):
        raise RuntimeError(f'fixed architecture regression checks failed: {checks}')
    return checks


@torch.no_grad()
def evaluate_version(job):
    """Fixed modality losses + free-running exact text; no candidate loss weights."""
    from ..runtime.generate import generate
    policy = ResearchPolicy(**job['policy'])
    model, descriptor = load_version(job['parent'], job['device'])
    model.eval()
    checks = architecture_checks(model, descriptor['n_loops'], job['device'])
    corpus = load_manifest(job['manifest'], model.cfg.codecs, split=job['split'],
                           max_items=policy.eval_items, max_tokens=model.cfg.max_seq_len)
    examples, text_count, correct = [], 0, 0
    for index, item in enumerate(corpus.items):
        torch.manual_seed(policy.seed + index)
        batch = TensorBatch(collate([item.sample], continuous_dims(model.cfg.codecs)), device=job['device'])
        out = model(batch, n_loops=descriptor['n_loops'])
        losses = model.codecs.losses(out.hidden, batch)
        supervised = {s.modality for s in item.sample.spans if s.supervised and len(s)}
        names = (supervised - {'control', 'action'}) | ({'text'} if 'control' in supervised else set())
        if 'action' in supervised:
            names |= {'action_op', 'action_scalar'}
        values = {key: float(losses[key]) for key in sorted(names)}
        examples.append({'id': item.sample.meta['id'], 'family': item.family, 'losses': values})
        # Prefix only: never leak assistant targets/later tool observations into generation.
        prefix = []
        for span in item.sample.spans:
            if span.supervised:
                break
            prefix.append(span)
        targets = [s for s in item.sample.spans if s.supervised]
        if targets and targets[0].modality == 'text' and len(targets[0]) <= 256:
            target = bytes(int(v)-16 for v in targets[0].payload).decode('utf-8', errors='replace')
            budget = min(256, model.cfg.max_seq_len - sum(map(len, prefix)))
            if budget <= len(targets[0]):
                raise ValueError('held-out text has insufficient generation/EOS budget')
            response = generate(model, Sample(prefix), max_new_tokens=budget, text_only=True,
                                n_loops=descriptor['n_loops'], seed=policy.seed + index)
            text_count += 1
            correct += int(response.text.strip() == target.strip() and response.stopped == 'eos')
        del out, losses, batch
    return {'examples': examples, 'text_count': text_count, 'checks': checks,
            'text_accuracy': correct / max(1, text_count), 'parameters': model.cfg.n_params}


def propose(job):
    from ..runtime.chat import Turn, conversation_sample
    from ..runtime.generate import generate
    model, descriptor = load_version(job['parent'], job['device'])
    prompt = ('Propose ONE training experiment as JSON. Keys: hypothesis (short), lr (0.000001..0.001), '
              'weight_decay (0..0.2), qk_norm (bool), gated_bank (bool), loop_identity (bool), '
              'n_loops (1..3). Keep gated_bank=false for controller_mode models. No prose. Current evidence: ' + json.dumps(job['evidence'], separators=(',', ':')))
    sample = conversation_sample([Turn('user', prompt)], False, True)
    budget = min(384, model.cfg.max_seq_len - len(sample))
    if budget < 96:
        raise ValueError('research proposal prompt exceeds model context')
    reply = generate(model, sample, text_only=True, max_new_tokens=budget, temperature=.7,
                     top_p=.9, seed=job['seed'], n_loops=descriptor['n_loops'])
    if reply.stopped != 'eos':
        raise ValueError('research model proposal was truncated')
    return json.loads(reply.text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text(encoding='utf-8'))
    started = time.monotonic()
    result = {'train': train_candidate, 'evaluate': evaluate_version, 'propose': propose}[job['kind']](job)
    Path(job['result']).write_text(json.dumps({'result': result, 'seconds': time.monotonic()-started},
                                             allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
