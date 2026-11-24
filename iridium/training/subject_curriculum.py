"""Subject-specific specialization followed by joint training in the notebook.

Uses only supplied real labelled items; no tests/evaluators run here. Optimizer
state is fresh for each specialization phase, explicitly recorded in checkpoints.
"""
from dataclasses import replace
from pathlib import Path

from .datasets import Corpus
from .trainer import Trainer
from .losses import LossWeights


def specialize(model, corpus, settings, steps_per_subject, out_dir, device='cpu'):
    if not model.cfg.controller_mode or settings.n_loops < 2:
        raise ValueError('specialization requires controller mode and at least two cycles')
    original = {name: p.requires_grad for name, p in model.named_parameters()}
    records = []
    try:
        for index, subject in enumerate(model.cfg.stacks.specializations):
            items = [item for item in corpus.items if item.sample.meta.get('subject') == subject]
            if not items:
                records.append({'subject': subject, 'status': 'skipped: no labelled data'})
                continue
            prefix = f'bank.stacks.{index}.'
            for name, param in model.named_parameters():
                param.requires_grad_(name.startswith(prefix))
            model.training_subject = subject
            config = replace(settings, steps=steps_per_subject, freeze=(), label='subject-' + subject)
            trainer = Trainer(model, Corpus(items, 'train'), config,
                LossWeights(router_balance=0, router_z=0, loop_kl=0),
                out_dir=Path(out_dir) / subject, device=device)
            trainer.train()
            trainer.save('final', extra={'subject': subject, 'fresh_optimizer': True,
                                         'examples': len(items)})
            records.append({'subject': subject, 'status': 'trained', 'steps': steps_per_subject})
            del trainer
    finally:
        model.training_subject = None
        for name, param in model.named_parameters():
            param.requires_grad_(original[name])
            param.grad = None
    return records
