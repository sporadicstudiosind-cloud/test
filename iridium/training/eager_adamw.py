"""Small eager AdamW implementation for notebook images with broken Dynamo imports.

Deliberately does not subclass torch.optim.Optimizer: even its constructor imports
Dynamo in some PyTorch releases. No compiler monkey patches or fake torch modules.
Only dense, real, fp32/fp64 parameters are supported. AMP keeps master weights fp32.
"""
from __future__ import annotations

import torch


class EagerAdamW:
    def __init__(self, params, lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-8):
        params = list(params)
        if not params or lr < 0 or weight_decay < 0 or eps <= 0:
            raise ValueError("invalid AdamW parameters")
        if any(not 0 <= b < 1 for b in betas):
            raise ValueError("betas must lie in [0, 1)")
        if any(p.dtype not in (torch.float32, torch.float64) for p in params):
            raise ValueError("EagerAdamW needs fp32/fp64 master weights; use autocast")
        self.param_groups = [dict(params=params, lr=lr, weight_decay=weight_decay,
                                  betas=betas, eps=eps)]
        self.state = {}

    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            for p in group['params']:
                if set_to_none:
                    p.grad = None
                elif p.grad is not None:
                    p.grad.detach_().zero_()

    @torch.no_grad()
    def step(self):
        # Allocate all missing moments BEFORE mutating weights. An allocation
        # failure must never leave half the model updated.
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise ValueError("EagerAdamW requires dense gradients")
                if p not in self.state:
                    self.state[p] = dict(step=0, exp_avg=torch.zeros_like(p),
                                         exp_avg_sq=torch.zeros_like(p))
        for group in self.param_groups:
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                s = self.state[p]
                s['step'] += 1
                m, v = s['exp_avg'], s['exp_avg_sq']
                m.lerp_(p.grad, 1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                denom = (v.sqrt() / (1 - b2 ** s['step']) ** 0.5).add_(group['eps'])
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.addcdiv_(m, denom, value=-group['lr'] / (1 - b1 ** s['step']))

    def state_dict(self):
        params = [p for g in self.param_groups for p in g['params']]
        indices = {p: i for i, p in enumerate(params)}
        return {'state': {indices[p]: s for p, s in self.state.items()},
                'param_groups': [{**g, 'params': [indices[p] for p in g['params']]}
                                 for g in self.param_groups]}

    def load_state_dict(self, data):
        params = [p for g in self.param_groups for p in g['params']]
        if len(data['param_groups']) != 1 or len(data['param_groups'][0]['params']) != len(params):
            raise ValueError("checkpoint optimizer parameter layout differs")
        self.param_groups[0].update({k: v for k, v in data['param_groups'][0].items()
                                     if k != 'params'})
        self.state = {params[int(i)]: {k: v.to(params[int(i)]) if torch.is_tensor(v) else v
                                      for k, v in s.items()}
                      for i, s in data['state'].items()}
