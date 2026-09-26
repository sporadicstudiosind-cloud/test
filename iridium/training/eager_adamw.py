"""Small eager AdamW implementation for notebook images with broken Dynamo imports.

Deliberately does not subclass torch.optim.Optimizer: even its constructor imports
Dynamo in some PyTorch releases. No compiler monkey patches or fake torch modules.
Only dense, real, fp32/fp64 parameters are supported. AMP keeps master weights fp32.

Two things here were wrong in the obvious version and are worth naming, because
both of them cost real training quality rather than merely being untidy:

* **One parameter group is not enough.** A single group means one weight decay
  for every tensor in the model, and decay is only correct for the ones that
  are matrices. Applied to an RMSNorm gain it pulls the gain toward zero, which
  is a multiplicative shrink on the whole residual stream; applied to an
  embedding row it punishes exactly the rare tokens that were seen least and
  need their vector preserved; applied to a halting bias it drags a calibrated
  stopping prior back to the middle. Llama, GPT-3, Chinchilla and every serious
  recipe since split the parameters and decay only the >=2-D weights. This
  class now takes any number of groups, and :func:`iridium.runtime.memory.
  decay_groups` builds the split.
* **Per-tensor loops are slow where it is least affordable.** A routed model of
  this shape has thousands of small parameter tensors, and a Python loop issuing
  four or five kernels for each of them is launch-bound on a GPU: the card sits
  idle between microscopic kernels. ``torch._foreach_*`` does the same
  arithmetic on a whole list at once. It is enabled by default on CUDA/HIP and
  off on CPU, where the launch overhead it removes does not exist and the extra
  temporaries are a straight loss.

The foreach path and the per-tensor path are checked against each other to
within fp32 round-off in ``tests/unit/test_eager_adamw.py``. They are the same
update, not two approximations of one.
"""
from __future__ import annotations

import torch


class EagerAdamW:
    """AdamW with decoupled weight decay, optional foreach fusion, N groups.

    ``params`` is either an iterable of parameters (one group, as before) or an
    iterable of group dicts in torch.optim style: ``{"params": [...],
    "weight_decay": 0.0, ...}``. Keys a group omits fall back to the constructor
    arguments, so the common case -- "everything as given, except these tensors
    are not decayed" -- is two short dicts.
    """

    def __init__(self, params, lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95),
                 eps=1e-8, foreach=None):
        if lr < 0 or weight_decay < 0 or eps <= 0:
            raise ValueError("invalid AdamW parameters")
        if any(not 0 <= b < 1 for b in betas):
            raise ValueError("betas must lie in [0, 1)")
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        self.param_groups = _build_groups(params, defaults)
        flat = [p for g in self.param_groups for p in g["params"]]
        if not flat:
            raise ValueError("invalid AdamW parameters")
        if any(p.dtype not in (torch.float32, torch.float64) for p in flat):
            raise ValueError("EagerAdamW needs fp32/fp64 master weights; use autocast")
        if len({id(p) for p in flat}) != len(flat):
            raise ValueError("a parameter appears in more than one group")
        # Launch overhead is a GPU problem. On CPU the foreach path only adds
        # list temporaries, so the default follows where the weights live.
        self.foreach = flat[0].is_cuda if foreach is None else bool(foreach)
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
            live = [p for p in group['params'] if p.grad is not None]
            if not live:
                continue
            if self.foreach:
                self._step_group_foreach(group, live)
            else:
                self._step_group_eager(group, live)

    def _step_group_eager(self, group, live):
        b1, b2 = group['betas']
        for p in live:
            s = self.state[p]
            s['step'] += 1
            m, v = s['exp_avg'], s['exp_avg_sq']
            m.lerp_(p.grad, 1 - b1)
            v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            denom = (v.sqrt() / (1 - b2 ** s['step']) ** 0.5).add_(group['eps'])
            p.mul_(1 - group['lr'] * group['weight_decay'])
            p.addcdiv_(m, denom, value=-group['lr'] / (1 - b1 ** s['step']))

    def _step_group_foreach(self, group, live):
        """The same update, issued as list operations rather than per tensor.

        Tensors are bucketed by ``(step, device, dtype)``, and all three parts
        of that key are load-bearing:

        * **step**, because tensors in one group do not always share a step
          count. A parameter whose gradient was None for a while — a frozen
          stage later unfrozen, a modality absent from the early batches — is
          behind, and applying the leader's bias correction to it would silently
          inflate its first real update.
        * **device**, because ``torch._foreach_*`` requires every tensor in one
          call to live on the same device, and this model is placed across
          several GPUs by ``runtime.placement``: the superstacks are distributed
          and the shared components stay on the primary. A single flat list
          would raise on the first multi-GPU run and on no test that fits on one
          card.
        * **dtype**, for the same reason, and because mixing fp32 and fp64
          masters in one call would silently promote.

        In the overwhelmingly common single-device, single-dtype, in-step case
        this is one bucket and the fusion is total.
        """
        b1, b2 = group['betas']
        lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
        buckets: dict[tuple, list] = {}
        for p in live:
            s = self.state[p]
            s['step'] += 1
            buckets.setdefault((s['step'], p.device, p.dtype), []).append(p)
        for (step, _device, _dtype), params in buckets.items():
            grads = [p.grad for p in params]
            exp_avgs = [self.state[p]['exp_avg'] for p in params]
            exp_avg_sqs = [self.state[p]['exp_avg_sq'] for p in params]
            torch._foreach_lerp_(exp_avgs, grads, 1 - b1)
            torch._foreach_mul_(exp_avg_sqs, b2)
            torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - b2)
            denom = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(denom, (1 - b2 ** step) ** 0.5)
            torch._foreach_add_(denom, eps)
            if wd:
                torch._foreach_mul_(params, 1 - lr * wd)
            torch._foreach_addcdiv_(params, exp_avgs, denom,
                                    value=-lr / (1 - b1 ** step))

    def state_dict(self):
        params = [p for g in self.param_groups for p in g['params']]
        indices = {p: i for i, p in enumerate(params)}
        return {'state': {indices[p]: s for p, s in self.state.items()},
                'param_groups': [{**g, 'params': [indices[p] for p in g['params']]}
                                 for g in self.param_groups]}

    def load_state_dict(self, data):
        params = [p for g in self.param_groups for p in g['params']]
        saved = data['param_groups']
        if len(saved) != len(self.param_groups) or sum(
                len(g['params']) for g in saved) != len(params):
            raise ValueError("checkpoint optimizer parameter layout differs")
        for group, stored in zip(self.param_groups, saved):
            if len(stored['params']) != len(group['params']):
                raise ValueError("checkpoint optimizer parameter layout differs")
            group.update({k: v for k, v in stored.items() if k != 'params'})
        self.state = {params[int(i)]: {k: v.to(params[int(i)]) if torch.is_tensor(v) else v
                                      for k, v in s.items()}
                      for i, s in data['state'].items()}


def _build_groups(params, defaults):
    """Accept a parameter iterable or torch.optim-style group dicts."""
    items = list(params)
    if items and isinstance(items[0], dict):
        groups = []
        for raw in items:
            if not isinstance(raw, dict) or 'params' not in raw:
                raise ValueError("every parameter group must be a dict with 'params'")
            group = dict(defaults)
            group.update({k: v for k, v in raw.items() if k != 'params'})
            group['params'] = list(raw['params'])
            groups.append(group)
        return groups
    return [dict(defaults, params=items)]
