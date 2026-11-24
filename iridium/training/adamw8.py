"""AdamW with 8-bit optimizer state, in plain PyTorch (runs on CUDA, ROCm, XLA, CPU).

fp32 AdamW keeps two fp32 moments per parameter: 8 of the ~16 bytes/param a
training run needs, and the reason a 1B model's state alone fills a free 16 GB
device. Storing the moments in 8 bits with a per-block scale (Dettmers et al.,
2022, "8-bit Optimizers via Block-wise Quantization") cuts them to ~2.03
bytes/param. ``bitsandbytes`` implements that with CUDA kernels only; this is
the same idea in portable tensor ops, slower per step but available on every
backend this project targets.

Per block of ``block`` values:

* first moment ``m``: int8 of ``sign(m) * sqrt(|m| / absmax)``;
* second moment ``v``: uint8 of ``(v / max)^(1/4)``.

The roots are companding, a cheap stand-in for bitsandbytes' dynamic
quantization map: they spend the 8 bits on relative rather than absolute
precision, so small entries in a block with one large one survive instead of
rounding to zero. Linear 8-bit was tried first and stalls (the test records
it): a second moment rounded to zero turns the update into ``m / eps``.

Rounding is stochastic-free but *unbiased on average* is not guaranteed, so
two safeguards: tensors smaller than ``min_8bit_size`` (norms, biases,
gates -- the parameters most sensitive to state noise) keep fp32 state, and
the update is computed in fp32 from the dequantized moments every step.
"""

from __future__ import annotations

import math

import torch

__all__ = ["AdamW8bit"]


def _quant_signed(x: torch.Tensor, block: int):
    flat = x.reshape(-1)
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.view(-1, block)
    scale = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    # Companding: store sign * sqrt(|x| / scale). Linear int8 zeroes every
    # value below scale/254, which is most of a block late in training.
    q = torch.round(torch.sign(blocks) * (blocks.abs() / scale).sqrt() * 127).to(torch.int8)
    return q, scale.squeeze(1)


def _dequant_signed(q, scale, shape, n):
    r = q.float() / 127
    return (torch.sign(r) * r.square() * scale.unsqueeze(1)).reshape(-1)[:n].view(shape)


def _quant_sqrt(v: torch.Tensor, block: int):
    # v^(1/4): a second moment spans many decades, and a zeroed v divides the
    # update by eps -- the one quantization error that is catastrophic.
    flat = v.reshape(-1).clamp_min(0).sqrt().sqrt()
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.view(-1, block)
    scale = blocks.amax(dim=1, keepdim=True).clamp_min(1e-30)
    q = torch.round(blocks / scale * 255).to(torch.uint8)
    return q, scale.squeeze(1)


def _dequant_sqrt(q, scale, shape, n):
    r = q.float() * (scale.unsqueeze(1) / 255)
    return r.square().square().reshape(-1)[:n].view(shape)


class AdamW8bit(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 3e-4, betas=(0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.01, block: int = 256, min_8bit_size: int = 4096):
        if lr <= 0 or not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("invalid AdamW8bit hyperparameters")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      block=block, min_8bit_size=min_8bit_size))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd, block = group["lr"], group["eps"], group["weight_decay"], group["block"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                small = p.numel() < group["min_8bit_size"]
                if not state:
                    state["step"] = 0
                    if small:
                        state["m"] = torch.zeros_like(p, dtype=torch.float32)
                        state["v"] = torch.zeros_like(p, dtype=torch.float32)
                    else:
                        z = torch.zeros_like(p, dtype=torch.float32)
                        state["m_q"], state["m_s"] = _quant_signed(z, block)
                        state["v_q"], state["v_s"] = _quant_sqrt(z, block)
                state["step"] += 1
                t = state["step"]
                if small:
                    m, v = state["m"], state["v"]
                else:
                    m = _dequant_signed(state["m_q"], state["m_s"], p.shape, p.numel())
                    v = _dequant_sqrt(state["v_q"], state["v_s"], p.shape, p.numel())
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                if not small:
                    state["m_q"], state["m_s"] = _quant_signed(m, block)
                    state["v_q"], state["v_s"] = _quant_sqrt(v, block)
                bc1 = 1 - b1 ** t
                bc2 = 1 - b2 ** t
                update = (m / bc1) / ((v / bc2).sqrt() + eps)
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(update.to(p.dtype), alpha=-lr)
        return loss

    @staticmethod
    def state_bytes_per_param(block: int = 256) -> float:
        """Two int8/uint8 moments plus two fp32 scales per block."""
        return 2 + 8 / block
