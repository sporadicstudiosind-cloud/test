"""Output heads: one model, several emission modes.

Discrete modalities (text, action opcodes) are ordinary softmax heads.
Continuous modalities (image, video, audio, field, geometry) use **conditional
flow matching**: the head learns a velocity field ``v(x_tau, tau, h)`` whose
integral transports noise to data. Regression to the mean is available and
tested against, because it is the honest baseline — it is what a plain MSE head
gives you, and it is blurry for a reason worth being explicit about.

The noise time ``tau`` is a *generative* coordinate and is kept strictly
separate from physical time, simulation time, arrival time and sequence
position. ``iridium/contracts/frames.py`` establishes that separation; a head
that conditions a fluid state on a denoising schedule has conflated two of
them.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm


class FlowMatchingHead(nn.Module):
    """Predicts the conditional flow velocity for one continuous modality.

    Training: sample ``tau ~ U(0,1)`` and ``x0 ~ N(0, I)``; form
    ``x_tau = (1-tau) x0 + tau x1``; regress ``v_theta(x_tau, tau, h)`` onto the
    conditional velocity ``x1 - x0``. That target is the standard rectified /
    conditional flow, and its expectation is the marginal velocity field.

    Sampling: integrate ``dx/dtau = v`` from ``x0`` with an explicit Euler or
    midpoint rule. Step count is a measured quantity, never a promise; see
    ``examples/`` for the measured quality-versus-steps curve.
    """

    def __init__(self, d_model: int, d_out: int, d_hidden: int = 0, n_tau: int = 64) -> None:
        super().__init__()
        d_hidden = d_hidden or max(d_out * 2, d_model)
        self.d_out = d_out
        self.n_tau = n_tau
        self.norm = RMSNorm(d_model)
        self.tau_embed = nn.Linear(2 * n_tau, d_hidden)
        self.x_in = nn.Linear(d_out, d_hidden)
        self.cond = nn.Linear(d_model, d_hidden)
        self.mix = nn.Sequential(
            nn.SiLU(), nn.Linear(d_hidden, d_hidden), nn.SiLU()
        )
        self.out = nn.Linear(d_hidden, d_out)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _tau_features(self, tau: torch.Tensor) -> torch.Tensor:
        freqs = torch.linspace(0, 8, self.n_tau, device=tau.device, dtype=torch.float32)
        ang = tau.float().unsqueeze(-1) * (2.0 ** freqs) * torch.pi
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    def velocity(
        self, h: torch.Tensor, x_tau: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        z = (
            self.cond(self.norm(h))
            + self.x_in(x_tau)
            + self.tau_embed(self._tau_features(tau))
        )
        return self.out(self.mix(z))

    def loss(
        self,
        h: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        x0 = torch.randn(target.shape, device=target.device, dtype=target.dtype,
                         generator=generator)
        tau = torch.rand(target.shape[:-1], device=target.device, dtype=target.dtype,
                         generator=generator)
        x_tau = (1.0 - tau.unsqueeze(-1)) * x0 + tau.unsqueeze(-1) * target
        v_target = target - x0
        v_pred = self.velocity(h, x_tau, tau)
        err = (v_pred.float() - v_target.float()).pow(2).mean(-1)
        if reduction == "none":
            return err
        if mask is not None:
            denom = mask.sum().clamp_min(1)
            return (err * mask).sum() / denom
        return err.mean()

    @torch.no_grad()
    def sample(
        self,
        h: torch.Tensor,
        steps: int = 16,
        generator: Optional[torch.Generator] = None,
        midpoint: bool = True,
    ) -> torch.Tensor:
        shape = h.shape[:-1] + (self.d_out,)
        x = torch.randn(shape, device=h.device, dtype=h.dtype, generator=generator)
        dt = 1.0 / steps
        for i in range(steps):
            tau = torch.full(h.shape[:-1], i * dt, device=h.device, dtype=h.dtype)
            if midpoint:
                v1 = self.velocity(h, x, tau)
                x_mid = x + 0.5 * dt * v1
                v = self.velocity(h, x_mid, tau + 0.5 * dt)
            else:
                v = self.velocity(h, x, tau)
            x = x + dt * v
        return x


class RegressionHead(nn.Module):
    """Deterministic continuous head. The blurry-but-honest baseline."""

    def __init__(self, d_model: int, d_out: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, d_out)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(h))

    def loss(
        self, h: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        err = (self.forward(h).float() - target.float()).pow(2).mean(-1)
        if reduction == "none":
            return err
        if mask is not None:
            return (err * mask).sum() / mask.sum().clamp_min(1)
        return err.mean()


class TextHead(nn.Module):
    """Softmax over the byte/BPE vocabulary, optionally weight-tied."""

    def __init__(self, d_model: int, vocab_size: int, embedding: Optional[nn.Embedding] = None) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.tied = embedding is not None
        if embedding is not None:
            self.embedding = embedding
            self.proj = None
        else:
            self.embedding = None
            self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.norm(h)
        if self.proj is not None:
            return self.proj(h)
        return h @ self.embedding.weight.t()


class ActionHead(nn.Module):
    """UI / tool actuation: a discrete opcode plus continuous operands.

    An action is ``<OP, x, y, value, ...>``. The opcode is categorical; the
    operands are continuous and are only supervised for the operands that the
    opcode actually uses, which the caller signals through ``operand_mask``.
    Supervising unused operand slots teaches the model to predict padding.
    """

    def __init__(self, d_model: int, n_ops: int, n_scalars: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.op = nn.Linear(d_model, n_ops)
        self.scalars = nn.Linear(d_model, n_scalars)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm(h)
        return self.op(h), self.scalars(h)


class SlotTypeHead(nn.Module):
    """Which modality the *next* slot is. Free-running generation needs this."""

    def __init__(self, d_model: int, n_modalities: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, n_modalities)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(h))


class ConfidenceHead(nn.Module):
    """Calibrated probability that the emission at this position is correct.

    Trained with a binary cross-entropy against realized correctness, not
    against a self-reported feeling. That is the only version of this head that
    means anything: ``tests/training/test_calibration.py`` measures expected
    calibration error against a held-out split and fails if the head is worse
    than a constant predictor.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(h)).squeeze(-1)

    def loss(self, h: torch.Tensor, correct: torch.Tensor, mask: Optional[torch.Tensor] = None):
        logit = self.forward(h)
        err = F.binary_cross_entropy_with_logits(
            logit, correct.to(logit.dtype), reduction="none"
        )
        if mask is not None:
            return (err * mask).sum() / mask.sum().clamp_min(1)
        return err.mean()
