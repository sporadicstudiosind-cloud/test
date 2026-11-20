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

A note on what this module can and cannot fix. Sample quality on a trained
checkpoint is overwhelmingly a function of training data and compute; nothing
below changes that. What *is* an architectural responsibility, and what this
module now owns, is not leaving quality on the table for reasons that have
nothing to do with data — an under-weighted training distribution over the
hard middle of the trajectory (see ``timestep_sampling``), a conditioning path
with less capacity than the literature has shown is available at the same
parameter cost (see ``conditioning="adaln"``), no way to trade sampling
compute for quality beyond one fixed integrator (see ``sample(integrator=)``),
and no classifier-free guidance path at all. These are the "for free" wins;
none of them are a substitute for training the model.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm


class FlowMatchingHead(nn.Module):
    """Predicts the conditional flow velocity for one continuous modality.

    Training: sample ``tau`` from ``timestep_sampling`` and ``x0 ~ N(0, I)``;
    form ``x_tau = (1-tau) x0 + tau x1``; regress ``v_theta(x_tau, tau, h)``
    onto the conditional velocity ``x1 - x0``. That target is the standard
    rectified / conditional flow, and its expectation is the marginal velocity
    field — see ``tests/unit/test_heads.py::test_gaussian_closed_form`` for a
    derivation-backed check that the trained head actually converges to it,
    rather than merely to *a* velocity field that reduces the loss.

    Sampling: integrate ``dx/dtau = v`` from ``x0`` with Euler, midpoint or
    Heun. Step count and integrator choice are measured quantities, never a
    promise: ``tests/unit/test_heads.py::test_step_count_quality_tradeoff``
    reports the actual error-versus-steps curve on a synthetic distribution
    with a known answer, rather than asserting one.

    Timestep sampling. ``tau ~ U(0,1)`` (the historical default, kept as
    ``timestep_sampling="uniform"``) spends as much training signal on the
    easy, nearly-linear ends of the trajectory (tau near 0 or 1, where the
    optimal velocity is close to a constant, data-independent direction) as on
    the hard middle (tau near 0.5, where the velocity field has to resolve
    which mode of a highly multimodal target the sample is heading toward).
    SD3 (Esser et al., 2024, "Scaling Rectified Flow Transformers for
    High-Resolution Image Synthesis") measured this directly and found
    logit-normal sampling — draw ``u ~ N(mean, std)``, set ``tau = sigmoid(u)``
    — outperforms uniform, several alternative weightings, and EDM-style
    sampling at matched compute, because it concentrates mass exactly where
    the regression problem is hardest. ``timestep_sampling="logit_normal"``
    implements it; ``logit_normal_mean``/``logit_normal_std`` are the
    concentration knobs (0.0/1.0 reproduces the SD3 default). Default stays
    ``"uniform"`` so existing checkpoints and training curves are unaffected.

    Resolution-dependent shift. The same paper's other observation: a
    higher-resolution (more-patches) target needs *more* of its schedule spent
    at high noise, because high-frequency detail is destroyed by noise faster
    than low-frequency structure — a 4x4 patch grid and a 64x64 patch grid are
    not the same denoising problem at the same nominal tau. The fix is a
    monotone reparameterisation ``tau' = s*tau / (1 + (s-1)*tau)`` with
    ``s = sqrt(n_patches / resolution_shift_base)`` (s>1 shifts mass toward
    tau=1, i.e. more high-noise steps); ``resolution_shift=True`` applies it to
    both the training-time tau draw and the sampling-time integration
    schedule, keyed off an explicit ``n_patches`` the caller passes (this head
    has no other way to know the patch count of what it's being asked to
    denoise — inferring it from tensor shape would silently mis-shift whenever
    the caller flattens or batches patches together). Default ``False``.

    Timestep conditioning. ``_tau_features`` turns scalar tau into a Fourier
    embedding the way a transformer turns integer position into one. The
    naive failure mode here — and it is exactly the "badly chosen frequency
    range" mentioned in this module's issue tracker — is coupling the
    embedding's *frequency range* to the embedding's *width* (``n_tau``): the
    previous implementation spanned exactly [2^0, 2^8] regardless of how many
    frequency bands ``n_tau`` requested, so raising ``n_tau`` bought only
    denser sampling of the same fixed octave range (diminishing returns, and
    for small ``n_tau`` an unnecessarily coarse top frequency that cannot
    resolve two taus 1/steps apart late in a many-step schedule). The fixed
    range is kept as the literal default (``tau_feature_max_octave=8``) so
    behaviour is unchanged with ``n_tau=64`` today; the width-coupled
    alternative is available as ``tau_feature_kind="scaled"``, which spans
    ``[2^0, 2^(log2(n_tau)+2)]`` — i.e. the top frequency now has a period
    about 4x the spacing between adjacent bands, so bands stay
    well-separated and useful as ``n_tau`` grows instead of redundant.
    """

    def __init__(
        self,
        d_model: int,
        d_out: int,
        d_hidden: int = 0,
        n_tau: int = 64,
        *,
        conditioning: str = "add",
        tau_feature_kind: str = "fixed",
        tau_feature_max_octave: float = 8.0,
        timestep_sampling: str = "uniform",
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
        resolution_shift: bool = False,
        resolution_shift_base: int = 256,
        cfg_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if conditioning not in ("add", "adaln"):
            raise ValueError(f"unknown conditioning: {conditioning!r}")
        if tau_feature_kind not in ("fixed", "scaled"):
            raise ValueError(f"unknown tau_feature_kind: {tau_feature_kind!r}")
        if timestep_sampling not in ("uniform", "logit_normal"):
            raise ValueError(f"unknown timestep_sampling: {timestep_sampling!r}")
        d_hidden = d_hidden or max(d_out * 2, d_model)
        self.d_out = d_out
        self.n_tau = n_tau
        self.conditioning = conditioning
        self.tau_feature_kind = tau_feature_kind
        self.tau_feature_max_octave = tau_feature_max_octave
        self.timestep_sampling = timestep_sampling
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std
        self.resolution_shift = resolution_shift
        self.resolution_shift_base = resolution_shift_base
        self.cfg_dropout = cfg_dropout

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

        # Classifier-free guidance needs an explicit "no conditioning" input to
        # contrast against at sampling time; a zero vector would work only if
        # nothing else in the network ever produces exactly zero (it does, at
        # init, before ``cond`` learns a nonzero bias). A learned null embedding
        # is the standard fix (Ho & Salimans, 2022) and costs d_model floats.
        self.null_cond = nn.Parameter(torch.zeros(d_model))

        if self.conditioning == "adaln":
            # AdaLN(-Zero) modulation, as in DiT (Peebles & Xie, 2023) and SD3's
            # MM-DiT block: conditioning (here, tau + h jointly) predicts a
            # per-channel shift and scale applied to the normalized data
            # branch, rather than being added into it. This gives the
            # conditioning information a *multiplicative* channel gate in
            # addition to an additive one, which is the standard explanation
            # for why AdaLN conditioning outperforms plain concatenation/
            # addition at matched parameter count in generative transformers.
            # Zero-initializing the modulation (shift=scale=0 at init) makes
            # this branch start out as a plain LayerNorm — identical in kind
            # to the existing zero-init on ``out``, so adding this option does
            # not destabilize the start of training.
            self.ada_norm = nn.LayerNorm(d_hidden, elementwise_affine=False)
            self.ada_mod = nn.Linear(d_hidden, 2 * d_hidden)
            nn.init.zeros_(self.ada_mod.weight)
            nn.init.zeros_(self.ada_mod.bias)

    def _tau_features(self, tau: torch.Tensor) -> torch.Tensor:
        if self.tau_feature_kind == "scaled":
            max_octave = math.log2(max(self.n_tau, 2)) + 2.0
        else:
            max_octave = self.tau_feature_max_octave
        freqs = torch.linspace(0, max_octave, self.n_tau, device=tau.device, dtype=torch.float32)
        ang = tau.float().unsqueeze(-1) * (2.0 ** freqs) * torch.pi
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    def _shift_tau(self, tau, n_patches: Optional[int]):
        """Works on both tensors (training-time tau draws) and plain floats
        (the sampling-time integration grid, computed once outside autograd)."""
        if not self.resolution_shift or n_patches is None:
            return tau
        shift = (float(n_patches) / float(self.resolution_shift_base)) ** 0.5
        shift = max(shift, 1e-6)
        return shift * tau / (1.0 + (shift - 1.0) * tau)

    def _sample_tau(
        self,
        shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator],
        n_patches: Optional[int],
    ) -> torch.Tensor:
        if self.timestep_sampling == "uniform":
            tau = torch.rand(shape, device=device, dtype=dtype, generator=generator)
        else:  # logit_normal
            u = torch.randn(shape, device=device, dtype=dtype, generator=generator)
            tau = torch.sigmoid(u * self.logit_normal_std + self.logit_normal_mean)
        return self._shift_tau(tau, n_patches)

    def velocity(
        self,
        h: torch.Tensor,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
        *,
        cond_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``cond_override`` replaces ``norm(h)`` (e.g. with ``null_cond``,
        broadcast to the same shape) for classifier-free guidance dropout at
        training time and guided sampling at inference time; it is never used
        on the default path."""
        cond_in = cond_override if cond_override is not None else self.norm(h)
        cond_feat = self.cond(cond_in)
        tau_feat = self.tau_embed(self._tau_features(tau))
        cond_emb = cond_feat + tau_feat
        if self.conditioning == "add":
            z = cond_emb + self.x_in(x_tau)
            return self.out(self.mix(z))
        # adaln
        x = self.ada_norm(self.x_in(x_tau))
        shift, scale = self.ada_mod(cond_emb).chunk(2, dim=-1)
        x = x * (1.0 + scale) + shift
        return self.out(self.mix(x))

    def loss(
        self,
        h: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        reduction: str = "mean",
        n_patches: Optional[int] = None,
    ) -> torch.Tensor:
        x0 = torch.randn(target.shape, device=target.device, dtype=target.dtype,
                         generator=generator)
        tau = self._sample_tau(target.shape[:-1], target.device, target.dtype,
                               generator, n_patches)
        x_tau = (1.0 - tau.unsqueeze(-1)) * x0 + tau.unsqueeze(-1) * target
        v_target = target - x0
        cond_override = None
        if self.cfg_dropout > 0.0:
            # Condition dropout (Ho & Salimans, 2022): at some positions,
            # train the head as if it had received no conditioning at all, so
            # it learns both p(x1|h) (conditional velocity) and p(x1) (the
            # marginal one, from the null embedding). Guidance at sampling
            # time extrapolates *away* from the unconditional velocity toward
            # the conditional one; that extrapolation is only meaningful if
            # the unconditional branch was actually trained, not just
            # initialized. Dropout is per generative-position (matches
            # tau's granularity), not per-batch, so a single training batch
            # updates both branches.
            drop = torch.rand(h.shape[:-1], device=h.device, generator=generator) < self.cfg_dropout
            null = self.null_cond.to(dtype=h.dtype, device=h.device)
            null = null.expand(h.shape[:-1] + (null.shape[-1],))
            cond_override = torch.where(drop.unsqueeze(-1), null, self.norm(h))
        v_pred = self.velocity(h, x_tau, tau, cond_override=cond_override)
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
        *,
        integrator: Optional[str] = None,
        guidance_scale: float = 1.0,
        n_patches: Optional[int] = None,
    ) -> torch.Tensor:
        """Integrate ``dx/dtau = v`` from noise to data.

        ``integrator`` in {"euler", "midpoint", "heun"}; ``None`` (the
        default) maps the legacy ``midpoint`` bool to "midpoint"/"euler" so
        every existing call site is unaffected. Euler is first-order (global
        error O(1/steps)); midpoint and Heun are both second-order
        (O(1/steps^2)) at 2x the network evaluations per step, which is
        usually the better trade at low step counts — see
        ``test_step_count_quality_tradeoff`` for measured numbers rather than
        this claim on its own. Heun (explicit trapezoidal / RK2) evaluates the
        velocity at the current point and at the Euler-predicted next point
        and averages; midpoint evaluates at the Euler-predicted *midpoint* and
        uses that alone. They agree to first order and differ in their
        second-order error constant; neither dominates the other for every
        velocity field, which is why both are offered rather than only one.

        ``guidance_scale`` != 1.0 requires the head to have been trained with
        ``cfg_dropout`` > 0 (an untrained null branch makes guidance
        directionless). The guided velocity is
        ``v_uncond + guidance_scale * (v_cond - v_uncond)``: an extrapolation
        *past* the conditional prediction, away from the unconditional one.
        This is classifier-free guidance (Ho & Salimans, 2022); its known
        failure mode is overshoot — at high scale (roughly >4-6 in the
        image-diffusion literature, and problem-dependent here) the
        extrapolated velocity routinely pushes samples off the data manifold
        the head was trained on, producing the oversaturated, over-sharpened,
        "AI slop" look precisely *because* it is an extrapolation and not an
        interpolation. There is no scale that is safe in general; it is a
        quality/adherence knob to be tuned per checkpoint, not a constant to
        hard-code. Default 1.0 disables guidance (single forward per step,
        identical to today).
        """
        if integrator is None:
            integrator = "midpoint" if midpoint else "euler"
        if integrator not in ("euler", "midpoint", "heun"):
            raise ValueError(f"unknown integrator: {integrator!r}")
        guided = guidance_scale != 1.0

        def v_of(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
            v_cond = self.velocity(h, x, tau)
            if not guided:
                return v_cond
            null = self.null_cond.to(dtype=h.dtype, device=h.device)
            null = null.expand(h.shape[:-1] + (null.shape[-1],))
            v_uncond = self.velocity(h, x, tau, cond_override=null)
            return v_uncond + guidance_scale * (v_cond - v_uncond)

        shape = h.shape[:-1] + (self.d_out,)
        x = torch.randn(shape, device=h.device, dtype=h.dtype, generator=generator)
        grid = [self._shift_tau(i / steps, n_patches) for i in range(steps + 1)]
        for i in range(steps):
            t0, t1 = grid[i], grid[i + 1]
            dt = t1 - t0
            tau = torch.full(h.shape[:-1], t0, device=h.device, dtype=h.dtype)
            if integrator == "euler":
                v = v_of(x, tau)
            elif integrator == "midpoint":
                v1 = v_of(x, tau)
                x_mid = x + 0.5 * dt * v1
                v = v_of(x_mid, tau + 0.5 * dt)
            else:  # heun
                v1 = v_of(x, tau)
                x_pred = x + dt * v1
                tau1 = torch.full(h.shape[:-1], t1, device=h.device, dtype=h.dtype)
                v2 = v_of(x_pred, tau1)
                v = 0.5 * (v1 + v2)
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
