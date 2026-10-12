"""First implementation slice: one shared trunk, three native heads.

Architecture §20.3. This trains a genuinely small Iridium-1 configuration on
advection-diffusion episodes that combine a symbolic instruction, an exact
physical field, and a native continuous output.

What this file is:
  * one dense shared trunk over interleaved text and field blocks (no router)
  * a conservative face-flux correction head (§9.3)
  * a direct field-prediction head, as the ablation control (§18.4)
  * a native flow-matching image head for the diagnostic (§7.1)

What this file is not: a foundation model. It is a correctness slice at a scale
where every claim can be checked on a laptop CPU.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.advection import (
    DX,
    MAX_INSTRUCTION_BYTES,
    N_CELLS,
    N_FRAMES,
    Episode,
    encode_instruction,
)
from ..model.core import IridiumCore, RecurrencePolicy
from ..model.inventory import TransformerConfig

FIELD_PATCH = 4
N_FIELD_TOKENS = N_CELLS // FIELD_PATCH
IMAGE_PATCH_T, IMAGE_PATCH_X = 4, 8
N_IMAGE_TOKENS = (N_FRAMES // IMAGE_PATCH_T) * (N_CELLS // IMAGE_PATCH_X)
IMAGE_PATCH_DIM = IMAGE_PATCH_T * IMAGE_PATCH_X

SLICE_CONFIG = TransformerConfig(
    name="iridium-1-slice",
    d_model=128,
    n_prelude=1,
    n_core=2,
    n_coda=1,
    d_ff=352,
    n_query_heads=4,
    n_kv_heads=2,
    d_head=32,
)


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_size: int = 32
    lr: float = 3e-4
    recurrence: int = 2
    weight_flux: float = 1.0
    weight_direct: float = 1.0
    weight_image: float = 0.25
    seed: int = 0
    log_every: int = 250


class IridiumSlice(nn.Module):
    """One checkpoint. Integrated codecs and every head live inside it."""

    def __init__(self, cfg: TransformerConfig = SLICE_CONFIG, recurrence: int = 2):
        super().__init__()
        self.cfg = cfg
        self.recurrence = recurrence
        d = cfg.d_model

        # --- integrated codecs (Iridium-1 components, not external models) ---
        self.byte_embed = nn.Embedding(256, d)
        self.field_encode = nn.Linear(FIELD_PATCH, d)
        self.image_encode = nn.Linear(IMAGE_PATCH_DIM, d)

        # Modality and position conditioning. Text position, field position and
        # image position are separate coordinate kinds (§5.3), so they get
        # separate embedding tables rather than one shared index.
        self.modality_embed = nn.Embedding(3, d)
        self.text_pos = nn.Embedding(MAX_INSTRUCTION_BYTES, d)
        self.field_pos = nn.Embedding(N_FIELD_TOKENS, d)
        self.image_pos = nn.Embedding(N_IMAGE_TOKENS, d)

        # Noise time tau is NOT physical time. Distinct embedding, distinct axis.
        self.noise_time = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))

        self.trunk = IridiumCore(cfg, max_recurrence=max(4, recurrence))

        # --- native heads ---
        # Conservative head: predicts the time-integrated flux through each of
        # the N_CELLS periodic faces. Applied antisymmetrically, so mass cannot
        # be invented whatever the network outputs.
        self.flux_head = nn.Linear(d, FIELD_PATCH)
        # Direct head: predicts a per-cell increment. This is the control for
        # the §18.4 ablation, and it is deliberately a *fair* control: it sees
        # the same trunk output and also predicts a change from u0, so the only
        # difference between the two heads is the conservation constraint.
        self.direct_head = nn.Linear(d, FIELD_PATCH)
        # Native continuous image head: predicts the flow-matching velocity.
        self.image_head = nn.Linear(d, IMAGE_PATCH_DIM)

        # Both field heads start at zero, so an untrained model predicts
        # persistence rather than a large random divergence.
        for head in (self.flux_head, self.direct_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    # -- encoding -----------------------------------------------------------

    def encode(
        self,
        instruction: torch.Tensor,   # (B, MAX_INSTRUCTION_BYTES) int64
        u0: torch.Tensor,            # (B, N_CELLS)
        image_noised: torch.Tensor,  # (B, N_FRAMES, N_CELLS)
        tau: torch.Tensor,           # (B, 1)
    ) -> torch.Tensor:
        b = instruction.shape[0]
        device = instruction.device

        text = self.byte_embed(instruction) + self.text_pos(
            torch.arange(MAX_INSTRUCTION_BYTES, device=device)
        )
        text = text + self.modality_embed(torch.zeros(1, dtype=torch.long, device=device))

        field = self.field_encode(u0.view(b, N_FIELD_TOKENS, FIELD_PATCH))
        field = field + self.field_pos(torch.arange(N_FIELD_TOKENS, device=device))
        field = field + self.modality_embed(torch.ones(1, dtype=torch.long, device=device))

        patches = (
            image_noised.view(
                b, N_FRAMES // IMAGE_PATCH_T, IMAGE_PATCH_T,
                N_CELLS // IMAGE_PATCH_X, IMAGE_PATCH_X,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(b, N_IMAGE_TOKENS, IMAGE_PATCH_DIM)
        )
        image = self.image_encode(patches)
        image = image + self.image_pos(torch.arange(N_IMAGE_TOKENS, device=device))
        image = image + self.modality_embed(
            torch.full((1,), 2, dtype=torch.long, device=device)
        )
        image = image + self.noise_time(tau).unsqueeze(1)

        return torch.cat([text, field, image], dim=1)

    def forward(
        self,
        instruction: torch.Tensor,
        u0: torch.Tensor,
        image_noised: torch.Tensor,
        tau: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h = self.encode(instruction, u0, image_noised, tau)
        h, _ = self.trunk(h, mask=None, policy=RecurrencePolicy(self.recurrence))

        b = u0.shape[0]
        f0 = MAX_INSTRUCTION_BYTES
        f1 = f0 + N_FIELD_TOKENS
        field_h = h[:, f0:f1]
        image_h = h[:, f1:]

        face_correction = self.flux_head(field_h).reshape(b, N_CELLS)
        direct = u0 + self.direct_head(field_h).reshape(b, N_CELLS)

        velocity = (
            self.image_head(image_h)
            .reshape(
                b, N_FRAMES // IMAGE_PATCH_T, N_CELLS // IMAGE_PATCH_X,
                IMAGE_PATCH_T, IMAGE_PATCH_X,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(b, N_FRAMES, N_CELLS)
        )
        return {
            "face_correction": face_correction,
            "direct": direct,
            "image_velocity": velocity,
        }

    # -- conservative application ------------------------------------------

    @staticmethod
    def apply_flux(u0: torch.Tensor, face_flux: torch.Tensor) -> torch.Tensor:
        """Apply a learned face transfer antisymmetrically on a periodic mesh.

        ``u_i <- u_i - (T_{i+1/2} - T_{i-1/2})``

        ``face_flux[i]`` is the *transfer* through the face to the left of cell
        ``i``, already expressed in cell-value units: ``T = F dt / dx``. Folding
        ``dt/dx`` into the learned quantity is a conditioning choice, not a
        weakening of the constraint — each face still appears once with each
        sign, so the interior telescopes and the total is unchanged to
        floating-point precision whatever the network outputs.

        Recovering the dimensional flux for export is ``F = T dx / dt``.
        """
        divergence = torch.roll(face_flux, -1, dims=-1) - face_flux
        return u0 - divergence

    def predict_field(
        self, instruction: torch.Tensor, u0: torch.Tensor, mode: str = "flux"
    ) -> torch.Tensor:
        b = u0.shape[0]
        dummy = torch.zeros(b, N_FRAMES, N_CELLS, device=u0.device, dtype=u0.dtype)
        tau = torch.zeros(b, 1, device=u0.device, dtype=u0.dtype)
        out = self(instruction, u0, dummy, tau)
        if mode == "flux":
            return self.apply_flux(u0, out["face_correction"])
        if mode == "direct":
            return out["direct"]
        raise ValueError(f"unknown mode: {mode}")

    @torch.no_grad()
    def sample_image(
        self,
        instruction: torch.Tensor,
        u0: torch.Tensor,
        steps: int = 8,
        seed: int = 0,
    ) -> torch.Tensor:
        """Euler integration of the learned flow from noise to image (§7.1)."""
        generator = torch.Generator(device=u0.device).manual_seed(seed)
        b = u0.shape[0]
        x = torch.randn(
            b, N_FRAMES, N_CELLS, generator=generator,
            device=u0.device, dtype=u0.dtype,
        )
        dt = 1.0 / steps
        for i in range(steps):
            tau = torch.full((b, 1), i * dt, device=u0.device, dtype=u0.dtype)
            v = self(instruction, u0, x, tau)["image_velocity"]
            x = x + dt * v
        return x


# ---------------------------------------------------------------------------


def batch_from_episodes(episodes: list[Episode], device="cpu"):
    instruction = torch.tensor(
        np.stack([encode_instruction(e.instruction) for e in episodes]), device=device
    )
    u0 = torch.tensor(np.stack([e.u0 for e in episodes]), dtype=torch.float32, device=device)
    target = torch.tensor(
        np.stack([e.u_target for e in episodes]), dtype=torch.float32, device=device
    )
    image = torch.tensor(
        np.stack([e.diagnostic for e in episodes]), dtype=torch.float32, device=device
    )
    return instruction, u0, target, image


def train(
    train_episodes: list[Episode],
    cfg: TrainConfig = TrainConfig(),
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[IridiumSlice, list[dict]]:
    torch.manual_seed(cfg.seed)
    model = IridiumSlice(recurrence=cfg.recurrence).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.1
    )
    rng = np.random.default_rng(cfg.seed)
    history: list[dict] = []
    started = time.time()

    for step in range(cfg.steps):
        idx = rng.integers(0, len(train_episodes), size=cfg.batch_size)
        batch = [train_episodes[i] for i in idx]
        instruction, u0, target, image = batch_from_episodes(batch, device)

        # Conditional flow matching for the diagnostic image (§7.1).
        z0 = torch.randn_like(image)
        tau = torch.rand(image.shape[0], 1, device=device)
        tau_b = tau.view(-1, 1, 1)
        z_tau = (1.0 - tau_b) * z0 + tau_b * image
        flow_target = image - z0

        out = model(instruction, u0, z_tau, tau)
        pred_flux = model.apply_flux(u0, out["face_correction"])

        loss_flux = F.mse_loss(pred_flux, target)
        loss_direct = F.mse_loss(out["direct"], target)
        loss_image = F.mse_loss(out["image_velocity"], flow_target)
        loss = (
            cfg.weight_flux * loss_flux
            + cfg.weight_direct * loss_direct
            + cfg.weight_image * loss_image
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            record = {
                "step": step,
                "loss": loss.item(),
                "loss_flux": loss_flux.item(),
                "loss_direct": loss_direct.item(),
                "loss_image": loss_image.item(),
                "elapsed_s": round(time.time() - started, 1),
            }
            history.append(record)
            if verbose:
                print(
                    f"step {step:5d}  total {record['loss']:.5f}  "
                    f"flux {record['loss_flux']:.5f}  "
                    f"direct {record['loss_direct']:.5f}  "
                    f"image {record['loss_image']:.5f}  "
                    f"[{record['elapsed_s']}s]"
                )
    return model, history


# ---------------------------------------------------------------------------


def _nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Normalized RMSE against the target's own spread, per episode."""
    err = np.sqrt(np.mean((pred - target) ** 2, axis=-1))
    scale = np.std(target, axis=-1)
    return float(np.mean(err / np.maximum(scale, 1e-12)))


@torch.no_grad()
def evaluate(
    model: IridiumSlice, episodes: list[Episode], device: str = "cpu"
) -> dict[str, float]:
    from ..data.advection import persistence_baseline, upwind_reference

    model.eval()
    instruction, u0, target, _ = batch_from_episodes(episodes, device)

    # Accumulate in float64 so a reported "machine precision" drift is a claim
    # about the scheme and not about the storage dtype.
    flux_pred = model.predict_field(instruction, u0, "flux").cpu().numpy().astype(np.float64)
    direct_pred = (
        model.predict_field(instruction, u0, "direct").cpu().numpy().astype(np.float64)
    )
    target_np = np.stack([e.u_target for e in episodes])       # exact, float64
    u0_np = np.stack([e.u0 for e in episodes])                 # exact, float64

    persistence = np.stack([persistence_baseline(e.u0) for e in episodes])
    upwind = np.stack(
        [upwind_reference(e.u0, e.speed, e.diffusivity, e.horizon) for e in episodes]
    )

    # The heads consume the float32 tensor, so their conservation is measured
    # against that same input; the exact solution is measured in float64.
    mass0_fed = u0.cpu().numpy().astype(np.float64).sum(axis=-1) * DX
    mass0 = u0_np.sum(axis=-1) * DX
    mass_flux = flux_pred.sum(axis=-1) * DX
    mass_direct = direct_pred.sum(axis=-1) * DX

    return {
        "n_episodes": len(episodes),
        "nrmse_flux_head": _nrmse(flux_pred, target_np),
        "nrmse_direct_head": _nrmse(direct_pred, target_np),
        "nrmse_persistence": _nrmse(persistence, target_np),
        "nrmse_upwind_reference": _nrmse(upwind, target_np),
        "mass_drift_flux_head": float(
            np.max(np.abs(mass_flux - mass0_fed) / np.maximum(np.abs(mass0_fed), 1e-12))
        ),
        "mass_drift_direct_head": float(
            np.max(np.abs(mass_direct - mass0_fed) / np.maximum(np.abs(mass0_fed), 1e-12))
        ),
        "mass_drift_exact": float(
            np.max(
                np.abs(target_np.sum(axis=-1) * DX - mass0)
                / np.maximum(np.abs(mass0), 1e-12)
            )
        ),
    }


@torch.no_grad()
def image_quality_vs_steps(
    model: IridiumSlice, episodes: list[Episode], step_counts=(1, 2, 4, 8, 16, 32),
    device: str = "cpu",
) -> dict[int, float]:
    """§7.1: choose inference steps from measurement, not a fixed promise."""
    model.eval()
    instruction, u0, _, image = batch_from_episodes(episodes, device)
    truth = image.cpu().numpy()
    out: dict[int, float] = {}
    for steps in step_counts:
        sampled = model.sample_image(instruction, u0, steps=steps, seed=0).cpu().numpy()
        out[steps] = _nrmse(sampled.reshape(len(episodes), -1),
                            truth.reshape(len(episodes), -1))
    return out


def save_run(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
